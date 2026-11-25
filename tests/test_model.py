import unittest

import torch

from src.model import (
    AudioFirstSiameseEncoder,
    AudioOnlySiameseEncoder,
    SiameseEncoder,
    contrastive_loss,
)


class SiameseEncoderTests(unittest.TestCase):
    def test_outputs_are_64d_unit_embeddings(self):
        encoder = SiameseEncoder(num_genres=4)
        bpm = torch.tensor([[90.0], [120.0]])
        lyrics = torch.randn(2, 5000)
        genres = torch.eye(4)[:2]

        embeddings = encoder(bpm, lyrics, genres)

        self.assertEqual(embeddings.shape, (2, 64))
        torch.testing.assert_close(
            torch.linalg.vector_norm(embeddings, dim=1), torch.ones(2)
        )

    def test_shared_encoder_gives_symmetric_pair_score(self):
        encoder = SiameseEncoder(num_genres=4)
        left = (torch.tensor([[90.0]]), torch.randn(1, 5000), torch.tensor([[1.0, 0, 0, 0]]))
        right = (torch.tensor([[130.0]]), torch.randn(1, 5000), torch.tensor([[0.0, 1, 0, 0]]))

        left_embedding = encoder(*left)
        right_embedding = encoder(*right)
        forward_score = torch.sum(left_embedding * right_embedding, dim=1)
        reverse_score = torch.sum(right_embedding * left_embedding, dim=1)

        torch.testing.assert_close(forward_score, reverse_score)


class AudioFirstSiameseEncoderTests(unittest.TestCase):
    def _inputs(self, batch_size=2):
        return (
            torch.randn(batch_size, 1024, 28),
            torch.randn(batch_size, 1),
            torch.randn(batch_size, 5000),
            torch.eye(4)[:batch_size],
        )

    def test_outputs_are_finite_64d_unit_embeddings(self):
        encoder = AudioFirstSiameseEncoder(num_genres=4)

        embeddings = encoder(*self._inputs())

        self.assertEqual(embeddings.shape, (2, 64))
        self.assertTrue(torch.isfinite(embeddings).all())
        torch.testing.assert_close(
            torch.linalg.vector_norm(embeddings, dim=1), torch.ones(2)
        )

    def test_eval_forward_is_deterministic(self):
        encoder = AudioFirstSiameseEncoder(num_genres=4).eval()
        inputs = self._inputs()

        first = encoder(*inputs)
        second = encoder(*inputs)

        torch.testing.assert_close(first, second)

    def test_cpu_backward_reaches_audio_and_all_auxiliary_branches(self):
        torch.manual_seed(0)
        encoder = AudioFirstSiameseEncoder(num_genres=4)
        left_inputs = tuple(value.requires_grad_() for value in self._inputs())
        right_inputs = tuple(value.detach() + 0.01 * torch.randn_like(value) for value in left_inputs)

        left_embedding = encoder(*left_inputs)
        right_embedding = encoder(*right_inputs)
        loss = contrastive_loss(
            left_embedding, right_embedding, torch.ones(2)
        )
        loss.backward()

        for name, value in zip(("audio", "bpm", "lyrics", "genre"), left_inputs):
            with self.subTest(input=name):
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertGreater(value.grad.abs().sum().item(), 0)


class AudioOnlySiameseEncoderTests(unittest.TestCase):
    def test_outputs_are_finite_32d_unit_embeddings(self):
        encoder = AudioOnlySiameseEncoder()

        embeddings = encoder(torch.randn(2, 1024, 28))

        self.assertEqual(embeddings.shape, (2, 32))
        self.assertTrue(torch.isfinite(embeddings).all())
        torch.testing.assert_close(
            torch.linalg.vector_norm(embeddings, dim=1), torch.ones(2)
        )

    def test_eval_forward_is_deterministic(self):
        encoder = AudioOnlySiameseEncoder().eval()
        audio = torch.randn(2, 1024, 28)

        first = encoder(audio)
        second = encoder(audio)

        torch.testing.assert_close(first, second)

    def test_same_encoder_instance_encodes_both_pair_sides(self):
        encoder = AudioOnlySiameseEncoder()
        left = torch.randn(2, 1024, 28)
        right = torch.randn(2, 1024, 28)

        left_embedding = encoder(left)
        right_embedding = encoder(right)

        self.assertEqual(left_embedding.shape, right_embedding.shape)
        self.assertTrue(torch.isfinite(left_embedding).all())
        self.assertTrue(torch.isfinite(right_embedding).all())

    def test_positive_pair_loss_has_nonzero_finite_input_gradients(self):
        torch.manual_seed(0)
        encoder = AudioOnlySiameseEncoder()
        left = torch.randn(2, 1024, 28, requires_grad=True)
        right = left.detach() + 0.1 * torch.randn_like(left)

        left_embedding = encoder(left)
        right_embedding = encoder(right)
        loss = contrastive_loss(
            left_embedding, right_embedding, torch.ones(2)
        )
        loss.backward()

        self.assertGreater(loss.item(), 0)
        self.assertIsNotNone(left.grad)
        self.assertTrue(torch.isfinite(left.grad).all())
        self.assertGreater(left.grad.abs().sum().item(), 0)


class ContrastiveLossTests(unittest.TestCase):
    def test_matches_hand_calculation(self):
        left = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        right = torch.tensor([[3.0, 4.0], [0.0, 0.5]])
        labels = torch.tensor([1.0, 0.0])

        loss = contrastive_loss(left, right, labels, margin=1.0)

        # Positive: 1 * 5^2. Negative: (1 - 0.5)^2. Mean: (25 + .25) / 2.
        torch.testing.assert_close(loss, torch.tensor(12.625))

    def test_rejects_non_binary_labels(self):
        left = torch.zeros(1, 2)
        right = torch.ones(1, 2)

        for label in (0.5, -1.0):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "labels must contain only 0 or 1"):
                    contrastive_loss(left, right, torch.tensor([label]))

    def test_rejects_complex_labels(self):
        with self.assertRaisesRegex(ValueError, "labels must be real-valued"):
            contrastive_loss(
                torch.zeros(1, 2),
                torch.ones(1, 2),
                torch.tensor([1 + 2j]),
            )

    def test_rejects_non_finite_real_labels(self):
        left = torch.zeros(1, 2)
        right = torch.ones(1, 2)

        for label in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "labels must contain only 0 or 1"):
                    contrastive_loss(left, right, torch.tensor([label]))

    def test_backward_has_finite_gradients(self):
        encoder = SiameseEncoder(num_genres=4)
        left = (torch.randn(2, 1), torch.randn(2, 5000), torch.eye(4)[:2])
        right = (torch.randn(2, 1), torch.randn(2, 5000), torch.eye(4)[2:])
        left_embedding = encoder(*left)
        right_embedding = encoder(*right)

        loss = contrastive_loss(
            left_embedding, right_embedding, torch.tensor([1.0, 0.0])
        )
        loss.backward()

        gradients = [parameter.grad for parameter in encoder.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
