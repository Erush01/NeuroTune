import unittest

import torch

from src.model_v1 import NeuroTuneV1Encoder


class NeuroTuneV1EncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(684)
        self.batch = 3
        self.genres = 4
        self.audio = torch.randn(self.batch, 5, 26)
        self.mask = torch.tensor(
            [[True, True, True, True, True], [True, True, False, False, False], [True, False, False, False, False]],
            dtype=torch.bool,
        )
        self.modalities = {
            "bpm": torch.randn(self.batch, 1),
            "lyrics": torch.randn(self.batch, 5000),
            "genre": torch.randn(self.batch, self.genres),
        }

    def inputs_for(self, variant, audio=None, mask=None):
        inputs = {"audio": self.audio if audio is None else audio, "mask": self.mask if mask is None else mask}
        if variant in {"B", "C", "D"}:
            inputs["bpm"] = self.modalities["bpm"]
        if variant in {"C", "D"}:
            inputs["genre"] = self.modalities["genre"]
        if variant == "D":
            inputs["lyrics"] = self.modalities["lyrics"]
        return inputs

    def test_active_modalities_property(self):
        expected = {
            "A": (),
            "B": ("bpm",),
            "C": ("bpm", "genre"),
            "D": ("bpm", "genre", "lyrics"),
        }
        for variant, modalities in expected.items():
            with self.subTest(variant=variant):
                self.assertEqual(NeuroTuneV1Encoder(variant, self.genres).active_modalities, modalities)

    def test_fusion_input_dimensions(self):
        audio_only = NeuroTuneV1Encoder("A", self.genres)
        self.assertFalse(hasattr(audio_only, "fusion"))

        expected_inputs = {"B": 72, "C": 88, "D": 120}
        for variant, input_width in expected_inputs.items():
            with self.subTest(variant=variant):
                model = NeuroTuneV1Encoder(variant, self.genres)
                self.assertEqual(model.fusion[0].in_features, input_width)

    def test_variants_output_valid_unit_embeddings_including_length_one(self):
        for variant in "ABCD":
            with self.subTest(variant=variant):
                model = NeuroTuneV1Encoder(variant, self.genres)
                inputs = self.inputs_for(variant)
                result = model(**inputs)
                self.assertEqual(result.shape, (self.batch, 64))
                self.assertTrue(torch.isfinite(result).all())
                torch.testing.assert_close(result.norm(dim=1), torch.ones(self.batch), atol=1e-6, rtol=1e-6)
                one = model(**self.inputs_for(variant, self.audio[:, :1], self.mask[:, :1]))
                self.assertEqual(one.shape, (self.batch, 64))
                self.assertTrue(torch.isfinite(one).all())

    def test_rejects_invalid_variant_and_modality_presence(self):
        with self.assertRaises(ValueError):
            NeuroTuneV1Encoder("E", self.genres)
        for variant in "ABCD":
            inputs = self.inputs_for(variant)
            expected = set(inputs) - {"audio", "mask"}
            for key in expected:
                with self.subTest(variant=variant, missing=key):
                    missing = dict(inputs)
                    del missing[key]
                    with self.assertRaises(ValueError):
                        NeuroTuneV1Encoder(variant, self.genres)(**missing)
            for key in set(self.modalities) - expected:
                with self.subTest(variant=variant, unexpected=key):
                    unexpected = dict(inputs)
                    unexpected[key] = self.modalities[key]
                    with self.assertRaises(ValueError):
                        NeuroTuneV1Encoder(variant, self.genres)(**unexpected)

    def test_rejects_bad_dimensions_and_invalid_masks(self):
        model = NeuroTuneV1Encoder("D", self.genres)
        base = self.inputs_for("D")
        for key, value in (
            ("audio", torch.randn(self.batch, 5, 25)),
            ("bpm", torch.randn(self.batch, 2)),
            ("lyrics", torch.randn(self.batch, 4999)),
            ("genre", torch.randn(self.batch, self.genres + 1)),
        ):
            invalid = dict(base)
            invalid[key] = value
            with self.subTest(dimension=key), self.assertRaises(ValueError):
                model(**invalid)
        for invalid_mask in (
            torch.ones(self.batch, 5),
            torch.ones(self.batch, 4, dtype=torch.bool),
            torch.tensor([[True] * 5, [False] * 5, [True] * 5]),
        ):
            invalid = dict(base)
            invalid["mask"] = invalid_mask
            with self.subTest(mask=invalid_mask.shape), self.assertRaises(ValueError):
                model(**invalid)

    def test_padding_invariance_with_convolution_biases(self):
        model = NeuroTuneV1Encoder("A", self.genres)
        short = self.audio[1:2, :2]
        solo = model(short, torch.ones(1, 2, dtype=torch.bool))
        batched_audio = torch.cat(
            (torch.cat((short, torch.randn(1, 7, 26)), dim=1), torch.randn(1, 9, 26)), dim=0
        )
        batched_mask = torch.tensor([[True, True] + [False] * 7, [True] * 9], dtype=torch.bool)
        padded = model(batched_audio, batched_mask)[0:1]
        extra_audio = torch.cat(
            (torch.cat((short, torch.randn(1, 13, 26)), dim=1), torch.randn(1, 15, 26)), dim=0
        )
        extra_mask = torch.tensor([[True, True] + [False] * 13, [True] * 15], dtype=torch.bool)
        extra_padded = model(extra_audio, extra_mask)[0:1]
        torch.testing.assert_close(solo, padded, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(solo, extra_padded, atol=1e-6, rtol=1e-6)

    def test_rejects_non_float32_audio_and_incompatible_modalities(self):
        audio_model = NeuroTuneV1Encoder("A", self.genres)
        with self.assertRaisesRegex(ValueError, "audio.*float32"):
            audio_model(self.audio.double(), self.mask)

        model = NeuroTuneV1Encoder("B", self.genres)
        with self.assertRaisesRegex(ValueError, "bpm.*dtype"):
            model(self.audio, self.mask, bpm=self.modalities["bpm"].double())

        try:
            mismatched_device = torch.empty(self.batch, 1, dtype=torch.float32, device="meta")
        except (RuntimeError, NotImplementedError):
            self.skipTest("meta tensors are unavailable")
        with self.assertRaisesRegex(ValueError, "bpm.*device"):
            model(self.audio, self.mask, bpm=mismatched_device)

    def test_triplet_forward_shares_weights_and_backpropagates_all_active_branches(self):
        for variant in "ABCD":
            with self.subTest(variant=variant):
                model = NeuroTuneV1Encoder(variant, self.genres)
                args = []
                for _ in range(3):
                    inputs = self.inputs_for(variant)
                    inputs["audio"] = inputs["audio"] + torch.randn_like(inputs["audio"]) * 0.3
                    for key in ("bpm", "lyrics", "genre"):
                        if key in inputs:
                            inputs[key] = inputs[key] + torch.randn_like(inputs[key]) * 0.3
                    args.append(inputs)
                anchor, positive, negative = model.forward_triplets(*args)
                self.assertEqual(anchor.shape, (self.batch, 64))
                loss = ((anchor - positive) ** 2).mean() + ((anchor - negative) ** 2).mean()
                loss.backward()
                active = ["audio_branch"]
                if variant in {"B", "C", "D"}:
                    active.append("bpm_branch")
                if variant in {"C", "D"}:
                    active.append("genre_branch")
                if variant == "D":
                    active.append("lyrics_branch")
                for branch in active:
                    grads = [p.grad for p in getattr(model, branch).parameters()]
                    self.assertTrue(grads and all(g is not None for g in grads), branch)
                    self.assertTrue(any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), branch)

    def test_triplet_accepts_collator_metadata(self):
        for variant in "ABCD":
            with self.subTest(variant=variant):
                model = NeuroTuneV1Encoder(variant, self.genres)
                inputs = {
                    **self.inputs_for(variant),
                    "bpm": self.modalities["bpm"],
                    "lyrics": self.modalities["lyrics"],
                    "genre": self.modalities["genre"],
                    "lengths": torch.tensor([5, 2, 1]),
                    "track_ids": ["a", "b", "c"],
                }
                results = model.forward_triplets(inputs, inputs, inputs)
                self.assertEqual(len(results), 3)
                self.assertEqual(results[0].shape, (self.batch, 64))

    def test_auxiliary_keep_masks_zero_or_preserve_embeddings_without_affecting_audio_or_padding(self):
        model = NeuroTuneV1Encoder("D", self.genres)
        for branch in (model.bpm_branch, model.genre_branch, model.lyrics_branch):
            branch[0].bias.data.fill_(1)
        inputs = self.inputs_for("D")
        captured = {}
        for name in ("bpm_branch", "genre_branch", "lyrics_branch"):
            getattr(model, name).register_forward_hook(
                lambda _module, _args, output, name=name: captured.setdefault(name, output.detach())
            )
        fused = []
        model.fusion.register_forward_pre_hook(lambda _module, args: fused.append(args[0].detach()))
        model(**inputs)
        unmasked_audio = fused[-1][:, :64].clone()
        unmasked_branches = {name: value.clone() for name, value in captured.items()}

        masks = {
            "bpm": torch.tensor([False, True, False]),
            "genre": torch.tensor([True, False, True]),
            "lyrics": torch.tensor([False, True, True]),
        }
        model(**inputs, keep_masks=masks)
        offsets = {"bpm_branch": (64, 72), "genre_branch": (72, 88), "lyrics_branch": (88, 120)}
        names = {"bpm": "bpm_branch", "genre": "genre_branch", "lyrics": "lyrics_branch"}
        for modality, name in names.items():
            start, end = offsets[name]
            keep = masks[modality]
            self.assertTrue(unmasked_branches[name].abs().sum() > 0)
            torch.testing.assert_close(fused[-1][:, start:end][~keep], torch.zeros_like(fused[-1][:, start:end][~keep]))
            torch.testing.assert_close(fused[-1][:, start:end][keep], unmasked_branches[name][keep])
        torch.testing.assert_close(fused[-1][:, :64], unmasked_audio)

        # Padding and audio inputs are independent of auxiliary keep decisions.
        changed_audio = inputs["audio"].clone()
        changed_audio[~self.mask] = 1e6
        model(**{**inputs, "audio": changed_audio}, keep_masks=masks)
        torch.testing.assert_close(fused[-1][:, :64], unmasked_audio)

    def test_default_forward_is_unchanged_and_keep_masks_validate(self):
        model = NeuroTuneV1Encoder("D", self.genres)
        inputs = self.inputs_for("D")
        expected = model(**inputs)
        torch.testing.assert_close(model(**inputs, keep_masks=None), expected, rtol=0, atol=0)
        invalid_masks = (
            {"unknown": torch.ones(self.batch, dtype=torch.bool)},
            {"audio": torch.ones(self.batch, dtype=torch.bool)},
            {"bpm": torch.ones(self.batch, dtype=torch.float32)},
            {"bpm": torch.ones(self.batch - 1, dtype=torch.bool)},
        )
        for keep_masks in invalid_masks:
            with self.subTest(keep_masks=keep_masks.keys()), self.assertRaises(ValueError):
                model(**inputs, keep_masks=keep_masks)
        with self.assertRaisesRegex(ValueError, "cannot be masked"):
            NeuroTuneV1Encoder("B", self.genres)(
                **self.inputs_for("B"), keep_masks={"genre": torch.ones(self.batch, dtype=torch.bool)}
            )
        try:
            wrong_device = torch.ones(self.batch, dtype=torch.bool, device="meta")
        except (RuntimeError, NotImplementedError):
            self.skipTest("meta tensors are unavailable")
        with self.assertRaisesRegex(ValueError, "same device"):
            model(**inputs, keep_masks={"bpm": wrong_device})

    def test_triplet_role_specific_keep_masks_are_routed(self):
        model = NeuroTuneV1Encoder("D", self.genres)
        roles = [self.inputs_for("D") for _ in range(3)]
        role_masks = [
            {"bpm": torch.tensor([False, True, True])},
            {"genre": torch.tensor([True, False, True])},
            {"lyrics": torch.tensor([True, True, False])},
        ]
        actual = model.forward_triplets(
            *roles,
            anchor_keep_masks=role_masks[0],
            positive_keep_masks=role_masks[1],
            negative_keep_masks=role_masks[2],
        )
        expected = tuple(model(**inputs, keep_masks=masks) for inputs, masks in zip(roles, role_masks))
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)

    def test_d4_uses_shared_encoder_and_means_valid_raw_window_projections(self):
        model = NeuroTuneV1Encoder("D", self.genres)
        windows = torch.randn(2, 3, 5, 26)
        mask = torch.tensor([
            [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
        ], dtype=torch.bool)
        window_mask = torch.tensor([[True, True, False], [True, False, True]])
        captured = []
        model.fusion.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach()))
        conv_calls = []
        hook = model.audio_branch[0].register_forward_hook(lambda _m, args, _out: conv_calls.append(args[0].shape[0]))
        result = model(audio_windows=windows, mask=mask, window_mask=window_mask, **{
            key: self.modalities[key][:2] for key in ("bpm", "genre", "lyrics")
        })
        hook.remove()

        expected = []
        for track in range(2):
            projected = []
            for window in range(3):
                if window_mask[track, window]:
                    valid = mask[track, window]
                    features = windows[track, window].masked_fill(~valid[:, None], 0).T[None]
                    for conv in model.audio_branch:
                        features = torch.relu(conv(features)).masked_fill(~valid[None, None], 0)
                    pooled = features.sum(dim=2) / valid.sum()
                    projected.append(model.audio_projection(pooled)[0])
            expected.append(torch.stack(projected).mean(dim=0))
        torch.testing.assert_close(captured[0][:, :64], torch.stack(expected))
        self.assertEqual(sum(conv_calls), int(window_mask.sum()))
        self.assertEqual(result.shape, (2, 64))
        torch.testing.assert_close(result.norm(dim=1), torch.ones(2))

    def test_d4_ignores_invalid_windows_validates_masks_and_backpropagates(self):
        model = NeuroTuneV1Encoder("D", self.genres)
        windows = torch.randn(2, 3, 4, 26, requires_grad=True)
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        mask[:, 0, :2] = True
        window_mask = torch.tensor([[True, False, False], [True, False, False]])
        aux = {key: value[:2] for key, value in self.modalities.items()}
        result = model(audio_windows=windows, mask=mask, window_mask=window_mask, **aux)
        changed = windows.detach().clone()
        changed[:, 1:] = 1e6
        ignored = model(audio_windows=changed, mask=mask, window_mask=window_mask, **aux)
        torch.testing.assert_close(result, ignored)
        result.sum().backward()
        self.assertTrue(windows.grad is not None and torch.isfinite(windows.grad).all())
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.audio_branch.parameters()))
        with self.assertRaises(ValueError):
            model(audio_windows=windows.detach(), mask=mask, window_mask=torch.zeros_like(window_mask), **aux)
        inconsistent = mask.clone()
        inconsistent[:, 1, 0] = True
        with self.assertRaises(ValueError):
            model(audio_windows=windows.detach(), mask=inconsistent, window_mask=window_mask, **aux)
        with self.assertRaises(ValueError):
            model(audio_windows=windows.detach().double(), mask=mask, window_mask=window_mask, **aux)
        with self.assertRaises(ValueError):
            model(audio_windows=windows.detach()[:, :2], mask=mask[:, :2],
                  window_mask=window_mask[:, :2], **aux)
        with self.assertRaises(ValueError):
            model(audio_windows=windows.detach(), mask=mask, window_mask=window_mask, **aux,
                  keep_masks={"bpm": torch.ones(1, dtype=torch.bool)})

    def test_d4_triplets_accept_window_collator_keys_and_d0_path_is_unchanged(self):
        model = NeuroTuneV1Encoder("D", self.genres)
        inputs = {
            "audio_windows": self.audio[:, None].expand(-1, 3, -1, -1).clone(),
            "mask": self.mask[:, None].expand(-1, 3, -1).clone(),
            "window_mask": torch.ones(self.batch, 3, dtype=torch.bool),
            **{key: self.modalities[key] for key in ("bpm", "genre", "lyrics")},
        }
        values = model.forward_triplets(inputs, inputs, inputs)
        self.assertEqual(values[0].shape, (self.batch, 64))
        legacy = NeuroTuneV1Encoder("A", self.genres)
        expected = legacy(self.audio, self.mask)
        torch.testing.assert_close(legacy(**{"audio": self.audio, "mask": self.mask}), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
