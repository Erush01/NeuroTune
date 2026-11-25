import json
import hashlib
import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.run_neurotune_v1 import run_experiment
from src.model_v1 import NeuroTuneV1Encoder


class RunnerTests(unittest.TestCase):
    def fixture(self, root):
        data = root / "data"
        data.mkdir()
        (data / "ab_catalog_audit.json").write_text(json.dumps({"counts":{"train_test_overlap_removed":288}}))
        (data / "ab_catalog_tracks.csv").write_text("track_id,artist_id,artist_name,tempo,genre\n")
        (data / "ab_audio_30s_26ch.h5").write_bytes(b"cache")
        (data / "ab_audio_30s_26ch_meta.json").write_text("{}")
        split = {"track_ids": {"train": ["a", "b", "c"], "validation": ["d", "e", "f"], "test": ["g", "h", "i"]},
                 "artist_keys": {"train": ["x"], "validation": ["y"], "test": ["z"]},
                 "test_source_ids": ["g"], "test_query_ids": ["g", "h"],
                 "source_sha256": {}, "counts": {}}
        (data / "neurotune_v1_split.json").write_text(json.dumps(split))
        for name, edges in (("train", [("a", "b")]), ("validation", [("d", "e")]), ("test", [("g", "h")])):
            (data / f"neurotune_v1_{name}_edges.csv").write_text("track_id_a,track_id_b\n" + "".join(f"{a},{b}\n" for a,b in edges))
        (data / "ab_lastfm_train_edges.csv").write_text("track_id_a,track_id_b\na,b\n")
        (data / "ab_lastfm_test_edges.csv").write_text("track_id_a,track_id_b\ng,h\n")
        return data

    def test_output_namespace_refusal_preserves_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.fixture(root)
            output = root / "results" / "neurotune_v1_ab_test"
            output.mkdir(parents=True)
            marker = output / "keep"
            marker.write_text("original")
            with self.assertRaises(FileExistsError):
                run_experiment(data, output.parent, epochs=1, seeds=(684,), variants=("A",), _test_only=True)
            self.assertEqual(marker.read_text(), "original")

    def test_preflight_rejects_split_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.fixture(root)
            with self.assertRaises((ValueError, KeyError)):
                run_experiment(data, root / "results", epochs=1, seeds=(684,), variants=("A",), _test_only=True)

    def test_production_protocol_rejects_overrides_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=self.fixture(root)
            for kwargs in ({"epochs":1},{"batch_size":2},{"seeds":(684,)},{"variants":("A",)}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    run_experiment(data,root/"results",**kwargs)
            self.assertFalse((root/"results"/"neurotune_v1_ab").exists())

    def test_audit_cache_split_hash_mismatches_and_ab_boundary_rejected(self):
        from scripts.run_neurotune_v1 import _preflight, EXPECTED_SOURCE_KEYS
        for mismatch in ("cache", "split", "source_boundary"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); data=self.fixture(root)
                sources={key:"1"*64 for key in EXPECTED_SOURCE_KEYS}
                audit={"status":"complete","window":{"seconds":30},"channels":[str(i) for i in range(26)],"counts":{"eligible_tracks":6967,"train_test_overlap_removed":288},"source_sha256":sources}
                (data/"ab_catalog_audit.json").write_text(json.dumps(audit))
                ah=hashlib.sha256((data/"ab_catalog_audit.json").read_bytes()).hexdigest()
                meta={"audit_sha256":ah,"window_seconds":30,"dtype":"float32","channels":[str(i) for i in range(26)],"source_sha256":sources}
                split=json.loads((data/"neurotune_v1_split.json").read_text())
                split["source_sha256"]={"audit":ah}
                if mismatch=="cache": meta["audit_sha256"]="0"*64
                if mismatch=="split": split["source_sha256"]["audit"]="0"*64
                if mismatch=="source_boundary": audit["source_sha256"]["C"]="2"*64; (data/"ab_catalog_audit.json").write_text(json.dumps(audit)); meta["source_sha256"]=audit["source_sha256"]; split["source_sha256"]["audit"]=hashlib.sha256((data/"ab_catalog_audit.json").read_bytes()).hexdigest(); meta["audit_sha256"]=split["source_sha256"]["audit"]
                (data/"ab_audio_30s_26ch_meta.json").write_text(json.dumps(meta))
                (data/"neurotune_v1_split.json").write_text(json.dumps(split))
                with self.assertRaises(ValueError): _preflight(data)

    def test_artist_assignment_must_be_disjoint_and_match_manifest(self):
        from scripts.run_neurotune_v1 import _validate_artist_disjointness
        catalog=[{"track_id":"a","artist_id":"x"},{"track_id":"b","artist_id":"y"},{"track_id":"c","artist_id":"z"}]
        partitions={"train":["a"],"validation":["b"],"test":["c"]}
        declared={"train":["x"],"validation":["y"],"test":["z"]}
        self.assertEqual(_validate_artist_disjointness(catalog,partitions,declared)["test"],{"z"})
        leaked={**partitions,"validation":["a"]}
        with self.assertRaises(ValueError): _validate_artist_disjointness(catalog,leaked,declared)
        with self.assertRaises(ValueError): _validate_artist_disjointness(catalog,partitions,{**declared,"test":["wrong"]})

    def test_partition_edges_reject_duplicates_and_cross_partition_links(self):
        from scripts.run_neurotune_v1 import _validate_partition_edges
        parts={"train":["a","b"],"validation":["c","d"],"test":["e","f"]}
        edges={"train":[("a","b")],"validation":[("c","d")],"test":[("e","f")]}
        self.assertEqual(len(_validate_partition_edges(edges,parts)["test"]),1)
        with self.assertRaises(ValueError): _validate_partition_edges({**edges,"train":[("a","b"),("b","a")]},parts)
        with self.assertRaises(ValueError): _validate_partition_edges({**edges,"train":[("a","e")]},parts)
        with self.assertRaises(ValueError): _validate_partition_edges({**edges,"test":[("a","b")]},parts)

    def test_partition_feature_matrices_are_contiguous_views_in_partition_order(self):
        from scripts.run_neurotune_v1 import _partition_feature_views
        partitions={"train":["a","b"],"validation":["c"],"test":["d","e"]}
        features={"audio":[np.full((1,26),i,dtype=np.float32) for i in range(5)],
                  "bpm":np.arange(5,dtype=np.float32).reshape(5,1),
                  "lyrics":np.arange(25,dtype=np.float32).reshape(5,5),
                  "genre":np.arange(10,dtype=np.float32).reshape(5,2)}
        views=_partition_feature_views(features,partitions)
        self.assertEqual(list(views),["train","validation","test"])
        self.assertEqual(views["train"]["audio"],[features["audio"][0],features["audio"][1]])
        self.assertIs(views["validation"]["audio"][0],features["audio"][2])
        self.assertTrue(np.shares_memory(views["train"]["lyrics"],features["lyrics"]))
        self.assertTrue(np.shares_memory(views["validation"]["bpm"],features["bpm"]))
        self.assertTrue(np.shares_memory(views["test"]["genre"],features["genre"]))
        np.testing.assert_array_equal(views["validation"]["bpm"][:,0],[2])
        np.testing.assert_array_equal(views["test"]["bpm"][:,0],[3,4])

    def test_partition_edge_files_must_derive_from_raw_inputs_and_exclusions(self):
        from scripts.run_neurotune_v1 import _validate_partition_provenance
        catalog=[{"track_id":track} for track in "abcdefghi"]
        parts={"train":["a","b","c"],"validation":["d","e","f"],"test":["g","h","i"]}
        raw_train=[("a","b"),("d","e"),("a","d"),("h","i")]
        raw_test=[("g","h"),("g","d")]
        expected={"train":[("a","b")],"validation":[("d","e")],"test":[("g","h")]}
        counts={"excluded_cross_partition_edges":{"train_source":1,"test_source":1}}
        _validate_partition_provenance(catalog,parts,raw_train,raw_test,expected,counts)
        # Same partition and edge count, but a fabricated train edge must be rejected.
        altered={**expected,"train":[("a","c")]}
        with self.assertRaises(ValueError):
            _validate_partition_provenance(catalog,parts,raw_train,raw_test,altered,counts)
        with self.assertRaises(ValueError):
            _validate_partition_provenance(catalog,parts,raw_train,raw_test,expected,{"excluded_cross_partition_edges":{"train_source":0,"test_source":1}})

    def test_audit_overlap_count_is_separate_from_disjoint_published_csvs(self):
        from scripts.run_neurotune_v1 import _validate_upstream_overlap, _validate_cache_metadata, _csv_edges
        with tempfile.TemporaryDirectory() as tmp:
            data=self.fixture(Path(tmp))
            audit=json.loads((data/"ab_catalog_audit.json").read_text())
            train=_csv_edges(data/"ab_lastfm_train_edges.csv")
            test=_csv_edges(data/"ab_lastfm_test_edges.csv")
            self.assertEqual(len({tuple(sorted(e)) for e in train}&{tuple(sorted(e)) for e in test}),0)
            _validate_upstream_overlap(audit,train,test)
        audit={"counts":{"train_test_overlap_removed":288}}
        _validate_upstream_overlap(audit,[("a","b"),("c","d")],[("e","f")])
        with self.assertRaises(ValueError): _validate_upstream_overlap(audit,[("a","b")],[("b","a")])
        with self.assertRaises(ValueError): _validate_upstream_overlap({"counts":{"train_test_overlap_removed":287}},[],[])
        _validate_cache_metadata({"normalization":"raw/unnormalized; deferred until train split","segment_count":17},17)
        with self.assertRaises(ValueError): _validate_cache_metadata({"normalization":"normalized","segment_count":17},17)
        with self.assertRaises(ValueError): _validate_cache_metadata({"normalization":"raw/unnormalized; deferred until train split","segment_count":16},17)

    def _synthetic_run(self, root, *, trainer_hook=None):
        from scripts import run_neurotune_v1 as runner
        data = root / "data"; data.mkdir()
        ids = {"train": ["a", "b", "c"], "validation": ["d", "e", "f"], "test": ["g", "h", "i"]}
        partitions = ids
        edges = {"train": [("a", "b")], "validation": [("d", "e")], "test": [("g", "h")]}
        split = {"counts": {"tracks": {"train":3,"validation":3,"test":3}, "edges":{"train":1,"validation":1,"test":1}, "test_queries":2, "excluded_cross_partition_edges":{"train_source":0,"test_source":0}}}
        audit = {"source_sha256": {"A":"x"}}
        hashes = {"audit":"a"*64,"split":"b"*64,"cache":"c"*64,"code":{}}
        paths = {"cache":data/"cache.h5"}; paths["cache"].write_text("stub")
        feature = {"audio":[np.ones((2,26),np.float32)*i for i in range(9)],"bpm":np.ones((9,1),np.float32),"lyrics":np.zeros((9,5000),np.float32),"genre":np.ones((9,1),np.float32)}
        def loader(_): return {}, [], {}, {}, ids["train"]
        def builder(*args, **kwargs): return feature, {"fit_ids_hash":"f"}
        trainer_calls=[]
        def trainer(train, validation, ts, vs, val_edges, variant, seed, checkpoint, transform, provenance, **kwargs):
            self.assertTrue(all(e["source"] in validation.track_ids and e["target"] in validation.track_ids for e in val_edges))
            self.assertNotIn("g", [x for edge in val_edges for x in (edge["source"],edge["target"])])
            self.assertLess(int(ts.max()), len(train))
            self.assertLess(int(vs.max()), len(validation))
            if trainer_hook: trainer_hook(trainer_calls, checkpoint, val_edges, ts, vs)
            trainer.schedule_refs.append(ts)
            model=NeuroTuneV1Encoder(variant,1)
            import torch
            torch.save({"model_state_dict":model.state_dict(),"variant":variant,"seed":seed,"epoch":1,"transform_manifest":transform,"provenance":provenance,"selection_metrics":{"NDCG@10":0.}},checkpoint)
            trainer_calls.append((variant,seed,ts,vs))
            return {"history":[{"epoch":1,"train_triplet_loss":1.,"validation_triplet_loss":1.,"Recall@10":0.,"NDCG@10":0.,"MRR@10":0.,"query_count":0,"eligible_positive_edges":1}],"best_epoch":1,"best_metrics":{"NDCG@10":0.}}
        trainer.schedule_refs=[]
        def baseline(features, song_ids): return np.ones((len(song_ids),2),np.float32)
        def evaluate(embeddings, song_ids, positive_edges): return {"Recall@10":0.,"NDCG@10":0.,"MRR@10":0.,"query_count":0,"eligible_positive_edges":len(list(positive_edges))}
        return runner, data, partitions, edges, split, audit, hashes, paths, loader, builder, trainer, baseline, evaluate

    def test_all_checkpoints_precede_test_embeddings_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (runner,data,parts,edges,split,audit,hashes,paths,loader,builder,trainer,baseline,evaluate)=self._synthetic_run(root)
            from scripts import run_neurotune_v1 as module
            def assert_selected():
                staged = list((root/"results").glob(".neurotune_v1_ab_test.stage-*/checkpoints/*.pt"))
                self.assertEqual(len(staged),12)
            def spy_embed(*args):
                assert_selected()
                return np.ones((len(args[1]),64),np.float32)
            calls=[]
            def eval_spy(emb, ids, positives):
                if ids == parts["test"]:
                    assert_selected(); calls.append("test")
                return evaluate(emb,ids,positives)
            with mock.patch.object(module,"_preflight",return_value=(paths,audit,{},split,parts,edges,hashes)), mock.patch.object(module,"_embed",side_effect=spy_embed), mock.patch.object(module,"build_neurotune_v1_cosine_baseline",side_effect=baseline), mock.patch.object(module,"evaluate_neurotune_v1_retrieval",side_effect=eval_spy):
                report=run_experiment(data,root/"results",epochs=1,loader=loader,feature_builder=builder,trainer=trainer,_test_only=True)
            self.assertEqual(len(calls),13)  # fixed baseline and twelve model tests
            self.assertEqual(len(report["runs"]),12)
            self.assertEqual(report["run_type"],"synthetic_test")
            self.assertTrue((root/"results"/"neurotune_v1_ab_test"/"training_curves.csv").is_file())
            self.assertEqual(len(list((root/"results"/"neurotune_v1_ab_test"/"checkpoints").glob("*.pt"))),12)
            self.assertFalse((root/"results"/"neurotune_v1_ab").exists())
            output=root/"results"/"neurotune_v1_ab_test"
            disk_report=json.loads((output/"experiment.json").read_text())
            self.assertEqual(len(disk_report["runs"]),12)
            self.assertEqual(len(disk_report["schedule_sha256"]),3)
            import torch
            checkpoint=torch.load(output/"checkpoints/A_seed_684.pt",map_location="cpu",weights_only=False)
            self.assertTrue({"model_state_dict","variant","seed","epoch","transform_manifest","provenance","selection_metrics"} <= set(checkpoint))
            with (output/"training_curves.csv").open(newline="") as stream:
                self.assertEqual(sum(1 for _ in csv.DictReader(stream)),12)

    def test_schedule_written_once_and_reused_across_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (runner,data,parts,edges,split,audit,hashes,paths,loader,builder,trainer,baseline,evaluate)=self._synthetic_run(root)
            from scripts import run_neurotune_v1 as module
            def spy(*args): return np.ones((len(args[1]),64),np.float32)
            with mock.patch.object(module,"_preflight",return_value=(paths,audit,{},split,parts,edges,hashes)), mock.patch.object(module,"_embed",side_effect=spy), mock.patch.object(module,"build_neurotune_v1_cosine_baseline",side_effect=baseline), mock.patch.object(module,"evaluate_neurotune_v1_retrieval",side_effect=evaluate):
                run_experiment(data,root/"results",epochs=1,seeds=(684,),variants=("A","B"),loader=loader,feature_builder=builder,trainer=trainer,_test_only=True)
            self.assertIs(trainer.schedule_refs[0], trainer.schedule_refs[1])
            saved=np.load(root/"results"/"neurotune_v1_ab_test"/"sampled_triplets_seed_684.npz")
            self.assertEqual(saved["train_schedule"].dtype,np.int32)
            self.assertEqual(set(saved["train_ids"].tolist()),set(parts["train"]))
            self.assertNotIn("g", saved["train_ids"].tolist())

    def test_failure_and_keyboard_interrupt_roll_back_staging(self):
        for error in (RuntimeError("fail"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); (runner,data,parts,edges,split,audit,hashes,paths,loader,builder,trainer,baseline,evaluate)=self._synthetic_run(root)
                def broken(*args, **kwargs): raise error
                from scripts import run_neurotune_v1 as module
                with mock.patch.object(module,"_preflight",return_value=(paths,audit,{},split,parts,edges,hashes)), mock.patch.object(module,"build_neurotune_v1_cosine_baseline",side_effect=baseline):
                    with self.assertRaises(type(error)):
                        run_experiment(data,root/"results",epochs=1,seeds=(684,),variants=("A",),loader=loader,feature_builder=builder,trainer=broken,_test_only=True)
                self.assertFalse((root/"results"/"neurotune_v1_ab_test").exists())
                self.assertFalse((root/"results"/"neurotune_v1_ab").exists())
                self.assertEqual(list((root/"results").glob(".neurotune_v1_ab_test.stage-*")),[])

    def test_late_failure_after_checkpoint_creation_rolls_back_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (runner,data,parts,edges,split,audit,hashes,paths,loader,builder,trainer,baseline,evaluate)=self._synthetic_run(root)
            from scripts import run_neurotune_v1 as module
            def broken_curves(path, records):
                staged=list((root/"results").glob(".neurotune_v1_ab_test.stage-*/checkpoints/*.pt"))
                self.assertEqual(len(staged),1)
                raise RuntimeError("curve serialization failed")
            test_evaluations=[]
            def evaluator_spy(embeddings, ids, positives):
                if ids == parts["test"]: test_evaluations.append(ids)
                return evaluate(embeddings,ids,positives)
            with mock.patch.object(module,"_preflight",return_value=(paths,audit,{},split,parts,edges,hashes)), mock.patch.object(module,"_embed",return_value=np.ones((3,64),np.float32)), mock.patch.object(module,"build_neurotune_v1_cosine_baseline",side_effect=baseline), mock.patch.object(module,"evaluate_neurotune_v1_retrieval",side_effect=evaluator_spy), mock.patch.object(module,"_write_curves",side_effect=broken_curves):
                with self.assertRaisesRegex(RuntimeError,"curve serialization failed"):
                    run_experiment(data,root/"results",epochs=1,seeds=(684,),variants=("A",),loader=loader,feature_builder=builder,trainer=trainer,_test_only=True)
            self.assertEqual(test_evaluations,[])
            self.assertFalse((root/"results"/"neurotune_v1_ab_test").exists())
            self.assertFalse((root/"results"/"neurotune_v1_ab").exists())
            self.assertEqual(list((root/"results").glob(".neurotune_v1_ab_test.stage-*")),[])


if __name__ == "__main__":
    unittest.main()
