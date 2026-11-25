"""Run the fixed, provenance-checked NeuroTune v1 A/B pilot experiment."""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from src.data.neurotune_v1_dataset import NeuroTuneV1Dataset, neurotune_v1_collate
from src.data.neurotune_v1_split import sha256_file
from src.evaluate_neurotune_v1 import build_neurotune_v1_cosine_baseline, evaluate_neurotune_v1_retrieval
from src.features_neurotune_v1 import build_neurotune_v1_features, load_neurotune_v1_inputs
from src.train_neurotune_v1 import train_neurotune_v1
from src.triplets import sample_triplet_schedule

VARIANTS = ("A", "B", "C", "D")
SEEDS = (684, 685, 686)
TRAIN_EPOCHS, BATCH_SIZE, LEARNING_RATE, MARGIN = 50, 64, 0.001, 0.2
EXPECTED_SOURCE_KEYS = {"A", "B", "duplicates", "genres", "lastfm_test", "lastfm_train", "mxm"}
CAVEATS = [
    "Exploratory result based on only 215 official test edges.",
    "Unobserved links are unverified negatives.",
    "Third-party A/B provenance is unverified.",
    "No significance or generalization claims are warranted.",
]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _csv_edges(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"track_id_a", "track_id_b"}.issubset(reader.fieldnames):
            raise ValueError(f"invalid edge file: {path}")
        return [(row["track_id_a"].strip(), row["track_id_b"].strip()) for row in reader]


def _tree_hash(path):
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    entries = []
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        entries.append((child.relative_to(path).as_posix(), sha256_file(child)))
    return _digest_json(entries)


def _validate_artist_disjointness(catalog, partitions, declared_artist_keys):
    artist_by_id = {row["track_id"].strip(): (row.get("artist_id", "").strip() or row.get("artist_name", "").strip() or row["track_id"].strip()) for row in catalog}
    artist_sets = {p: {artist_by_id[x] for x in partitions[p]} for p in partitions}
    if any(artist_sets[a] & artist_sets[b] for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("artist leakage across split partitions")
    if any(set(declared_artist_keys.get(p, [])) != artist_sets[p] for p in partitions):
        raise ValueError("split artist keys do not match catalog assignment")
    return artist_sets


def _validate_partition_edges(edges, partitions):
    edge_sets = {}
    for partition, rows in edges.items():
        ids = set(partitions[partition]); canonical = [tuple(sorted(edge)) for edge in rows]
        if len(set(canonical)) != len(canonical) or any(a not in ids or b not in ids or a == b for a,b in rows):
            raise ValueError(f"{partition} edges duplicate or cross partition")
        edge_sets[partition] = set(canonical)
    if edge_sets["train"] & edge_sets["test"]:
        raise ValueError("train/test edges overlap")
    return edge_sets


def _validate_partition_provenance(catalog, partitions, raw_train, raw_test, published_edges, split_counts):
    """Reconstruct split-builder edge policy and reconcile its exclusion counts."""
    track_ids = {row["track_id"].strip() for row in catalog}
    partition_by_id = {track_id: partition for partition, ids in partitions.items() for track_id in ids}
    if set(partition_by_id) != track_ids:
        raise ValueError("partition IDs do not exactly cover catalog for edge provenance")
    expected = {"train": set(), "validation": set(), "test": set()}
    excluded = {"train_source": 0, "test_source": 0}
    for source, rows in (("train_source", raw_train), ("test_source", raw_test)):
        for left, right in rows:
            if left not in partition_by_id or right not in partition_by_id or left == right:
                raise ValueError("raw edge endpoint is absent from catalog or is a self-edge")
            left_partition, right_partition = partition_by_id[left], partition_by_id[right]
            destination = None
            if left_partition != right_partition:
                excluded[source] += 1
            elif source == "train_source" and left_partition in ("train", "validation"):
                destination = left_partition
            elif source == "test_source" and left_partition == "test":
                destination = "test"
            if destination is not None:
                edge = tuple(sorted((left, right)))
                if edge in expected[destination]:
                    raise ValueError(f"duplicate raw edge assigned to {destination}: {edge}")
                expected[destination].add(edge)
    published = _validate_partition_edges(published_edges, partitions)
    if expected != published:
        raise ValueError("published partition edge CSVs do not match raw-source split policy")
    if split_counts.get("excluded_cross_partition_edges") != excluded:
        raise ValueError("split cross-partition exclusion counts do not match raw edge inputs")
    return expected


def _validate_upstream_overlap(audit, raw_train, raw_test):
    if audit.get("counts", {}).get("train_test_overlap_removed") != 288:
        raise ValueError("audit does not record 288 upstream train/test overlap removals")
    train = {tuple(sorted(edge)) for edge in raw_train}
    test = {tuple(sorted(edge)) for edge in raw_test}
    if train & test:
        raise ValueError("published raw train/test edge files must be disjoint")


def _validate_cache_metadata(meta, segment_count):
    if meta.get("normalization") != "raw/unnormalized; deferred until train split":
        raise ValueError("audio cache must contain raw unnormalized segments")
    if meta.get("segment_count") != segment_count:
        raise ValueError("cache metadata segment count does not match HDF5 features")


def _partition_feature_views(features, partitions):
    """Slice row-aligned transformed features without copying dense matrices."""
    result = {}
    start = 0
    for partition, ids in partitions.items():
        stop = start + len(ids)
        result[partition] = {
            name: (values[start:stop] if name == "audio" else values[start:stop, :])
            for name, values in features.items()
        }
        start = stop
    return result


def _preflight(data_dir):
    data_dir = Path(data_dir)
    paths = {name: data_dir / filename for name, filename in {
        "audit": "ab_catalog_audit.json", "catalog": "ab_catalog_tracks.csv",
        "cache": "ab_audio_30s_26ch.h5", "cache_meta": "ab_audio_30s_26ch_meta.json",
        "split": "neurotune_v1_split.json", "train_edges": "neurotune_v1_train_edges.csv",
        "validation_edges": "neurotune_v1_validation_edges.csv", "test_edges": "neurotune_v1_test_edges.csv",
        "raw_train": "ab_lastfm_train_edges.csv", "raw_test": "ab_lastfm_test_edges.csv",
    }.items()}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
    audit, meta, split = (_json(paths[key]) for key in ("audit", "cache_meta", "split"))
    ah, sh = sha256_file(paths["audit"]), sha256_file(paths["split"])
    if audit.get("status") != "complete" or audit.get("window", {}).get("seconds") != 30 or len(audit.get("channels", [])) != 26 or audit.get("counts", {}).get("eligible_tracks") != 6967:
        raise ValueError("audit violates complete 30s/26-channel contract")
    if meta.get("window_seconds") != 30 or meta.get("dtype") != "float32" or len(meta.get("channels", [])) != 26:
        raise ValueError("cache metadata violates 30s/26-channel contract")
    if meta.get("channels") != audit.get("channels") or audit.get("counts", {}).get("train_test_overlap_removed") != 288:
        raise ValueError("audit/cache channel or upstream-overlap contract mismatch")
    audit_sources, meta_sources = audit.get("source_sha256", {}), meta.get("source_sha256", {})
    if set(audit_sources) != EXPECTED_SOURCE_KEYS or set(meta_sources) != EXPECTED_SOURCE_KEYS:
        raise ValueError("audit/cache sources must be exactly the approved A/B source boundary")
    if ah != meta.get("audit_sha256") or ah != split.get("source_sha256", {}).get("audit"):
        raise ValueError("audit/cache/split SHA-256 mismatch")
    if audit_sources != meta_sources:
        raise ValueError("cache source hashes differ from audit")
    source_files = {"catalog": paths["catalog"], "train_edges": paths["raw_train"], "test_edges": paths["raw_test"]}
    for key, path in source_files.items():
        if split.get("source_sha256", {}).get(key) != sha256_file(path):
            raise ValueError(f"split {key} source hash mismatch")
    if split.get("source_sha256", {}).get("lastfm_test") != audit_sources["lastfm_test"]:
        raise ValueError("split Last.fm test hash differs from audit")
    if split.get("contract") != {"window_seconds": 30, "channels": 26}:
        raise ValueError("split manifest violates 30s/26-channel contract")
    ids_by_part = split.get("track_ids", {})
    partitions = {p: ids_by_part.get(p, []) for p in ("train", "validation", "test")}
    flat = [x for part in partitions.values() for x in part]
    if any(len(v) != len(set(v)) for v in partitions.values()) or len(flat) != len(set(flat)):
        raise ValueError("split track IDs must be unique and disjoint")
    with paths["catalog"].open(newline="", encoding="utf-8") as stream:
        catalog = list(csv.DictReader(stream))
    catalog_ids = {row["track_id"].strip() for row in catalog}
    if set(flat) != catalog_ids:
        raise ValueError("split IDs do not exactly cover catalog IDs")
    artist_keys = split.get("artist_keys", {})
    artist_sets = _validate_artist_disjointness(catalog, partitions, artist_keys)
    counts = split.get("counts", {})
    if counts.get("artists") != {p: len(artist_sets[p]) for p in partitions}:
        raise ValueError("split artist counts mismatch")
    if not set(split.get("test_source_ids", [])) <= set(partitions["test"]) or not set(split.get("test_query_ids", [])) <= set(partitions["test"]):
        raise ValueError("test-source/query IDs are not held out")
    if set(split.get("test_query_ids", [])) != {track_id for edge in _csv_edges(paths["test_edges"]) for track_id in edge}:
        raise ValueError("test query IDs do not match official test edge endpoints")
    with h5py.File(paths["cache"], "r") as h5:
        features = h5["features"]
        _validate_cache_metadata(meta, features.shape[0])
        cache_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["track_ids"][:]]
        offsets = h5["offsets"]
        if features.shape[1:] != (26,) or features.dtype != np.dtype("float32") or cache_ids != sorted(catalog_ids) or len(cache_ids) != len(catalog_ids) or meta.get("track_count") != len(cache_ids):
            raise ValueError("audio cache shape/dtype/IDs differ from catalog contract")
        lengths = np.diff(offsets[:])
        if offsets.dtype != np.dtype("int64") or len(offsets) != len(cache_ids)+1 or int(offsets[0]) != 0 or int(offsets[-1]) != features.shape[0] or np.any(lengths <= 0):
            raise ValueError("audio cache offsets violate track coverage contract")
        meta_lengths = meta.get("sequence_lengths", {})
        if set(meta_lengths) != set(cache_ids) or any(meta_lengths[track_id] != int(lengths[i]) for i, track_id in enumerate(cache_ids)):
            raise ValueError("cache metadata sequence lengths do not match cache offsets")
    edges = {p: _csv_edges(paths[f"{p}_edges"]) for p in partitions}
    counts = split.get("counts", {})
    expected = {"tracks": {"train":5632,"validation":610,"test":725}, "edges":{"train":12916,"validation":185,"test":215}, "test_queries":265, "excluded_cross_partition_edges":{"train_source":4435,"test_source":1725}}
    for key, value in expected.items():
        if counts.get(key) != value:
            raise ValueError(f"split manifest {key} counts mismatch")
    raw_train, raw_test = _csv_edges(paths["raw_train"]), _csv_edges(paths["raw_test"])
    _validate_upstream_overlap(audit, raw_train, raw_test)
    _validate_partition_provenance(catalog, partitions, raw_train, raw_test, edges, counts)
    code_paths = ("scripts/run_neurotune_v1.py", "scripts/preprocess_msd_hf_ab.py", "src/features_neurotune_v1.py", "src/data/neurotune_v1_dataset.py", "src/data/neurotune_v1_split.py", "src/data/parse_mxm.py", "src/triplets.py", "src/train_neurotune_v1.py", "src/evaluate_neurotune_v1.py", "src/evaluate.py", "src/retrieve.py", "src/model_v1.py")
    return paths, audit, meta, split, partitions, edges, {"audit":ah,"split":sh,"cache":_tree_hash(paths["cache"]),"input_files":{key:sha256_file(path) for key,path in paths.items()},"code":{p:_tree_hash(Path(__file__).resolve().parents[1]/p) for p in code_paths}}


def _embed(model, dataset, batch_size=64):
    model.eval(); batches = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            inp = neurotune_v1_collate([dataset[i] for i in range(start, min(start+batch_size, len(dataset)))])
            batches.append(model(**{k:inp[k] for k in ("audio","mask",*model.active_modalities)}).cpu().numpy())
    return np.concatenate(batches)


def _write_curves(path, records):
    fields = ("variant","seed","epoch","train_triplet_loss","validation_triplet_loss","Recall@10","NDCG@10","MRR@10","query_count","eligible_positive_edges")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)


def run_experiment(data_dir="data/msd_hf_pilot", results_dir="results", *, epochs=TRAIN_EPOCHS, seeds=SEEDS, variants=VARIANTS, batch_size=BATCH_SIZE, loader=load_neurotune_v1_inputs, feature_builder=build_neurotune_v1_features, trainer=train_neurotune_v1, _test_only=False):
    """Run production protocol, or isolated synthetic-test runs in a separate namespace."""
    if not isinstance(_test_only, bool):
        raise ValueError("_test_only must be a boolean")
    if not _test_only and (tuple(variants) != VARIANTS or tuple(seeds) != SEEDS or epochs != TRAIN_EPOCHS or batch_size != BATCH_SIZE or loader is not load_neurotune_v1_inputs or feature_builder is not build_neurotune_v1_features or trainer is not train_neurotune_v1):
        raise ValueError("production runs are locked to A-D, seeds 684/685/686, 50 epochs, and batch size 64")
    variants, seeds = tuple(variants), tuple(seeds)
    if not variants or len(set(variants)) != len(variants) or any(variant not in VARIANTS for variant in variants):
        raise ValueError("variants must be a nonempty subset of A, B, C, D")
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonempty nonnegative integers")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    data_dir = Path(data_dir)
    namespace = "neurotune_v1_ab_test" if _test_only else "neurotune_v1_ab"
    final = Path(results_dir)/namespace
    if final.exists(): raise FileExistsError(f"refusing to overwrite output namespace: {final}")
    paths, audit, meta, split, partitions, edges, hashes = _preflight(data_dir)
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{namespace}.stage-", dir=final.parent))
    try:
        stage.mkdir(exist_ok=True)
        loaded = loader(data_dir)
        _, tracks, lyrics, genres, _ = loaded
        all_ids = partitions["train"] + partitions["validation"] + partitions["test"]
        feature_values, transform = feature_builder(paths["cache"], tracks, lyrics, genres, partitions["train"], all_ids, split_hash=hashes["split"])
        partition_features = _partition_feature_views(feature_values, partitions)
        datasets = {p: NeuroTuneV1Dataset(ids, partition_features[p]) for p,ids in partitions.items()}
        val_baseline = evaluate_neurotune_v1_retrieval(build_neurotune_v1_cosine_baseline(partition_features["validation"], partitions["validation"]), partitions["validation"], edges["validation"])
        reports, curves, schedule_hashes = [], [], {}
        for seed in seeds:
            train_schedule = sample_triplet_schedule(partitions["train"], edges["train"], epochs, seed)
            validation_schedule = sample_triplet_schedule(partitions["validation"], edges["validation"], 1, seed)
            schedule_path = stage/f"sampled_triplets_seed_{seed}.npz"
            np.savez_compressed(schedule_path, train_ids=np.asarray(partitions["train"], dtype="U"), validation_ids=np.asarray(partitions["validation"], dtype="U"), train_schedule=train_schedule.astype(np.int32, copy=False), validation_schedule=validation_schedule.astype(np.int32, copy=False))
            schedule_hash = sha256_file(schedule_path)
            schedule_hashes[str(seed)] = schedule_hash
            for variant in variants:
                ckpt = stage/"checkpoints"/f"{variant}_seed_{seed}.pt"; ckpt.parent.mkdir(exist_ok=True)
                result = trainer(datasets["train"], datasets["validation"], train_schedule, validation_schedule, [{"source":a,"target":b} for a,b in edges["validation"]], variant, seed, ckpt, transform, {**hashes,"schedule_sha256":schedule_hash}, epochs=epochs, batch_size=batch_size)
                curves.extend({"variant":variant,"seed":seed,**row} for row in result["history"])
                reports.append({"variant":variant,"seed":seed,"selected_epoch":result["best_epoch"],"validation":result["best_metrics"],"checkpoint_sha256":sha256_file(ckpt),"checkpoint_path":f"checkpoints/{variant}_seed_{seed}.pt"})
                del result
            del train_schedule, validation_schedule
        # Leakage barrier: every selected checkpoint exists before any test embedding/evaluation.
        expected_checkpoints = [stage/"checkpoints"/f"{v}_seed_{s}.pt" for s in seeds for v in variants]
        if not all(p.is_file() for p in expected_checkpoints): raise RuntimeError("not all selected checkpoints were saved")
        _write_curves(stage/"training_curves.csv", curves)
        baseline_test = evaluate_neurotune_v1_retrieval(build_neurotune_v1_cosine_baseline(partition_features["test"], partitions["test"]), partitions["test"], edges["test"])
        for report in reports:
            checkpoint = torch.load(stage/report["checkpoint_path"], map_location="cpu", weights_only=False)
            from src.model_v1 import NeuroTuneV1Encoder
            model = NeuroTuneV1Encoder(report["variant"], datasets["train"].features["genre"].shape[1]).cpu()
            model.load_state_dict(checkpoint["model_state_dict"])
            report["test"] = evaluate_neurotune_v1_retrieval(_embed(model, datasets["test"], batch_size), partitions["test"], edges["test"])
            del model, checkpoint
        report = {"run_type":"synthetic_test" if _test_only else "production","protocol":{"variants":list(variants),"seeds":list(seeds),"epochs":epochs,"batch_size":batch_size,"optimizer":"Adam","learning_rate":LEARNING_RATE,"margin":MARGIN,"device":"cpu"},"runtime":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__},"counts":split["counts"],"upstream_train_test_overlap_removals":288,"source_sha256":audit["source_sha256"],"hashes":hashes,"transform_sha256":_digest_json(transform),"schedule_sha256":schedule_hashes,"baseline":{"validation":val_baseline,"test":baseline_test,"fixed_weights":True},"runs":reports,"caveats":CAVEATS}
        (stage/"experiment.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists(): raise FileExistsError(final)
        os.replace(stage, final)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--data-dir", default="data/msd_hf_pilot"); parser.add_argument("--results-dir", default="results")
    args=parser.parse_args(); run_experiment(args.data_dir,args.results_dir)


if __name__ == "__main__": main()
