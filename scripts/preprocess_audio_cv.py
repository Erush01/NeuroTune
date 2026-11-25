"""Build one full-catalog normalized audio cache per CV test fold."""

import argparse
import json
from pathlib import Path

import h5py

from src.data.preprocess_audio import _track_id, build_audio_sequence_cache


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "results/msd_audio_only_cv_audit.json"
DEFAULT_INPUT_ROOT = ROOT / "data/msd_subset_raw/MillionSongSubset"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed"


def _load_folds(audit_path):
    try:
        with audit_path.open(encoding="utf-8") as source:
            audit = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read CV audit {audit_path}: {error}") from error
    if not isinstance(audit, dict) or audit.get("status") != "ready":
        raise ValueError("CV audit status must be 'ready'")
    fold_count = audit.get("fold_count")
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count <= 0:
        raise ValueError("CV audit fold_count must be a positive integer")
    folds = audit.get("folds")
    expected_keys = [str(index) for index in range(fold_count)]
    if not isinstance(folds, dict) or set(folds) != set(expected_keys):
        raise ValueError(f"CV audit folds must have keys {expected_keys!r}")

    normalized = {}
    catalog = []
    for key in expected_keys:
        fold = folds[key]
        ids = fold.get("track_ids") if isinstance(fold, dict) else None
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"CV audit fold {key} must contain a non-empty track_ids list")
        normalized_ids = []
        for value in ids:
            if not isinstance(value, (str, bytes)) or not _track_id(value):
                raise ValueError(f"CV audit fold {key} contains an invalid track ID")
            normalized_ids.append(_track_id(value))
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError(f"CV audit fold {key} contains duplicate track IDs")
        normalized[key] = normalized_ids
        catalog.extend(normalized_ids)
    if len(set(catalog)) != len(catalog):
        raise ValueError("CV audit folds contain overlapping or duplicate track IDs")
    catalog_counts = audit.get("catalog_counts", {})
    if not isinstance(catalog_counts, dict):
        raise ValueError("CV audit catalog_counts must be an object")
    declared_count = catalog_counts.get("eligible_tracks")
    if declared_count is not None and (
        isinstance(declared_count, bool) or not isinstance(declared_count, int)
        or declared_count <= 0
    ):
        raise ValueError("CV audit catalog_counts.eligible_tracks must be a positive integer")
    if declared_count is not None and declared_count != len(catalog):
        raise ValueError(
            f"CV audit catalog coverage mismatch: folds contain {len(catalog)} IDs, "
            f"catalog declares {declared_count}"
        )
    return normalized, catalog


def _validate_inputs(input_root, catalog):
    if not input_root.is_dir():
        raise ValueError(f"raw input root does not exist or is not a directory: {input_root}")
    sources = {track_id: [] for track_id in catalog}
    for path in sorted(input_root.rglob("*.h5")):
        try:
            with h5py.File(path, "r") as handle:
                track_id = _track_id(handle["analysis"]["songs"][0]["track_id"])
        except (OSError, KeyError, IndexError, ValueError):
            continue
        if track_id in sources:
            sources[track_id].append(path)
    missing = [track_id for track_id, paths in sources.items() if not paths]
    duplicates = [track_id for track_id, paths in sources.items() if len(paths) > 1]
    if missing:
        raise ValueError(f"missing source HDF5 files for catalog IDs: {missing!r}")
    if duplicates:
        raise ValueError(f"duplicate source HDF5 files for catalog IDs: {duplicates!r}")


def run(input_root=DEFAULT_INPUT_ROOT, audit=DEFAULT_AUDIT,
        output_dir=DEFAULT_OUTPUT_DIR, grid_size=1024):
    audit_path = Path(audit).resolve()
    input_path = Path(input_root).resolve()
    destination = Path(output_dir).resolve()
    folds, catalog = _load_folds(audit_path)
    output_pairs = []
    for index in range(len(folds)):
        output_pairs.append((
            destination / f"msd_audio_only_cv_fold_{index}.h5",
            destination / f"msd_audio_only_cv_fold_{index}_meta.json",
        ))
    output_paths = [path for pair in output_pairs for path in pair]
    resolved_outputs = [path.resolve() for path in output_paths]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("generated output paths collide")
    if audit_path in resolved_outputs:
        raise ValueError("generated output path collides with CV audit")
    if any(path == input_path or input_path in path.parents for path in resolved_outputs):
        raise ValueError("generated output paths must not be inside the raw input root")
    existing_outputs = [path for path in resolved_outputs if path.exists()]
    if existing_outputs:
        raise ValueError(f"generated output target already exists: {existing_outputs[0]}")

    _validate_inputs(input_path, catalog)
    for index, (hdf5_path, metadata_path) in enumerate(output_pairs):
        test_ids = folds[str(index)]
        train_ids = [track_id for fold_key, ids in folds.items()
                     if fold_key != str(index) for track_id in ids]
        build_audio_sequence_cache(
            input_root=input_path,
            split_ids={"train": train_ids, "validation": [], "test": test_ids},
            audit_json_path=audit_path,
            hdf5_output_path=hdf5_path,
            metadata_json_path=metadata_path,
            grid_size=grid_size,
        )
    return output_pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-size", type=int, default=1024)
    args = parser.parse_args(argv)
    return run(input_root=args.input_root, audit=args.audit,
               output_dir=args.output_dir, grid_size=args.grid_size)


if __name__ == "__main__":
    main()
