"""Detect exact duplicate image files across dataset splits before model selection."""
import argparse
import hashlib
import json
from pathlib import Path
from src.base_predictor import CLASS_NAMES
from src.member2_deeplearning.dataset_aug import EXTENSIONS


def audit_dataset(split_root="data/standard_split", allow_synthetic=False):
    base = Path(split_root)
    seen = {}
    counts = {}
    duplicates = []
    manifest = []
    synthetic = []
    for split in ("train", "val", "test"):
        if not (base / split).is_dir():
            if split == "test":
                continue
            raise FileNotFoundError(f"Missing required {split} directory: {base / split}")
        counts[split] = {}
        for cls in CLASS_NAMES:
            folder = base / split / cls
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing class directory: {folder}")
            files = sorted(p for p in folder.iterdir()
                           if p.is_file() and p.suffix.lower() in EXTENSIONS)
            counts[split][cls] = len(files)
            if not files and split in ("train", "val"):
                raise ValueError(f"Empty class directory: {folder}")
            for p in files:
                if p.name.lower().startswith("synth_"):
                    synthetic.append(str(p))
                sha = hashlib.sha256(p.read_bytes()).hexdigest()
                rel = p.relative_to(base).as_posix()
                if sha in seen and seen[sha].split("/")[0] != split:
                    duplicates.append([seen[sha], rel])
                else:
                    seen[sha] = rel
                manifest.append([rel, sha])
    if duplicates:
        raise ValueError(f"Data leakage: {len(duplicates)} cross-split file duplicates. "
                         f"Example: {duplicates[:3]}")
    if synthetic and not allow_synthetic:
        raise ValueError(f"Found {len(synthetic)} synthetic debug files. "
                         "Do not report them as real accuracy. "
                         "For a smoke test only, pass --allow_synthetic.")
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return {"split_root": str(base), "counts": counts, "manifest_sha256": fingerprint,
            "n_files": len(manifest), "synthetic_files": len(synthetic),
            "cross_split_duplicates": 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/standard_split")
    p.add_argument("--allow_synthetic", action="store_true")
    p.add_argument("--out", default="results/dataset_audit.json")
    args = p.parse_args()
    result = audit_dataset(args.data, allow_synthetic=args.allow_synthetic)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2))
