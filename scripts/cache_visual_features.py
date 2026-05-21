"""Precompute visual token features for WIT-VZ path prediction samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.backbones import build_visual_encoder
from src.wit_vz.dataset import load_rgb_tensor
from src.wit_vz.io import load_json, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen visual token features for processed WIT-VZ samples.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", default="dinov3-convnext-tiny")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--payload",
        choices=["dict", "tensor"],
        default="dict",
        help="Store each feature as a metadata dict or as the raw tensor only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_raw_root(dataset_dir: Path, raw_dir: str | Path) -> Path:
    path = Path(raw_dir)
    if path.is_absolute():
        return path
    candidate = (dataset_dir / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def resolve_raw_dirs(dataset_dir: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    if "raw_dirs" in manifest:
        return {
            str(source_id): resolve_raw_root(dataset_dir, raw_dir)
            for source_id, raw_dir in manifest["raw_dirs"].items()
        }
    return {"default": resolve_raw_root(dataset_dir, manifest["raw_dir"])}


def resolve_raw_path(
    raw_dirs: dict[str, Path],
    rel_path: str,
    source_id: str | None = None,
) -> Path:
    selected_source_id = source_id
    rel = rel_path
    if "::" in rel_path:
        selected_source_id, rel = rel_path.split("::", 1)
    path = Path(rel)
    if path.is_absolute():
        return path
    if selected_source_id is not None and selected_source_id in raw_dirs:
        return raw_dirs[selected_source_id] / path
    return next(iter(raw_dirs.values())) / path


def save_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_manifest: dict[str, Any],
    token_shape: tuple[int, ...],
    num_cached: int,
) -> None:
    manifest = {
        "dataset": args.dataset.as_posix(),
        "dataset_id": dataset_manifest.get("dataset_id"),
        "backbone": args.backbone,
        "image_size": args.image_size,
        "dtype": args.dtype,
        "token_shape": list(token_shape),
        "history_frames": dataset_manifest.get("history_frames"),
        "num_cached": num_cached,
        "payload": args.payload,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    (output_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    dataset_manifest = load_json(args.dataset / "dataset_manifest.json")
    raw_dirs = resolve_raw_dirs(args.dataset, dataset_manifest)
    samples = read_jsonl(args.dataset / "samples.jsonl")
    if args.limit > 0:
        samples = samples[: args.limit]
    total_unsharded = len(samples)
    if args.num_shards > 1:
        samples = [
            sample for index, sample in enumerate(samples)
            if index % args.num_shards == args.shard_index
        ]

    encoder = build_visual_encoder(args.backbone, hidden_dim=128, freeze_backbone=True).to(device)
    encoder.eval()
    save_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    pending = []
    cached = 0
    token_shape: tuple[int, ...] | None = None

    def flush(batch_samples: list[dict[str, Any]]) -> None:
        nonlocal cached, token_shape
        if not batch_samples:
            return
        histories = []
        for sample in batch_samples:
            source_id = sample.get("source", {}).get("source_id") or sample.get("metadata", {}).get("source_id")
            frames = [
                load_rgb_tensor(resolve_raw_path(raw_dirs, rel_path, source_id), args.image_size)
                for rel_path in sample["rgb_history_paths"]
            ]
            histories.append(torch.stack(frames, dim=0))
        rgb = torch.stack(histories, dim=0).to(device)
        with torch.no_grad(), torch.amp.autocast(
            "cuda",
            enabled=args.mixed_precision and device.type == "cuda",
        ):
            tokens = encoder(rgb)
        tokens = tokens.detach().cpu().to(save_dtype)
        if token_shape is None:
            token_shape = tuple(tokens.shape[1:])
        for sample, sample_tokens in zip(batch_samples, tokens, strict=True):
            payload: torch.Tensor | dict[str, Any]
            if args.payload == "tensor":
                payload = sample_tokens.clone()
            else:
                payload = {
                    "sample_id": sample["sample_id"],
                    "visual_tokens": sample_tokens.clone(),
                    "backbone": args.backbone,
                    "image_size": args.image_size,
                }
            torch.save(payload, feature_dir / f"{sample['sample_id']}.pt")
            cached += 1
        print(
            f"shard={args.shard_index}/{args.num_shards} "
            f"cached={cached}/{len(samples)} total={total_unsharded}"
        )

    for sample in samples:
        output_path = feature_dir / f"{sample['sample_id']}.pt"
        if output_path.exists() and not args.overwrite:
            cached += 1
            if token_shape is None:
                cached_payload = torch.load(output_path, map_location="cpu")
                cached_tokens = cached_payload["visual_tokens"] if isinstance(cached_payload, dict) else cached_payload
                token_shape = tuple(cached_tokens.shape)
            continue
        pending.append(sample)
        if len(pending) >= args.batch_size:
            flush(pending)
            pending = []
    flush(pending)
    if token_shape is None:
        raise ValueError("No feature tensors were cached")
    if args.num_shards == 1:
        save_manifest(args.output_dir, args, dataset_manifest, token_shape, cached)
    else:
        shard_manifest = {
            "dataset": args.dataset.as_posix(),
            "dataset_id": dataset_manifest.get("dataset_id"),
            "backbone": args.backbone,
            "image_size": args.image_size,
            "dtype": args.dtype,
            "token_shape": list(token_shape),
            "history_frames": dataset_manifest.get("history_frames"),
            "num_cached": cached,
            "payload": args.payload,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        }
        (args.output_dir / f"feature_manifest_shard_{args.shard_index:03d}.json").write_text(
            json.dumps(shard_manifest, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
