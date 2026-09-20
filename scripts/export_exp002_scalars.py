#!/usr/bin/env python3
"""Export EXP-002 TensorBoard scalar events to a portable CSV file.

The TensorBoard dependency is imported only when the script runs. This keeps
the repository's local JSON analysis usable on machines without TensorBoard.
The script reads event files only; it never starts a TensorBoard server.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FORMAL_RUNS = tuple(
    f"{condition}_seed{seed}"
    for condition in ("clean", "noisy")
    for seed in (1001, 1002, 1003)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--runs", nargs="*", choices=FORMAL_RUNS, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:  # pragma: no cover - depends on execution host
        raise SystemExit(
            "TensorBoard is required. Use the maniskill environment's Python."
        ) from exc

    run_names = tuple(args.runs) if args.runs else FORMAL_RUNS
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    tags_seen: dict[str, int] = {}
    rows: list[dict[str, object]] = []

    for run_name in run_names:
        event_files = sorted((args.runs_root / run_name / "tensorboard").glob("events.out.tfevents.*"))
        if not event_files:
            raise SystemExit(f"No TensorBoard event file found for {run_name}")
        accumulator = EventAccumulator(
            str(event_files[-1]),
            size_guidance={"scalars": 0},
        )
        accumulator.Reload()
        scalar_tags = sorted(accumulator.Tags().get("scalars", []))
        tags_seen[run_name] = len(scalar_tags)
        for tag in scalar_tags:
            for event in accumulator.Scalars(tag):
                rows.append(
                    {
                        "run": run_name,
                        "tag": tag,
                        "step": event.step,
                        "wall_time": event.wall_time,
                        "value": event.value,
                    }
                )

    rows.sort(key=lambda row: (str(row["run"]), str(row["tag"]), int(row["step"])))
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run", "tag", "step", "wall_time", "value"),
        )
        writer.writeheader()
        writer.writerows(rows)

    for run_name, tag_count in tags_seen.items():
        print(f"{run_name}: scalar_tags={tag_count}")
    print(f"rows={len(rows)} output={args.output_csv}")
    print("EXP002_SCALARS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
