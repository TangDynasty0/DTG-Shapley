"""DTG-Shapley public implementation experiments on structured and sparse-text tables.

Each dataset/subset condition is value-audited before its reference and method
cases run. Results are persisted per case and can be resumed safely.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from experiment_core import (
    ExperimentCase,
    _atomic_json,
    _json_safe,
    refresh_outputs,
    run_case,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reproduced_results"
DEFAULT_RUN_NAME = "dtg_structured_text"
DEFAULT_MAX_HOURS = 9.75
DEVELOPMENT_SEEDS = (11, 23, 42)
FULL_SEEDS = (11, 23, 42, 67, 89)


def _deduplicate(cases: Iterable[ExperimentCase]) -> list[ExperimentCase]:
    return list({case.case_id: case for case in cases}.values())


def _condition_cases(
    dataset: str,
    subset_count: int,
    seed: int,
    paths: int,
    reference_paths: int,
    value_quantifier: str = "dataset_default",
) -> list[ExperimentCase]:
    common = {
        "experiment": "structured_text_public",
        "scenario": "documented_or_natural_low_tail",
        "dataset": dataset,
        "seed": seed,
        "artificial_low_count": 0,
        "threshold_mode": "relative",
        "relative_threshold_factor": 0.5,
        "feature_subset_count": subset_count,
        "feature_subset_seed": seed,
        "implementation_version": "dtg_structured_text_gate",
        "value_quantifier": value_quantifier,
    }
    return [
        ExperimentCase(
            algorithm="value_audit",
            equivalent_mc_paths=0,
            **common,
        ),
        ExperimentCase(
            algorithm="mc_reference",
            equivalent_mc_paths=reference_paths,
            reference=True,
            **common,
        ),
        ExperimentCase(
            algorithm="mc",
            equivalent_mc_paths=paths,
            **common,
        ),
        ExperimentCase(
            algorithm="dtg_shapley",
            equivalent_mc_paths=paths,
            **common,
        ),
    ]


def _audit_only_case(
    dataset: str,
    subset_count: int,
    seed: int,
    value_quantifier: str = "dataset_default",
) -> ExperimentCase:
    return ExperimentCase(
        experiment="structured_text_value_audit",
        scenario="value_function_applicability_check",
        dataset=dataset,
        algorithm="value_audit",
        seed=seed,
        artificial_low_count=0,
        equivalent_mc_paths=0,
        threshold_mode="relative",
        relative_threshold_factor=0.5,
        feature_subset_count=subset_count,
        feature_subset_seed=seed,
        implementation_version="dtg_structured_text_gate",
        value_quantifier=value_quantifier,
    )


def _build_quantifier_cases(
    profile: str,
    value_quantifier: str,
) -> list[ExperimentCase]:
    if profile == "smoke":
        return _condition_cases(
            "spambase", 20, 11, 10, 30, value_quantifier
        )
    if profile == "pilot":
        cases = [
            _audit_only_case("dexter", 100, 11, value_quantifier)
        ]
        for dataset, count in (
            ("internet_advertisements", 50),
            ("cnae_9", 50),
            ("farm_ads", 50),
            ("spambase", 57),
        ):
            cases.extend(
                _condition_cases(
                    dataset, count, 11, 20, 60, value_quantifier
                )
            )
        return _deduplicate(cases)

    seeds = FULL_SEEDS if profile == "full" else DEVELOPMENT_SEEDS
    subset_counts = {
        "internet_advertisements": (
            (50, 100, 200) if profile == "full" else (50, 100)
        ),
        "cnae_9": (50, 100, 200) if profile == "full" else (50, 100),
        "farm_ads": (50, 100, 200) if profile == "full" else (50, 100),
        "spambase": (57,),
    }
    paths = 150 if profile == "full" else 100
    reference_paths = 1_200 if profile == "full" else 600
    dexter_counts = (50, 100, 200) if profile == "full" else (50, 100)
    cases: list[ExperimentCase] = [
        _audit_only_case("dexter", count, seed, value_quantifier)
        for count in dexter_counts
        for seed in seeds
    ]
    for dataset, counts in subset_counts.items():
        for count in counts:
            for seed in seeds:
                cases.extend(
                    _condition_cases(
                        dataset,
                        count,
                        seed,
                        paths,
                        reference_paths,
                        value_quantifier,
                    )
                )
    return _deduplicate(cases)


def build_cases(profile: str) -> list[ExperimentCase]:
    # Preserve this order: finish the existing predictive-information
    # experiment before starting the otherwise identical MI experiment.
    cases: list[ExperimentCase] = []
    for value_quantifier in (
        "dataset_default",
        "mutual_information",
    ):
        cases.extend(_build_quantifier_cases(profile, value_quantifier))
    return _deduplicate(cases)


def _audit_case_map(cases: Iterable[ExperimentCase]) -> dict[tuple, ExperimentCase]:
    return {
        case.condition_key: case
        for case in cases
        if case.algorithm == "value_audit"
    }


def _audit_allows_shapley(audit_path: Path) -> tuple[bool, str]:
    if not audit_path.exists():
        return False, "value audit result is missing"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "success":
        return False, "value audit failed"
    decision = (
        audit.get("metrics", {})
        .get("diagnostics", {})
        .get("decision")
    )
    return decision == "eligible_for_shapley", str(decision)


def _skipped_result(case: ExperimentCase, reason: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "status": "skipped",
        "case_id": case.case_id,
        "case": _json_safe(asdict(case)),
        "metrics": {
            "started_at": now,
            "ended_at": now,
            "runtime_seconds": 0.0,
            "actual_value_function_calls": 0,
        },
        "skip_reason": reason,
    }


def _write_actual_call_chart(run_dir: Path) -> None:
    aggregate_path = run_dir / "summaries" / "aggregate.json"
    if not aggregate_path.exists():
        return
    aggregates = json.loads(aggregate_path.read_text(encoding="utf-8"))
    points = [
        row
        for row in aggregates
        if row.get("actual_value_function_calls_mean") is not None
        and row.get("high_nmae_mean") is not None
        and row.get("algorithm") in {"mc", "dtg_shapley"}
    ]
    if not points:
        return
    width, height = 920, 560
    left, top, plot_width, plot_height = 90, 70, 760, 400
    max_x = max(float(row["actual_value_function_calls_mean"]) for row in points)
    max_y = max(float(row["high_nmae_mean"]) for row in points) or 1.0
    colors = {"mc": "#2563eb", "dtg_shapley": "#dc2626"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#20242b}.axis{stroke:#59636f}.label{font-size:13px}.title{font-size:20px;font-weight:700}</style>',
        '<text x="70" y="34" class="title">High-field accuracy vs actual value-function calls</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>',
        f'<text x="{left + plot_width / 2 - 85}" y="{height - 35}" class="label">Actual value-function calls</text>',
        f'<text x="18" y="{top + plot_height / 2}" class="label" transform="rotate(-90 18 {top + plot_height / 2})">High-member NMAE</text>',
    ]
    for row in points:
        calls = float(row["actual_value_function_calls_mean"])
        error = float(row["high_nmae_mean"])
        x = left + plot_width * calls / max(max_x, 1.0)
        y = top + plot_height * (1.0 - error / max_y)
        algorithm = row["algorithm"]
        quantifier = row.get("value_quantifier", "dataset_default")
        quantifier_label = "MI" if quantifier == "mutual_information" else "HNB"
        label = (
            f"{row['dataset']} n={row['feature_subset_count']} "
            f"{quantifier_label}"
        )
        svg.extend(
            [
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{colors[algorithm]}"/>',
                f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" class="label">{label}</text>',
            ]
        )
    svg.append('</svg>')
    output_path = run_dir / "charts" / "actual_value_calls.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def _refresh(run_dir: Path) -> tuple[int, int]:
    counts = refresh_outputs(run_dir)
    _write_actual_call_chart(run_dir)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "pilot", "overnight", "full"),
        default="overnight",
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--refresh-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_every <= 0:
        raise ValueError("--refresh-every must be greater than zero.")
    run_dir = args.output_dir / args.run_name
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        total, successful = _refresh(run_dir)
        print(f"Rebuilt outputs from {successful}/{total} successful cases.")
        return

    cases = build_cases(args.profile)
    audits = _audit_case_map(cases)
    manifest = {
        "schema_version": 1,
        "run_name": args.run_name,
        "profile": args.profile,
        "created_or_refreshed_at": datetime.now(timezone.utc).isoformat(),
        "max_hours": args.max_hours,
        "case_count": len(cases),
        "algorithm_version": "dtg_shapley_structured_text",
        "primary_cost_metric": "actual_value_function_calls",
        "value_quantifier_execution_order": [
            "dataset_default",
            "mutual_information",
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "scipy", "scikit-learn")
            },
        },
        "cases": [
            _json_safe(asdict(case)) | {"case_id": case.case_id}
            for case in cases
        ],
    }
    _atomic_json(run_dir / "manifest.json", manifest)
    deadline = time.perf_counter() + args.max_hours * 3600.0
    completed_since_refresh = 0
    for position, case in enumerate(cases, start=1):
        if time.perf_counter() >= deadline:
            print("Time budget reached before starting the next case.")
            break
        output_path = raw_dir / f"{case.case_id}.json"
        if output_path.exists():
            previous = json.loads(output_path.read_text(encoding="utf-8"))
            if previous.get("status") == "success" or not args.rerun_failed:
                print(f"[{position}/{len(cases)}] reuse {case.case_id}")
                continue
        if case.algorithm != "value_audit":
            audit_case = audits[case.condition_key]
            allowed, reason = _audit_allows_shapley(
                raw_dir / f"{audit_case.case_id}.json"
            )
            if not allowed:
                result = _skipped_result(case, reason)
                _atomic_json(output_path, result)
                print(f"[{position}/{len(cases)}] skipped: {reason}")
                continue
        print(f"[{position}/{len(cases)}] run {case.case_id}")
        result = run_case(case, run_dir / "dataset_cache")
        _atomic_json(output_path, result)
        completed_since_refresh += 1
        if completed_since_refresh >= args.refresh_every:
            _refresh(run_dir)
            completed_since_refresh = 0
        calls = result.get("metrics", {}).get("actual_value_function_calls")
        print(
            f"  {result['status']} in "
            f"{result['metrics']['runtime_seconds']:.3f}s; actual calls={calls}"
        )
    total, successful = _refresh(run_dir)
    print(f"Finished with {successful}/{total} successful persisted cases.")
    print(f"Results: {run_dir}")
    print(f"Actual-call chart: {run_dir / 'charts' / 'actual_value_calls.svg'}")


if __name__ == "__main__":
    main()
