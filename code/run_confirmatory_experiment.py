"""Compact confirmatory benchmark for the DTG-Shapley paper.

The matrix separates target low-contribution games from ordinary games and
compares DTG-Shapley with implementation-matched MC, antithetic MC, TMC,
coalition-size stratified sampling, and KernelSHAP. Cases are persisted one at
a time and the default wall-clock limit is below five hours.
"""

from __future__ import annotations

import argparse
import html
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from experiment_core import (
    ExperimentCase,
    _atomic_json,
    _json_safe,
    refresh_outputs,
    run_case,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reproduced_results"
DEFAULT_RUN_NAME = "dtg_confirmatory"
DEFAULT_MAX_HOURS = 4.75
METHODS = (
    "mc",
    "antithetic_mc",
    "tmc",
    "stratified",
    "kernel_shap",
    "dtg_shapley",
)
METHOD_COLORS = {
    "mc": "#2563eb",
    "antithetic_mc": "#0891b2",
    "tmc": "#ea580c",
    "stratified": "#7c3aed",
    "kernel_shap": "#475569",
    "dtg_shapley": "#dc2626",
}


def _deduplicate(cases: Iterable[ExperimentCase]) -> list[ExperimentCase]:
    return list({case.case_id: case for case in cases}.values())


def _synthetic_block(
    experiment: str,
    scenario: str,
    seeds: Sequence[int],
    low_count: int,
    paths: int,
    *,
    interaction_order: int = 0,
) -> list[ExperimentCase]:
    reference_group = f"{scenario}_low{low_count}_order{interaction_order}"
    common: dict[str, Any] = {
        "experiment": experiment,
        "scenario": scenario,
        "dataset": "synthetic",
        "artificial_low_count": low_count,
        "contribution_threshold": 0.04,
        "interaction_order": interaction_order,
        "threshold_mode": "relative",
        "relative_threshold_factor": 0.5,
        "implementation_version": "dtg_paper_confirmatory",
        "reference_group": reference_group,
    }
    cases = [
        ExperimentCase(
            algorithm="analytic_reference",
            seed=seeds[0],
            equivalent_mc_paths=0,
            reference=True,
            **common,
        )
    ]
    for seed in seeds:
        cases.extend(
            ExperimentCase(
                algorithm=algorithm,
                seed=seed,
                equivalent_mc_paths=paths,
                **common,
            )
            for algorithm in METHODS
        )
    return cases


def _public_general_block(
    dataset: str,
    value_quantifier: str,
    seeds: Sequence[int],
    paths: int,
) -> list[ExperimentCase]:
    reference_group = f"{dataset}_{value_quantifier}_unaltered"
    common: dict[str, Any] = {
        "experiment": "g2_public_general",
        "scenario": "public_compatibility",
        "dataset": dataset,
        "artificial_low_count": 0,
        "contribution_threshold": 0.04,
        "threshold_mode": "relative",
        "relative_threshold_factor": 0.5,
        "value_quantifier": value_quantifier,
        "implementation_version": "dtg_paper_confirmatory",
        "reference_group": reference_group,
    }
    cases = [
        ExperimentCase(
            algorithm="exact",
            seed=seeds[0],
            equivalent_mc_paths=0,
            reference=True,
            **common,
        )
    ]
    for seed in seeds:
        cases.extend(
            ExperimentCase(
                algorithm=algorithm,
                seed=seed,
                equivalent_mc_paths=paths,
                **common,
            )
            for algorithm in METHODS
        )
    return cases


def _public_tail_block(
    seed: int,
    low_count: int,
    paths: int,
) -> list[ExperimentCase]:
    common: dict[str, Any] = {
        "experiment": "l3_public_injected_tail",
        "scenario": "public_tail",
        "dataset": "iris",
        "seed": seed,
        "artificial_low_count": low_count,
        "contribution_threshold": 0.04,
        "threshold_mode": "relative",
        "relative_threshold_factor": 0.5,
        "value_quantifier": "mutual_information",
        "implementation_version": "dtg_paper_confirmatory",
    }
    return [
        ExperimentCase(
            algorithm="exact",
            equivalent_mc_paths=0,
            reference=True,
            **common,
        ),
        *(
            ExperimentCase(
                algorithm=algorithm,
                equivalent_mc_paths=paths,
                **common,
            )
            for algorithm in METHODS
        ),
    ]


def build_cases(profile: str) -> list[ExperimentCase]:
    if profile == "smoke":
        cases = []
        cases.extend(
            _synthetic_block(
                "l1_low_tail_scaling",
                "contextual_tail",
                (11,),
                8,
                20,
            )
        )
        cases.extend(
            _synthetic_block(
                "g1_synthetic_general",
                "balanced_interaction",
                (11,),
                0,
                20,
            )
        )
        return _deduplicate(cases)

    seeds = (11, 23, 42, 67, 89)
    public_seeds = (11, 23, 42)
    cases: list[ExperimentCase] = []

    # Target regime first: individually weak tails with exact analytic truth.
    for low_count in (8, 16, 32):
        cases.extend(
            _synthetic_block(
                "l1_low_tail_scaling",
                "contextual_tail",
                seeds,
                low_count,
                100,
            )
        )
    for order in (2, 3):
        cases.extend(
            _synthetic_block(
                "l2_hidden_interaction_safety",
                f"synergy_tail{order}",
                public_seeds,
                16,
                120,
                interaction_order=order,
            )
        )
    for seed in public_seeds:
        cases.extend(_public_tail_block(seed, 8, 100))

    # General regime: no assumption that a large low-contribution tail exists.
    cases.extend(
        _synthetic_block(
            "g1_synthetic_general",
            "balanced_interaction",
            seeds,
            0,
            100,
        )
    )
    for dataset in ("iris", "wine"):
        for quantifier in ("holdout_naive_bayes", "mutual_information"):
            cases.extend(
                _public_general_block(
                    dataset,
                    quantifier,
                    public_seeds,
                    100,
                )
            )
    return _deduplicate(cases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("smoke", "overnight"), default="overnight"
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--refresh-every", type=int, default=20)
    return parser.parse_args()


def _comparison_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["experiment"],
        row["scenario"],
        row["dataset"],
        row["value_quantifier"],
        row["artificial_low_count"],
        row["interaction_order"],
        row["equivalent_mc_paths"],
        row["effective_contribution_threshold"],
    )


def write_paper_chart(run_dir: Path) -> None:
    aggregate_path = run_dir / "summaries" / "aggregate.json"
    if not aggregate_path.exists():
        return
    rows = json.loads(aggregate_path.read_text(encoding="utf-8"))
    mc_calls = {
        _comparison_key(row): row.get("actual_value_function_calls_mean")
        for row in rows
        if row.get("algorithm") == "mc"
    }
    points = []
    for row in rows:
        if row.get("algorithm") not in METHODS:
            continue
        calls = row.get("actual_value_function_calls_mean")
        baseline = mc_calls.get(_comparison_key(row))
        error = row.get("high_nmae_mean")
        if calls is None or not baseline or error is None:
            continue
        points.append((row, float(calls) / float(baseline), float(error)))
    if not points:
        return

    width, height = 1240, 650
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#20242b}.axis{stroke:#59636f}.grid{stroke:#e2e8f0}.small{font-size:12px}.title{font-size:20px;font-weight:700}.panel{font-size:15px;font-weight:700}</style>',
        '<text x="55" y="32" class="title">Accuracy-cost comparison against measured MC cost</text>',
    ]
    legend_x = 55
    for algorithm in METHODS:
        color = METHOD_COLORS[algorithm]
        svg.extend(
            [
                f'<circle cx="{legend_x}" cy="58" r="5" fill="{color}"/>',
                f'<text x="{legend_x + 10}" y="62" class="small">{algorithm}</text>',
            ]
        )
        legend_x += 70 + 7 * len(algorithm)

    panels = (
        ("l", 65, "Low-contribution target scenarios"),
        ("g", 650, "General-applicability scenarios"),
    )
    for prefix, left, title in panels:
        selected = [item for item in points if item[0]["experiment"].startswith(prefix)]
        if not selected:
            continue
        top, panel_width, panel_height = 105, 525, 450
        max_x = max(1.05, max(item[1] for item in selected) * 1.08)
        max_y = max(item[2] for item in selected) * 1.08 or 1.0
        svg.extend(
            [
                f'<text x="{left}" y="90" class="panel">{title}</text>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}" class="axis"/>',
                f'<line x1="{left}" y1="{top + panel_height}" x2="{left + panel_width}" y2="{top + panel_height}" class="axis"/>',
            ]
        )
        for step in range(5):
            y = top + panel_height * step / 4
            svg.append(
                f'<line x1="{left}" y1="{y}" x2="{left + panel_width}" y2="{y}" class="grid"/>'
            )
        for row, call_ratio, error in selected:
            x = left + panel_width * call_ratio / max_x
            y = top + panel_height * (1.0 - error / max_y)
            label = html.escape(
                f"{row['algorithm']}; {row['scenario']}; {row['dataset']}; "
                f"low={row['artificial_low_count']}; calls/MC={call_ratio:.3f}; "
                f"high NMAE={error:.6f}"
            )
            svg.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{METHOD_COLORS[row["algorithm"]]}" fill-opacity="0.78"><title>{label}</title></circle>'
            )
        mc_x = left + panel_width / max_x
        svg.append(
            f'<line x1="{mc_x:.1f}" y1="{top}" x2="{mc_x:.1f}" y2="{top + panel_height}" stroke="#64748b" stroke-dasharray="5 4"/>'
        )
        svg.append(
            f'<text x="{left + panel_width / 2}" y="{top + panel_height + 35}" text-anchor="middle" class="small">Actual value-function calls / matched MC calls</text>'
        )
    svg.append(
        '<text x="18" y="335" transform="rotate(-90 18 335)" text-anchor="middle" class="small">Reference-high NMAE (lower is better)</text>'
    )
    svg.append('</svg>')
    chart_path = run_dir / "charts" / "paper_comparison.svg"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_hours <= 0:
        raise ValueError("--max-hours must be greater than zero.")
    if args.refresh_every <= 0:
        raise ValueError("--refresh-every must be greater than zero.")

    run_dir = args.output_dir / args.run_name
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        total, successful = refresh_outputs(run_dir)
        write_paper_chart(run_dir)
        print(f"Rebuilt outputs from {successful}/{total} successful cases.")
        return

    cases = build_cases(args.profile)
    manifest = {
        "schema_version": 1,
        "run_name": args.run_name,
        "profile": args.profile,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_hours": args.max_hours,
        "case_count": len(cases),
        "algorithm_version": "dtg_shapley_paper_confirmatory",
        "evidence_blocks": {
            "low_contribution": [
                "l1_low_tail_scaling",
                "l2_hidden_interaction_safety",
                "l3_public_injected_tail",
            ],
            "general_applicability": [
                "g1_synthetic_general",
                "g2_public_general",
            ],
            "comparators": list(METHODS),
        },
        "budget_rule": (
            "MC, antithetic MC, TMC, and DTG receive the same path count. "
            "Stratified sampling uses approximately the same uncached value-"
            "request budget, and KernelSHAP receives n times the path count "
            "in coalition samples. Conclusions use measured value-function "
            "calls and runtime rather than nominal budget alone."
        ),
        "primary_endpoints": [
            "reference-high normalized MAE",
            "reference-high recall",
            "actual value-function calls",
            "wall-clock time",
        ],
        "secondary_endpoints": [
            "all-member normalized MAE",
            "Spearman rank correlation",
            "low-to-high member sampling ratio",
            "TMC truncation rate",
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
        print(f"[{position}/{len(cases)}] run {case.case_id}")
        result = run_case(case, run_dir / "dataset_cache")
        _atomic_json(output_path, result)
        completed_since_refresh += 1
        if completed_since_refresh >= args.refresh_every:
            refresh_outputs(run_dir)
            write_paper_chart(run_dir)
            completed_since_refresh = 0
        print(f"  {result['status']} in {result['metrics']['runtime_seconds']:.3f}s")

    total, successful = refresh_outputs(run_dir)
    write_paper_chart(run_dir)
    print(f"Finished with {successful}/{total} successful persisted cases.")
    print(f"Results: {run_dir}")
    print(f"Chart: {run_dir / 'charts' / 'paper_comparison.svg'}")


if __name__ == "__main__":
    main()
