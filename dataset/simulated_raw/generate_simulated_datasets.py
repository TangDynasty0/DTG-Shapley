"""Generate the raw simulated datasets reported in the DTG-Shapley paper.

The script is self-contained apart from NumPy and scikit-learn. It exports the
synthetic cooperative-game definitions and the three Iris datasets augmented
with eight independent binary weak fields as human- and machine-readable CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.datasets import load_iris


IRIS_WEAK_FIELD_SEEDS = (11, 23, 42)
IRIS_WEAK_FIELD_COUNT = 8


@dataclass(frozen=True)
class SyntheticGame:
    experiment_id: str
    scenario: str
    low_count: int
    interaction_order: int
    member_names: tuple[str, ...]
    additive_values: tuple[float, ...]
    interactions: tuple[tuple[tuple[int, ...], float], ...]

    def exact_shapley(self) -> tuple[float, ...]:
        values = list(self.additive_values)
        for members, bonus in self.interactions:
            share = bonus / len(members)
            for index in members:
                values[index] += share
        return tuple(values)


def build_contextual_tail(low_count: int) -> SyntheticGame:
    if low_count < 4:
        raise ValueError("contextual_tail requires at least four low members")
    high_additive = (0.26, 0.15, 0.13, 0.09)
    low_additive = (0.08 / low_count,) * low_count
    interactions: list[tuple[tuple[int, ...], float]] = [
        ((0, 1), 0.16),
        ((1, 2, 3), 0.09),
    ]
    interactions.extend(
        ((2, 4 + offset), 0.04 / low_count) for offset in range(low_count)
    )
    return SyntheticGame(
        experiment_id=f"L1_contextual_tail_low{low_count}",
        scenario="contextual_tail",
        low_count=low_count,
        interaction_order=0,
        member_names=tuple(
            [*(f"high_{index + 1}" for index in range(4)),
             *(f"low_{index + 1}" for index in range(low_count))]
        ),
        additive_values=(*high_additive, *low_additive),
        interactions=tuple(interactions),
    )


def build_synergy_tail(order: int, low_count: int = 16) -> SyntheticGame:
    if order < 2 or low_count < order:
        raise ValueError("invalid hidden-interaction configuration")
    high_values = (0.34, 0.24, 0.17)
    interaction_bonus = 0.06 * order
    noise_count = low_count - order
    noise_mass = 1.0 - sum(high_values) - interaction_bonus
    additive_values = (
        *high_values,
        *((0.0,) * order),
        *((noise_mass / noise_count,) * noise_count if noise_count else ()),
    )
    member_names = tuple(
        [*(f"high_{index + 1}" for index in range(len(high_values))),
         *(f"hidden_interaction_{index + 1}" for index in range(order)),
         *(f"low_{index + 1}" for index in range(noise_count))]
    )
    interaction_members = tuple(
        range(len(high_values), len(high_values) + order)
    )
    return SyntheticGame(
        experiment_id=f"L2_hidden_interaction_order{order}_low{low_count}",
        scenario=f"synergy_tail{order}",
        low_count=low_count,
        interaction_order=order,
        member_names=member_names,
        additive_values=additive_values,
        interactions=((interaction_members, interaction_bonus),),
    )


def build_balanced_interaction() -> SyntheticGame:
    count = 12
    pair_bonus = 0.03
    additive_values = (1.0 / count - pair_bonus,) * count
    interactions = tuple(
        ((index, (index + 1) % count), pair_bonus) for index in range(count)
    )
    return SyntheticGame(
        experiment_id="G1_balanced_interaction",
        scenario="balanced_interaction",
        low_count=0,
        interaction_order=0,
        member_names=tuple(f"ordinary_{index + 1}" for index in range(count)),
        additive_values=additive_values,
        interactions=interactions,
    )


def paper_games() -> tuple[SyntheticGame, ...]:
    return (
        *(build_contextual_tail(count) for count in (8, 16, 32)),
        build_synergy_tail(2),
        build_synergy_tail(3),
        build_balanced_interaction(),
    )


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def format_float(value: float) -> str:
    return format(float(value), ".17g")


def generate_game_tables(output_dir: Path) -> list[tuple[Path, int, str]]:
    games = paper_games()
    member_rows: list[Sequence[object]] = []
    interaction_rows: list[Sequence[object]] = []
    for game in games:
        exact_values = game.exact_shapley()
        for index, (name, additive, exact) in enumerate(
            zip(game.member_names, game.additive_values, exact_values)
        ):
            member_rows.append(
                (
                    game.experiment_id,
                    game.scenario,
                    game.low_count,
                    game.interaction_order,
                    index,
                    name,
                    format_float(additive),
                    format_float(exact),
                )
            )
        for interaction_index, (members, bonus) in enumerate(game.interactions, 1):
            interaction_rows.append(
                (
                    game.experiment_id,
                    interaction_index,
                    "|".join(str(index) for index in members),
                    "|".join(game.member_names[index] for index in members),
                    len(members),
                    format_float(bonus),
                )
            )

    member_path = output_dir / "simulated_games_members.csv"
    member_count = write_csv(
        member_path,
        (
            "experiment_id",
            "scenario",
            "low_count",
            "interaction_order",
            "member_index_zero_based",
            "member_name",
            "additive_value",
            "exact_shapley_value",
        ),
        member_rows,
    )
    interaction_path = output_dir / "simulated_games_interactions.csv"
    interaction_count = write_csv(
        interaction_path,
        (
            "experiment_id",
            "interaction_id",
            "member_indices_zero_based",
            "member_names",
            "interaction_order",
            "interaction_bonus",
        ),
        interaction_rows,
    )
    return [
        (member_path, member_count, "Member-level definitions of all simulated games"),
        (
            interaction_path,
            interaction_count,
            "Unanimity-interaction definitions of all simulated games",
        ),
    ]


def generate_iris_tables(output_dir: Path) -> list[tuple[Path, int, str]]:
    iris = load_iris()
    feature_names = [
        name.replace(" (cm)", "").replace(" ", "_").replace("/", "_").replace("-", "_")
        for name in iris.feature_names
    ]
    target_names = tuple(str(name) for name in iris.target_names)
    generated: list[tuple[Path, int, str]] = []
    for seed in IRIS_WEAK_FIELD_SEEDS:
        rng = np.random.default_rng(seed)
        # Match the experiment framework's feature-by-sample generation order,
        # then transpose to the conventional sample-by-feature CSV layout.
        weak_fields = rng.integers(
            1,
            3,
            size=(IRIS_WEAK_FIELD_COUNT, len(iris.data)),
            dtype=np.int8,
        ).T
        path = output_dir / f"iris_with_weak_fields_seed{seed}.csv"
        rows = (
            (
                sample_index,
                *(format_float(value) for value in iris.data[sample_index]),
                *(int(value) for value in weak_fields[sample_index]),
                int(iris.target[sample_index]),
                target_names[int(iris.target[sample_index])],
            )
            for sample_index in range(len(iris.data))
        )
        row_count = write_csv(
            path,
            (
                "sample_index_zero_based",
                *feature_names,
                *(f"synthetic_low_{index + 1}" for index in range(IRIS_WEAK_FIELD_COUNT)),
                "target_code",
                "target_label",
            ),
            rows,
        )
        generated.append(
            (
                path,
                row_count,
                f"Complete Iris data with 8 independent binary weak fields; seed={seed}",
            )
        )
    return generated


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = [*generate_game_tables(output_dir), *generate_iris_tables(output_dir)]
    manifest_path = output_dir / "simulated_dataset_manifest.csv"
    manifest_rows = [
        (path.name, rows, sha256(path), description)
        for path, rows, description in generated
    ]
    write_csv(
        manifest_path,
        ("file_name", "data_rows", "sha256", "description"),
        manifest_rows,
    )
    print(f"Generated {len(generated)} raw CSV files in {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_output = (
        script_dir.parent / "dataset" / "simulated_raw"
        if script_dir.name == "code"
        else script_dir
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Output directory (default: {default_output})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments.output_dir)
