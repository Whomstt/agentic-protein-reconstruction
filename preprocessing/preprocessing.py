from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from Bio import SeqIO

from config import cfg

ROOT = Path(__file__).resolve().parent.parent


def repo_path(value: str) -> Path:
    return ROOT / value


def run_filter() -> None:
    data_dir = repo_path(cfg["data"]["raw_uniprot"])
    output_dir = repo_path(cfg["data"]["raw_ecoli_yeast"])
    patterns = ["Escherichia coli", "Saccharomyces cerevisiae"]
    record_iterator = SeqIO.parse(data_dir, "fasta")

    counts = {pattern: 0 for pattern in patterns}
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with output_dir.open("w") as output_handle:
        for record in record_iterator:
            for pattern in patterns:
                if pattern in record.description:
                    counts[pattern] += 1
                    SeqIO.write(record, output_handle, "fasta")
                    break

    for pattern in patterns:
        print(f"{pattern}: {counts[pattern]} records")


def cut_at_positions(
    sequence: str,
    site_positions: list[int],
    missed_cleavage_ratio: float,
) -> list[str]:
    cut_positions = [
        pos
        for pos in site_positions
        if random.random() > missed_cleavage_prob(sequence, pos, missed_cleavage_ratio)
    ]
    fragments: list[str] = []
    prev = 0
    for pos in sorted(cut_positions):
        fragment = sequence[prev:pos]
        if fragment:
            fragments.append(fragment)
        prev = pos
    tail = sequence[prev:]
    if tail:
        fragments.append(tail)
    return fragments


def missed_cleavage_prob(sequence: str, pos: int, base_ratio: float) -> float:
    if pos <= 0:
        return base_ratio
    if sequence[pos - 1] == "K":
        return min(base_ratio * 1.5, 1.0)
    return base_ratio


def trypsin_digest(sequence: str, missed_cleavage_ratio: float) -> list[str]:
    cleavage_pattern = r"(?<=[KR])(?!P)"
    site_positions = [m.end() for m in re.finditer(cleavage_pattern, sequence)]
    return cut_at_positions(sequence, site_positions, missed_cleavage_ratio)


def lys_c_digest(sequence: str, missed_cleavage_ratio: float) -> list[str]:
    cleavage_pattern = r"(?<=K)"
    site_positions = [m.end() for m in re.finditer(cleavage_pattern, sequence)]
    return cut_at_positions(sequence, site_positions, missed_cleavage_ratio)


def asp_n_digest(sequence: str, missed_cleavage_ratio: float) -> list[str]:
    cleavage_pattern = r"(?=[D])"
    site_positions = [m.start() for m in re.finditer(cleavage_pattern, sequence)]
    return cut_at_positions(sequence, site_positions, missed_cleavage_ratio)


def glu_c_digest(sequence: str, missed_cleavage_ratio: float) -> list[str]:
    cleavage_pattern = r"(?<=[DE])"
    site_positions = [m.end() for m in re.finditer(cleavage_pattern, sequence)]
    return cut_at_positions(sequence, site_positions, missed_cleavage_ratio)


def run_fragment() -> None:
    data_dir = repo_path(cfg["data"]["raw_ecoli_yeast"])
    ecoli_output_file = repo_path(cfg["data"]["fragmented_ecoli"])
    yeast_output_file = repo_path(cfg["data"]["fragmented_yeast"])
    mixture_output_file = repo_path(cfg["data"]["fragmented_mixture"])
    records = list(SeqIO.parse(data_dir, "fasta"))
    ecoli = [r for r in records if "Escherichia coli" in r.description]
    yeast = [r for r in records if "Saccharomyces cerevisiae" in r.description]

    random.seed(cfg["misc"]["seed"])
    missed_cleavage_ratio = cfg["data"]["missed_cleavage_ratio"]
    sample_count = cfg["data"].get("sample_count", 1)

    def generate_fragment_samples(sequence: str) -> list[list[str]]:
        return [
            trypsin_digest(sequence, missed_cleavage_ratio) for _ in range(sample_count)
        ]

    def flatten_unique(samples: list[list[str]]) -> list[str]:
        unique_fragments: list[str] = []
        seen: set[str] = set()
        for sample in samples:
            for fragment in sample:
                if fragment not in seen:
                    seen.add(fragment)
                    unique_fragments.append(fragment)
        return unique_fragments

    fragmented_ecoli = []
    fragmented_yeast = []
    fragmented_mixture = []

    for record in ecoli:
        fragment_samples = generate_fragment_samples(str(record.seq))
        fragments = flatten_unique(fragment_samples)
        random.shuffle(fragments)
        fragmented_ecoli.append(
            {
                "ecoli_original": str(record.seq),
                "fragments": fragments,
                "fragment_samples": fragment_samples,
                "num_fragments": len(fragments),
                "sample_count": sample_count,
                "missed_cleavage_ratio": missed_cleavage_ratio,
            }
        )

    for record in yeast:
        fragment_samples = generate_fragment_samples(str(record.seq))
        fragments = flatten_unique(fragment_samples)
        random.shuffle(fragments)
        fragmented_yeast.append(
            {
                "yeast_original": str(record.seq),
                "fragments": fragments,
                "fragment_samples": fragment_samples,
                "num_fragments": len(fragments),
                "sample_count": sample_count,
                "missed_cleavage_ratio": missed_cleavage_ratio,
            }
        )

    for record_ecoli, record_yeast in zip(ecoli, yeast):
        ecoli_samples = generate_fragment_samples(str(record_ecoli.seq))
        yeast_samples = generate_fragment_samples(str(record_yeast.seq))
        mixed_samples = ecoli_samples + yeast_samples
        mixed_fragments = flatten_unique(mixed_samples)
        random.shuffle(mixed_fragments)
        fragmented_mixture.append(
            {
                "ecoli_original": str(record_ecoli.seq),
                "yeast_original": str(record_yeast.seq),
                "fragments": mixed_fragments,
                "fragment_samples": mixed_samples,
                "num_fragments": len(mixed_fragments),
                "sample_count": sample_count,
                "missed_cleavage_ratio": missed_cleavage_ratio,
            }
        )

    for path, data in [
        (ecoli_output_file, fragmented_ecoli),
        (yeast_output_file, fragmented_yeast),
        (mixture_output_file, fragmented_mixture),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for record in data:
                handle.write(json.dumps(record) + "\n")
        print(f"Wrote {len(data)} records to {path}")


def split_data(
    input_file: Path, train_split: Path, test_split: Path, test_ratio: float
) -> None:
    with input_file.open("r") as handle:
        lines = handle.readlines()

    random.shuffle(lines)

    split_index = int(len(lines) * (1 - test_ratio))
    train_lines = lines[:split_index]
    test_lines = lines[split_index:]

    train_split.parent.mkdir(parents=True, exist_ok=True)
    test_split.parent.mkdir(parents=True, exist_ok=True)

    with train_split.open("w") as handle:
        handle.writelines(train_lines)

    with test_split.open("w") as handle:
        handle.writelines(test_lines)


def run_split() -> None:
    random.seed(cfg["misc"]["seed"])

    ecoli_input_file = repo_path(cfg["data"]["fragmented_ecoli"])
    yeast_input_file = repo_path(cfg["data"]["fragmented_yeast"])
    mixture_input_file = repo_path(cfg["data"]["fragmented_mixture"])
    ecoli_train_split = repo_path(cfg["data"]["ecoli_train_split"])
    yeast_train_split = repo_path(cfg["data"]["yeast_train_split"])
    mixture_train_split = repo_path(cfg["data"]["mixture_train_split"])
    ecoli_test_split = repo_path(cfg["data"]["ecoli_test_split"])
    yeast_test_split = repo_path(cfg["data"]["yeast_test_split"])
    mixture_test_split = repo_path(cfg["data"]["mixture_test_split"])
    test_ratio = cfg["data"]["test_ratio"]

    split_data(ecoli_input_file, ecoli_train_split, ecoli_test_split, test_ratio)
    split_data(yeast_input_file, yeast_train_split, yeast_test_split, test_ratio)
    split_data(mixture_input_file, mixture_train_split, mixture_test_split, test_ratio)


def main() -> None:
    os.chdir(ROOT)
    run_filter()
    run_fragment()
    run_split()


if __name__ == "__main__":
    main()
