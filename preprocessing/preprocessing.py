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


def _selected_organism() -> str:
    return cfg["data"].get("organism", "ecoli")


def _organism_pattern() -> str:
    organism = _selected_organism()
    if organism == "yeast":
        return "Saccharomyces cerevisiae"
    if organism == "mixture":
        return ""
    return "Escherichia coli"


def _filter_patterns() -> list[str]:
    organism = _selected_organism()
    if organism == "mixture":
        return ["Escherichia coli", "Saccharomyces cerevisiae"]
    return [_organism_pattern()]


def _original_key() -> str:
    return cfg["data"].get("active_target_key", "ecoli_original")


def _active_fragmented_output() -> Path:
    return repo_path(cfg["data"]["active_fragmented_split"])


def _meta_path() -> Path:
    return _active_fragmented_output().with_suffix(".meta.json")


def _preprocessing_fingerprint() -> dict:
    """The subset of config.yaml that changes what preprocessing writes to disk.
    Compared against the sidecar .meta.json saved next to the fragmented output
    to decide whether the file on disk still matches the active config."""
    return {
        "organism": cfg["data"].get("organism"),
        "replica_count": cfg["data"].get("replica_count"),
        "missed_cleavage_ratio": cfg["data"].get("missed_cleavage_ratio"),
    }


def is_stale() -> bool:
    output = _active_fragmented_output()
    meta_path = _meta_path()
    if not output.exists() or not meta_path.exists():
        return True
    with meta_path.open() as f:
        saved_fingerprint = json.load(f)
    return saved_fingerprint != _preprocessing_fingerprint()


def ensure_fresh_dataset() -> None:
    """Regenerates the active organism's fragmented dataset if organism,
    replica_count, or missed_cleavage_ratio have changed since it was last
    written. Safe to call before every run (sweep combo or standalone) since
    it's a no-op when the dataset already matches the active config."""
    if is_stale():
        organism = cfg["data"].get("organism")
        print(
            f"Preprocessing inputs changed for organism='{organism}' — regenerating fragmented dataset..."
        )
        main()
    else:
        print("Fragmented dataset already matches the active config; skipping preprocessing.")


_GENE_NAME_PATTERN = re.compile(r"GN=(\S+)")


def _gene_key(record) -> str:
    """UniProt entries include many near-duplicate strains of the same gene
    (e.g. E. coli K12, O157:H7, O6:H1, ... all carrying the same "yfbR" gene).
    Left undeduped, a random sample is dominated by whichever strains happen
    to be over-represented rather than by distinct proteins. GN=<gene> is the
    dedup key; entries without one (~0.3% of records) fall back to their
    accession, which is always unique so they're kept as-is."""
    match = _GENE_NAME_PATTERN.search(record.description)
    return match.group(1) if match else record.id


def _dedupe_by_gene(records: list) -> list:
    """First-seen-wins per gene, preserving original file order."""
    seen: set[str] = set()
    unique = []
    for record in records:
        key = _gene_key(record)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def run_filter() -> None:
    data_dir = repo_path(cfg["data"]["raw_uniprot"])
    output_dir = repo_path(cfg["data"]["raw_ecoli_yeast"])
    patterns = _filter_patterns()
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
    records = list(SeqIO.parse(data_dir, "fasta"))
    organism = _selected_organism()
    pattern = _organism_pattern()
    if organism == "mixture":
        ecoli = _dedupe_by_gene([r for r in records if "Escherichia coli" in r.description])
        yeast = _dedupe_by_gene([r for r in records if "Saccharomyces cerevisiae" in r.description])
    else:
        selected_records = _dedupe_by_gene([r for r in records if pattern in r.description])

    missed_cleavage_ratio = cfg["data"]["missed_cleavage_ratio"]
    replica_count = cfg["data"].get("replica_count", cfg["data"].get("sample_count", 1))

    def generate_fragment_samples(sequence: str) -> list[list[str]]:
        return [
            trypsin_digest(sequence, missed_cleavage_ratio)
            for _ in range(replica_count)
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

    output_file = _active_fragmented_output()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Stream each protein's record straight to disk instead of accumulating the
    # whole dataset in a list first. At high replica_count (e.g. 100) that list
    # held every gene's fragment_samples in RAM at once and was a multi-GB spike
    # that could OOM a 32 GB box; writing per-record keeps only one protein's
    # fragments live at a time.
    written = 0
    with output_file.open("w") as handle:
        if organism == "mixture":
            for record_ecoli, record_yeast in zip(ecoli, yeast):
                ecoli_samples = generate_fragment_samples(str(record_ecoli.seq))
                yeast_samples = generate_fragment_samples(str(record_yeast.seq))
                mixed_samples = ecoli_samples + yeast_samples
                mixed_fragments = flatten_unique(mixed_samples)
                random.shuffle(mixed_fragments)
                record = {
                    "ecoli_original": str(record_ecoli.seq),
                    "yeast_original": str(record_yeast.seq),
                    "target_reconstruction": str(record_ecoli.seq)
                    + str(record_yeast.seq),
                    "fragments": mixed_fragments,
                    "fragment_samples": mixed_samples,
                    "num_fragments": len(mixed_fragments),
                    "replica_count": replica_count,
                    "missed_cleavage_ratio": missed_cleavage_ratio,
                }
                handle.write(json.dumps(record) + "\n")
                written += 1
        else:
            for record in selected_records:
                fragment_samples = generate_fragment_samples(str(record.seq))
                fragments = flatten_unique(fragment_samples)
                random.shuffle(fragments)
                out = {
                    _original_key(): str(record.seq),
                    "target_reconstruction": str(record.seq),
                    "fragments": fragments,
                    "fragment_samples": fragment_samples,
                    "num_fragments": len(fragments),
                    "replica_count": replica_count,
                    "missed_cleavage_ratio": missed_cleavage_ratio,
                }
                handle.write(json.dumps(out) + "\n")
                written += 1
    print(f"Wrote {written} records to {output_file}")

    with _meta_path().open("w") as handle:
        json.dump(_preprocessing_fingerprint(), handle)


def main() -> None:
    os.chdir(ROOT)
    run_filter()
    run_fragment()


if __name__ == "__main__":
    main()
