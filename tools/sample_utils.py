from __future__ import annotations


def normalize_fragment_samples(fragment_samples):
    if not fragment_samples:
        return []
    first = fragment_samples[0]
    if isinstance(first, str):
        return [list(fragment_samples)]
    return [list(sample) for sample in fragment_samples]


def primary_fragments(fragment_samples):
    normalized = normalize_fragment_samples(fragment_samples)
    return normalized[0] if normalized else []
