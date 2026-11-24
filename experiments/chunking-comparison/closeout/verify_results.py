"""Independent re-derivation of the headline counts from the saved machine-readable results.

Reads only ``results/per-question.json``, ``aggregates.json``, ``paired-statistics.json`` and
``builds/*.json`` (plus the sizes of the persisted index files when the index directories are
present). It does not import the experiment package, so it is a second implementation of the
partition counts, the paired contingency table, the exact McNemar p-value and the bootstrap
interval. Nothing is measured or changed: run it, compare, exit non-zero on any disagreement.

    python experiments/chunking-comparison/closeout/verify_results.py [--write]
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experiments" / "chunking-comparison" / "results"
ARMS = "ABCD"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def hit(row: dict, k: int) -> bool:
    rank = row["first_hit_rank"]
    return rank is not None and rank <= k


def partition(hits: dict[str, list[bool]], ids: list[str]) -> dict:
    all_hit = [q for i, q in enumerate(ids) if all(hits[a][i] for a in ARMS)]
    all_miss = [q for i, q in enumerate(ids) if not any(hits[a][i] for a in ARMS)]
    differ = [q for q in ids if q not in all_hit and q not in all_miss]
    assert len(all_hit) + len(all_miss) + len(differ) == len(ids)
    return {"all_arms_hit": all_hit, "all_arms_miss": all_miss, "arms_differ": differ}


def exact_mcnemar(only_x: int, only_y: int) -> float:
    n = only_x + only_y
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(only_x, only_y) + 1))
    return min(1.0, 2 * tail / 2**n)


def main(write: bool) -> int:
    pq = load(RESULTS / "per-question.json")["arms"]
    paired = load(RESULTS / "paired-statistics.json")
    agg = load(RESULTS / "aggregates.json")["arms"]
    ids = [r["question_id"] for r in pq["A"]]
    problems: list[str] = []
    for arm in ARMS:
        if [r["question_id"] for r in pq[arm]] != ids:
            problems.append(f"arm {arm}: question order differs")
    out: dict = {"questions": len(ids), "unique_question_ids": len(set(ids))}
    if len(set(ids)) != len(ids):
        problems.append("duplicate question ids")

    for k in (1, 3, 5, 10):
        hits = {a: [hit(r, k) for r in pq[a]] for a in ARMS}
        out[f"hit@{k}_counts"] = {a: sum(hits[a]) for a in ARMS}
        for a in ARMS:
            saved = agg[a]["binary"][f"hit@{k}"]["count"]
            if saved != sum(hits[a]):
                problems.append(f"hit@{k} {a}: saved {saved} != recomputed {sum(hits[a])}")
        if k in (5, 10):
            p = partition(hits, ids)
            out[f"partition_at_hit@{k}"] = {
                "counts": {name: len(v) for name, v in p.items()},
                **p,
            }

    x = np.array([hit(r, 5) for r in pq["B"]])
    y = np.array([hit(r, 5) for r in pq["A"]])
    table = {
        "both": int((x & y).sum()),
        "only_x": int((x & ~y).sum()),
        "only_y": int((~x & y).sum()),
        "neither": int((~x & ~y).sum()),
    }
    saved = paired["comparisons"]["B-A"]["binary"]["hit@5"]
    if table != saved["table"]:
        problems.append(f"primary table: recomputed {table} != saved {saved['table']}")
    p_val = exact_mcnemar(table["only_x"], table["only_y"])
    if abs(p_val - saved["mcnemar_p"]) > 1e-12:
        problems.append(f"McNemar: recomputed {p_val} != saved {saved['mcnemar_p']}")

    boot = paired["bootstrap"]
    idx = np.random.default_rng(boot["seed"]).integers(
        0, len(ids), size=(boot["resamples"], len(ids))
    )
    diff = x.astype(float) - y.astype(float)
    means = diff[idx].mean(axis=1) * 100.0
    alpha = (1 - boot["confidence"]) / 2
    lo, hi = (float(v) for v in np.quantile(means, [alpha, 1 - alpha]))
    if abs(lo - saved["ci_pp"][0]) > 1e-9 or abs(hi - saved["ci_pp"][1]) > 1e-9:
        problems.append(f"bootstrap CI: recomputed [{lo}, {hi}] != saved {saved['ci_pp']}")
    out["primary_B_vs_A_hit@5"] = {
        "table": table,
        "difference_pp": float(diff.mean() * 100),
        "mcnemar_exact_two_sided_p": p_val,
        "bootstrap_ci_pp": [lo, hi],
        "bootstrap": {
            **boot,
            "method": "paired percentile bootstrap of the mean per-question difference",
        },
        "matches_saved": not problems,
    }

    sizes = {}
    for arm in ARMS:
        build = load(RESULTS / "builds" / f"{arm}.json")
        persisted = build["bytes"]["artifacts"]
        total = sum(persisted.values())
        entry = {
            "files": persisted,
            "sum_of_files": total,
            "saved_index_total": build["bytes"]["index_total"],
        }
        if total != build["bytes"]["index_total"]:
            problems.append(f"index size {arm}: files sum {total} != index_total")
        if agg[arm].get("index_bytes") not in (None, total):
            problems.append(f"index size {arm}: aggregates index_bytes differs")
        index_dir = (
            ROOT / "data" / "experiments" / "chunking-comparison" / "indexes" / build["index_path"]
        )
        if index_dir.is_dir():
            on_disk = {f: (index_dir / f).stat().st_size for f in persisted}
            entry["on_disk_now"] = on_disk
            if on_disk != persisted:
                problems.append(f"index size {arm}: on-disk sizes differ from the record")
        entry["embedding_seconds_recorded"] = build["seconds"]["embed"]
        entry["model_cold_start_seconds_not_included"] = build["model_cold_start_seconds"]
        sizes[arm] = entry
    out["index_sizes"] = sizes
    out["problems"] = problems
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    if write:
        target = Path(__file__).with_name("verification.json")
        if target.exists():
            print(f"{target} exists; not overwriting", file=sys.stderr)
            return 2
        target.write_text(text, encoding="utf-8")
    print(text if not write else f"wrote verification.json; problems: {problems or 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main("--write" in sys.argv))
