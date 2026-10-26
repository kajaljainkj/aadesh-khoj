"""The headline experiment: how much does each pipeline stage improve retrieval?

  A  naive RAG     raw PDF text (no legacy-font fix, no OCR), vector search, query as typed
  B  + font fix    Kruti Dev converted, scanned pages OCR'd
  C  + hybrid      B + keyword search fused with RRF
  D  + rewrite     C + LLM rewrites the query into formal Hindi   (needs the LLM)
  E  + rerank      D + cross-encoder reranks the top 20           (needs the ~2 GB reranker model)

Run from the project root:
    python -m eval.run_eval              # all five (D and E make one LLM call per question)
    python -m eval.run_eval --no-llm     # A-C only: fast, no LLM needed
    python -m eval.run_eval --no-rerank  # skip E (no reranker model download)
    python -m eval.run_eval --skip-ingest   # reuse the eval collections from a previous run

Writes eval/results.csv (per question) and eval/results.md (summary table for your README).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CONFIGS = [
    # name, label, collection, mode, strategy, rerank
    ("A", "naive (raw text)", "eval_raw", "vector", "direct", False),
    ("B", "+ font fix & OCR", "eval_fixed", "vector", "direct", False),
    ("C", "+ hybrid search", "eval_fixed", "hybrid", "direct", False),
    ("D", "+ query rewrite", "eval_fixed", "hybrid", "rewrite", False),
    ("E", "+ reranker", "eval_fixed", "hybrid", "rewrite", True),
]
LANGS = ["en", "hi", "hinglish"]


class _fixed_rewrite:
    """Temporarily make rewrite_query return a known value (no extra LLM call, same query)."""

    def __init__(self, value: str):
        self.value = value

    def __enter__(self):
        import app.rag as rag

        self._orig = rag.rewrite_query
        rag.rewrite_query = lambda q: self.value
        return self

    def __exit__(self, *exc):
        import app.rag as rag

        rag.rewrite_query = self._orig


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_hit_rank(hits: list[dict], expected: str) -> int | None:
    for rank, h in enumerate(hits, 1):
        if Path(h["source"]).stem == expected:
            return rank
    return None


def summarize(rows: list[dict], configs, k: int) -> str:
    agg = defaultdict(lambda: [0, 0, 0.0])  # n, hits, rr
    for r in rows:
        for key in ((r["config"], r["lang"]), (r["config"], "all")):
            agg[key][0] += 1
            agg[key][1] += r["hit"]
            agg[key][2] += r["rr"]
    cols = LANGS + ["all"]
    lines = [f"| Config | " + " | ".join(f"{c} R@{k}" for c in cols) + " | all MRR |",
             "|---|" + "---|" * (len(cols) + 1)]
    for name, label, *_ in configs:
        cells = []
        for c in cols:
            n, h, _ = agg.get((name, c), (0, 0, 0))
            cells.append(f"{h / n:.2f}" if n else "-")
        n, _, rr = agg.get((name, "all"), (0, 0, 0))
        cells.append(f"{rr / n:.2f}" if n else "-")
        lines.append(f"| {name}. {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--docs", default=str(ROOT / "data" / "samples"))
    ap.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    ap.add_argument("--no-llm", action="store_true", help="skip configs D and E")
    ap.add_argument("--no-rerank", action="store_true", help="skip config E")
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args()

    from app.ingest import ingest_folder
    from app.rag import retrieve

    configs = [c for c in CONFIGS
               if not (args.no_llm and c[4] == "rewrite") and not (args.no_rerank and c[5])]
    if not args.skip_ingest:
        print("Ingesting baseline (no font fix, no OCR) ...")
        ingest_folder(Path(args.docs), "eval_raw", fix_legacy=False, ocr=False, quiet=True)
        print("Ingesting full pipeline ...")
        s = ingest_folder(Path(args.docs), "eval_fixed", fix_legacy=True, ocr=True, quiet=True)
        print(f"  extraction methods: {s['methods']}")

    questions = load_questions(Path(args.questions))
    rows = []
    rewrites: dict[str, str] = {}   # reuse D's rewrite in E, so E differs from D only by reranking
    for name, label, collection, mode, strategy, rerank in configs:
        for q in questions:
            if strategy == "rewrite" and q["id"] in rewrites:
                with _fixed_rewrite(rewrites[q["id"]]):
                    query, hits = retrieve(q["question"], strategy, mode, args.k, collection, rerank)
            else:
                query, hits = retrieve(q["question"], strategy, mode, args.k, collection, rerank)
                if strategy == "rewrite":
                    rewrites[q["id"]] = query
            rank = first_hit_rank(hits, q["expected_source"])
            rows.append({"config": name, "id": q["id"], "lang": q["lang"], "question": q["question"],
                         "search_query": query, "expected": q["expected_source"],
                         "top1": Path(hits[0]["source"]).stem if hits else "", "rank": rank or "",
                         "hit": int(rank is not None), "rr": round(1 / rank, 4) if rank else 0.0})
        print(f"  {name}. {label}: done")

    with (HERE / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table = summarize(rows, configs, args.k)
    (HERE / "results.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nPer-question results: {HERE / 'results.csv'}")


if __name__ == "__main__":
    main()
