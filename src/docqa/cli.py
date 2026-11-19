"""Command line: works on a laptop or a plain Linux VPS, no Docker required.

  python -m docqa.cli synth  --out data
  python -m docqa.cli index  --docs data/docs --acl data/acl.json --out data/index
  python -m docqa.cli ask    --index data/index "What is the fee for ...?" --groups partners
  python -m docqa.cli eval   --index data/index --questions data/questions.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .answer import ExtractiveBackend, HFBackend, OllamaBackend, answer
from .eval import format_report, load_questions, run_eval, save_results
from .index import HashEmbedder, HybridIndex, SentenceTransformerEmbedder
from .ingest import ingest_folder, load_acl
from .synth import write_all


def make_embedder(name: str):
    if name == "none":
        return None
    if name == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(name if name != "st" else "BAAI/bge-small-en-v1.5")


def make_backend(args):
    if args.backend == "extractive":
        return ExtractiveBackend()
    if args.backend == "ollama":
        return OllamaBackend(model=args.model or "qwen2.5:3b")
    return HFBackend(model_id=args.model or "Qwen/Qwen2.5-3B-Instruct", load_in_4bit=args.four_bit)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="docqa")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate synthetic documents + questions")
    s.add_argument("--out", default="data")
    s.add_argument("--seed", type=int, default=7)

    i = sub.add_parser("index", help="ingest documents and build the search index")
    i.add_argument("--docs", required=True)
    i.add_argument("--acl", default=None)
    i.add_argument("--out", default="data/index")
    i.add_argument("--embedder", default="st", help="st | hash | none | <sentence-transformers model>")

    for name in ("ask", "eval"):
        a = sub.add_parser(name)
        a.add_argument("--index", required=True)
        a.add_argument("--embedder", default="st")
        a.add_argument("--backend", default="extractive", choices=["extractive", "hf", "ollama"])
        a.add_argument("--model", default=None)
        a.add_argument("--four-bit", action="store_true")
        a.add_argument("--k", type=int, default=4)
        if name == "ask":
            a.add_argument("question")
            a.add_argument("--groups", default="all", help="comma-separated user groups")
            a.add_argument("--audit", default=None)
        else:
            a.add_argument("--questions", required=True)
            a.add_argument("--limit", type=int, default=None)
            a.add_argument("--out", default="results.json")

    args = p.parse_args(argv)

    if args.cmd == "synth":
        print(write_all(args.out, args.seed))
    elif args.cmd == "index":
        chunks = ingest_folder(args.docs, load_acl(args.acl))
        idx = HybridIndex(make_embedder(args.embedder)).build(chunks)
        idx.save(args.out)
        print(f"Indexed {len(chunks)} chunks from {len({c.doc for c in chunks})} documents -> {args.out}")
    elif args.cmd == "ask":
        idx = HybridIndex.load(args.index, make_embedder(args.embedder))
        ans = answer(args.question, idx, make_backend(args), k=args.k,
                     user_groups=args.groups.split(","), audit_path=args.audit)
        print(ans.text)
        for c in ans.citations:
            print(f"  [{c['n']}] {c['doc']} ({c['loc']})")
    elif args.cmd == "eval":
        idx = HybridIndex.load(args.index, make_embedder(args.embedder))
        result = run_eval(idx, make_backend(args), load_questions(args.questions),
                          k=args.k, limit=args.limit)
        save_results(result, args.out)
        print("\n" + format_report(result))
        Path(args.out).with_suffix(".md").write_text(format_report(result))


if __name__ == "__main__":
    main()
