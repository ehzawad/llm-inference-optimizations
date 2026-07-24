#!/usr/bin/env python3
"""Generate deterministic prefill-heavy and decode-heavy prompt corpora.

The balanced short-chat corpus (~256-tok prompt, 256-tok generation) lives in
prompts/short-chat.jsonl. To isolate the prefill vs decode phases in the
multi-GPU study we also need two extreme shapes:

  * prefill-heavy: a LONG prompt with a SHORT generation. Stresses the prompt
    ingestion / KV-fill path; TTFT dominates.
  * decode-heavy:  a SHORT prompt with a LONG generation. Stresses the
    autoregressive decode path; steady-state tok/s dominates.

Each prompt carries a unique nonce so prompt caching cannot silently dedupe
them. Domain matches the adapter (legal contract-intake ops) and is benign.

Usage: make_shapes.py [--outdir prompts] [--count 512]
                      [--long-sentences 80] [--short-words 12]

The defaults regenerate the exact committed corpora byte-for-byte, so the
prefill/decode workloads are reproducible without the files being committed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SENTENCES = [
    "Review the following contract passage and identify the controlling clause family.",
    "The vendor agrees to maintain production content within the specified jurisdictions.",
    "Payment terms are net thirty days from the date of the accepted invoice.",
    "Any escrow arrangement must name a neutral third-party agent acceptable to both sides.",
    "Data residency obligations require storage and processing to remain within named regions.",
    "The parties acknowledge that confidential information survives termination of the agreement.",
    "Indemnification is limited to direct damages and excludes consequential losses.",
    "Renewal is automatic for successive twelve-month terms unless notice is given in writing.",
    "Termination for convenience requires sixty days of advance written notice to the counterparty.",
    "Service levels are measured monthly and reported through the standard operations dashboard.",
    "Intellectual property created under this statement of work vests in the commissioning party.",
    "Warranties are provided on an as-is basis except where local law prohibits such exclusion.",
    "The governing law is that of the stated jurisdiction without regard to conflict rules.",
    "Assignment of the agreement requires the prior written consent of the non-assigning party.",
    "Force majeure suspends performance for events beyond the reasonable control of a party.",
    "Audit rights permit one inspection per year upon reasonable advance written notice.",
    "Limitation of liability caps aggregate damages at the fees paid in the prior twelve months.",
    "The processor shall implement appropriate technical and organizational security measures.",
    "Change orders must be documented and signed before any additional work is undertaken.",
    "Dispute resolution proceeds first through good-faith negotiation and then binding arbitration.",
]

CLAUSE_CHOICES = (
    "Choose exactly one clause family: data_residency, escrow, payment_terms, "
    "indemnification, termination, confidentiality, liability_cap, audit_rights."
)


def long_prompt(i: int, sentences: int) -> str:
    n = len(SENTENCES)
    picks = [SENTENCES[(i * 7 + k * 3) % n] for k in range(sentences)]
    body = " ".join(picks)
    nonce = f"[req-{i:06d}]"
    return (
        f"{nonce} You are an intake assistant for a contracts operations team. "
        f"Read the full passage carefully before answering. {body} "
        f"Based strictly on the passage above, decide the single best label. "
        f"{CLAUSE_CHOICES} Answer with only the label."
    )


def short_prompt(i: int, words: int) -> str:
    nonce = f"[req-{i:06d}]"
    return f"{nonce} Write a {words}-sentence plain-language summary of a standard NDA."


def write_corpus(path: Path, count: int, builder) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps({"id": i, "prompt": builder(i)}) + "\n")
    print(f"[shapes] wrote {count} prompts -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="prompts")
    ap.add_argument("--count", type=int, default=512)
    ap.add_argument("--long-sentences", type=int, default=80,
                    help="sentences per prefill-heavy prompt (~15 tok each); "
                         "80 reproduces the committed ~1240-token workload")
    ap.add_argument("--short-words", type=int, default=12)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    write_corpus(outdir / "long-prompt.jsonl", args.count,
                 lambda i: long_prompt(i, args.long_sentences))
    write_corpus(outdir / "short-prompt.jsonl", args.count,
                 lambda i: short_prompt(i, args.short_words))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
