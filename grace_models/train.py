"""CLI for the GRACE single-modality baselines.

Run from project/grace with the torch env:
    PY=~/miniconda3/envs/torch/bin/python
    $PY -m grace_models.train structured --window 6                         # XGBoost on triage+vitals+CC tokens
    $PY -m grace_models.train structured --window 6 --blocks triage vitals cc medrecon
    $PY -m grace_models.train text --window 6 --source cc                   # TF-IDF + LR on chief complaint
    $PY -m grace_models.train text --window 24 --source radiology           # TF-IDF + LR on radiology reports
    $PY -m grace_models.train text --window 24 --source radiology --encoder biowordvec --vectors /path/BioWordVec.bin
    $PY -m grace_models.train echo-manifest --window 24                     # list echo DICOMs to download
    $PY -m grace_models.train echo --window 24                              # LR on frozen EchoCLIP embeddings

Common flags: --admitted-only (train AND evaluate on admitted stays), --eval-test (score the test split;
leave off during development), --n-boot (bootstrap reps for CIs), --targets.
"""
from __future__ import annotations

import argparse

from . import heads
from .common import TARGETS, WINDOWS_H, load_cohort
from .trainer import IdentityEncoder, fit_eval


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modality", choices=["structured", "text", "echo", "echo-manifest"])
    ap.add_argument("--window", type=int, choices=WINDOWS_H, default=6)
    ap.add_argument("--admitted-only", action="store_true")
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--no-cache", action="store_true")
    # structured
    ap.add_argument("--blocks", nargs="+", default=["triage", "vitals", "cc"],
                    choices=["triage", "vitals", "cc", "medrecon"])
    # text
    ap.add_argument("--source", choices=["cc", "radiology"], default="radiology")
    ap.add_argument("--encoder", choices=["tfidf", "biowordvec"], default="tfidf")
    ap.add_argument("--vectors", help="path to BioWordVec_PubMed_MIMICIII_d200.vec.bin")
    ap.add_argument("--lookback-h", type=float, default=0.0,
                    help="also allow notes/echo up to this many hours BEFORE ED arrival (default 0)")
    args = ap.parse_args(argv)

    cohort = load_cohort(args.window, admitted_only=args.admitted_only)
    suffix = "_admitted" if args.admitted_only else ""
    common = dict(cohort=cohort, window_h=args.window, targets=args.targets, eval_test=args.eval_test,
                  n_boot=args.n_boot)

    if args.modality == "structured":
        from .features import triage
        X = triage.build(cohort, args.window, blocks=tuple(args.blocks), cache=not args.no_cache)
        head = heads.gbm()
        fit_eval(X, encoder=IdentityEncoder(), head=head, run_name=f"structured-{'-'.join(args.blocks)}_{head.name}{suffix}",
                 meta={"blocks": args.blocks, "feature_names": list(X.columns)}, **common)

    elif args.modality == "text":
        from .encoders.text import BioWordVecEncoder, TfidfEncoder
        from .features import notes
        texts = notes.build(cohort, args.window, source=args.source, cache=not args.no_cache,
                            lookback_h=args.lookback_h)
        if args.encoder == "tfidf":
            # chief complaints are a few words: smaller vocabulary threshold
            enc = TfidfEncoder(min_df=3 if args.source == "cc" else 5)
            head = heads.logistic(sparse=True)
        else:
            if not args.vectors:
                ap.error("--encoder biowordvec needs --vectors")
            enc = BioWordVecEncoder(args.vectors, cache_tag=f"{args.source}_{args.window}h")
            head = heads.logistic(sparse=False)
        fit_eval(texts, encoder=enc, head=head, run_name=f"text-{args.source}_{args.encoder}_{head.name}{suffix}",
                 meta={"source": args.source, "lookback_h": args.lookback_h}, **common)

    elif args.modality == "echo-manifest":
        from .features import echo
        echo.manifest(cohort, args.window, lookback_h=args.lookback_h)

    elif args.modality == "echo":
        from .features import echo
        X = echo.build(cohort, args.window, lookback_h=args.lookback_h)
        fit_eval(X, encoder=IdentityEncoder(), head=heads.logistic(sparse=False),
                 run_name=f"echo-echoclip_logreg{suffix}", meta={"lookback_h": args.lookback_h}, **common)


if __name__ == "__main__":
    main()
