"""CLI for the GRACE single-modality baselines.

Run from project/grace with the torch env:
    PY=~/miniconda3/envs/torch/bin/python
    $PY -m grace_models.train structured --window 6                         # XGBoost on triage+vitals+CC tokens
    $PY -m grace_models.train structured --window 6 --blocks triage vitals cc medrecon
    $PY -m grace_models.train labs --window 6                               # XGBoost on blood labs resulted by t_pred
    $PY -m grace_models.train structured-labs --window 6                    # XGBoost on structured blocks + labs
    $PY -m grace_models.train ecg --window 6                                # XGBoost on last-ECG intervals/axes
    $PY -m grace_models.train ecg --window 6 --ecg-interp                   # ... + machine interpretation statements
    $PY -m grace_models.train structured-labs-ecg --window 6                # XGBoost on structured + labs + ECG
    $PY -m grace_models.train text --window 6 --source cc                   # TF-IDF + LR on chief complaint
    $PY -m grace_models.train text --window 24 --source radiology           # TF-IDF + LR on radiology reports
    $PY -m grace_models.train text --window 24 --source radiology --encoder biowordvec --vectors /path/BioWordVec.bin
    $PY -m grace_models.train echo-manifest --window 24                     # list echo DICOMs to download
    $PY -m grace_models.train echo --window 24                              # LR on frozen EchoCLIP embeddings

Common flags: --admitted-only (train AND evaluate on admitted stays), --eval-test (score the test split;
leave off during development), --n-boot (bootstrap reps for CIs), --targets.
Labs flags: --lookback-h (also allow labs resulted up to this many hours BEFORE ED arrival; default 0),
--no-lab-deltas (drop the troponin last-minus-first features).
ECG flags: --ecg-interp (add the ECG machine's interpretation statements; report with and without).
"""
from __future__ import annotations

import argparse

from . import heads
from .common import TARGETS, WINDOWS_H, load_cohort
from .trainer import IdentityEncoder, fit_eval


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modality", choices=["structured", "labs", "structured-labs", "ecg",
                                         "structured-labs-ecg", "text", "echo", "echo-manifest"])
    ap.add_argument("--window", type=int, choices=WINDOWS_H, default=6)
    ap.add_argument("--admitted-only", action="store_true")
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--no-cache", action="store_true")
    # structured
    ap.add_argument("--blocks", nargs="+", default=["triage", "vitals", "cc"],
                    choices=["triage", "vitals", "cc", "medrecon"])
    # labs
    ap.add_argument("--no-lab-deltas", action="store_true",
                    help="labs: drop troponin last-minus-first features")
    # ecg
    ap.add_argument("--ecg-interp", action="store_true",
                    help="ecg: add one-hot machine interpretation statements")
    # text
    ap.add_argument("--source", choices=["cc", "radiology"], default="radiology")
    ap.add_argument("--encoder", choices=["tfidf", "biowordvec"], default="tfidf")
    ap.add_argument("--vectors", help="path to BioWordVec_PubMed_MIMICIII_d200.vec.bin")
    ap.add_argument("--lookback-h", type=float, default=0.0,
                    help="also allow notes/echo/labs up to this many hours BEFORE ED arrival (default 0)")
    args = ap.parse_args(argv)

    cohort = load_cohort(args.window, admitted_only=args.admitted_only)
    suffix = "_admitted" if args.admitted_only else ""
    lb = f"_lb{args.lookback_h:g}" if args.lookback_h else ""
    nd = "_nodelta" if args.no_lab_deltas else ""
    ei = "-interp" if args.ecg_interp else ""
    common = dict(cohort=cohort, window_h=args.window, targets=args.targets, eval_test=args.eval_test,
                  n_boot=args.n_boot)

    if args.modality == "structured":
        from .features import triage
        X = triage.build(cohort, args.window, blocks=tuple(args.blocks), cache=not args.no_cache)
        head = heads.gbm()
        fit_eval(X, encoder=IdentityEncoder(), head=head, run_name=f"structured-{'-'.join(args.blocks)}_{head.name}{suffix}",
                 meta={"blocks": args.blocks, "feature_names": list(X.columns)}, **common)

    elif args.modality == "labs":
        from .features import labs
        X = labs.build(cohort, args.window, cache=not args.no_cache, lookback_h=args.lookback_h,
                       deltas=not args.no_lab_deltas)
        head = heads.gbm()
        fit_eval(X, encoder=IdentityEncoder(), head=head, run_name=f"labs{lb}{nd}_{head.name}{suffix}",
                 meta={"lookback_h": args.lookback_h, "deltas": not args.no_lab_deltas,
                       "itemids": labs.resolve_itemids(verbose=False), "feature_names": list(X.columns)},
                 **common)

    elif args.modality == "structured-labs":
        from .features import labs, triage
        X_s = triage.build(cohort, args.window, blocks=tuple(args.blocks), cache=not args.no_cache)
        X_l = labs.build(cohort, args.window, cache=not args.no_cache, lookback_h=args.lookback_h,
                         deltas=not args.no_lab_deltas)
        X = X_s.join(X_l, how="left")  # keep every structured stay; labs NaN where none resulted
        head = heads.gbm()
        fit_eval(X, encoder=IdentityEncoder(), head=head,
                 run_name=f"structured-{'-'.join(args.blocks)}-labs{lb}{nd}_{head.name}{suffix}",
                 meta={"blocks": args.blocks, "lookback_h": args.lookback_h, "deltas": not args.no_lab_deltas,
                       "feature_names": list(X.columns)}, **common)

    elif args.modality == "ecg":
        from .features import ecg
        X = ecg.build(cohort, args.window, cache=not args.no_cache, lookback_h=args.lookback_h,
                      interp=args.ecg_interp)
        head = heads.gbm()
        fit_eval(X, encoder=IdentityEncoder(), head=head, run_name=f"ecg{ei}{lb}_{head.name}{suffix}",
                 meta={"lookback_h": args.lookback_h, "interp": args.ecg_interp,
                       "feature_names": list(X.columns)}, **common)

    elif args.modality == "structured-labs-ecg":
        from .features import ecg, labs, triage
        X_s = triage.build(cohort, args.window, blocks=tuple(args.blocks), cache=not args.no_cache)
        X_l = labs.build(cohort, args.window, cache=not args.no_cache, lookback_h=args.lookback_h,
                         deltas=not args.no_lab_deltas)
        X_e = ecg.build(cohort, args.window, cache=not args.no_cache, lookback_h=args.lookback_h,
                        interp=args.ecg_interp)
        X = X_s.join(X_l, how="left").join(X_e, how="left")  # every structured stay; NaN where missing
        head = heads.gbm()
        fit_eval(X, encoder=IdentityEncoder(), head=head,
                 run_name=f"structured-{'-'.join(args.blocks)}-labs{nd}-ecg{ei}{lb}_{head.name}{suffix}",
                 meta={"blocks": args.blocks, "lookback_h": args.lookback_h, "deltas": not args.no_lab_deltas,
                       "interp": args.ecg_interp, "feature_names": list(X.columns)}, **common)

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
