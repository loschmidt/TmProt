"""
Fit the Tm probability calibrator, validate it, and export the lookup-table artifact.

Consumes prediction CSVs produced by `src/scripts/predict_dataset.py` and writes a
versioned JSON artifact plus diagnostic figures.

The calibrator is fitted on one set and reported on another. Fitting and reporting on
the same rows makes the calibration error look better than it is, so the default flow
fits on the ProMelt validation split and quotes quality only from the held-out test
split and from the independent evaluation sets.

Example::

    python src/scripts/run_calibration.py \
        --fit_csv models/esm2_lora/predictions/promelt/val_promelt_seq.csv \
        --report_csv models/esm2_lora/predictions/promelt/test_promelt_seq.csv \
        --independent_dir models/esm2_lora \
        --outdir models/esm2_lora/calibration --img_dir images \
        --thresholds 50 55 60 65 70 75 --refit_pooled --version v1
"""
import argparse
import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.eval.calibration import (
    build_grid,
    calibration_report,
    compare_calibrators,
    cumulative_grid,
    cumulative_precision,
    fit_kernel_conditional,
    fit_location_scale,
    fit_per_cutoff_logistic,
    interp_local,
    local_empirical_bins,
    local_support,
    logistic_prob,
    per_set_summary,
    plot_calibration_definition,
    plot_calibration_table,
    plot_ece_vs_threshold,
    plot_local_curves,
    plot_prediction_density,
    plot_local_vs_cumulative,
    plot_reliability,
    plot_residual_diagnostics,
    residual_diagnostics,
    save_artifact,
    write_summary_table,
)

INDEPENDENT_SETS = ["brenda", "fireprot", "cas", "hld", "ered_wt", "ered_asr"]

PRED_GRID = {"start": 25.0, "stop": 110.0, "step": 0.5}
THRESHOLD_GRID = {"start": 40.0, "stop": 100.0, "step": 1.0}
CUTOFF_GRID = {"start": 30.0, "stop": 100.0, "step": 2.5}


def load_predictions(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a prediction CSV, returning (actual, predicted)."""
    df = pd.read_csv(path)
    return df["Tm_Actual"].to_numpy(dtype=float), df["Tm_Predicted"].to_numpy(dtype=float)


def load_independent(directory: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read every available independent evaluation-set prediction CSV."""
    out = {}
    for name in INDEPENDENT_SETS:
        path = os.path.join(directory, f"{name}.csv")
        if os.path.exists(path):
            out[name] = load_predictions(path)
        else:
            print(f"[calibration] missing independent set: {path}", file=sys.stderr)
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def sha256_of(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "unknown"


def build_caveats(
    fit_meta: dict,
    test_rate: float,
    indep_rate: float,
    per_set: Dict[str, Tuple[np.ndarray, np.ndarray]],
    threshold: float,
) -> List[str]:
    """Assemble the caveat list with the actual measured numbers baked in."""
    empty = [n for n, (a, _) in per_set.items() if (a >= threshold).sum() == 0]
    sizes = ", ".join(f"{n} {len(a)}" for n, (a, _) in per_set.items())
    max_indep = max((float(a.max()) for a, _ in per_set.values()), default=float("nan"))
    return [
        f"Calibrator fitted on {fit_meta['name']} (n={fit_meta['n']}); quality is quoted "
        f"only from held-out sets.",
        f"Base-rate shift at T={threshold:g}C: fit set {fit_meta['base_rate']:.3f}, "
        f"ProMelt test {test_rate:.3f}, independent pool {indep_rate:.3f}. A calibrator "
        f"fitted on ProMelt will under-predict on sets with a higher fraction of "
        f"thermostable proteins.",
        f"The independent pool is NOT a random sample: {sizes}. Four of the six sets are "
        f"mutation series around a handful of scaffolds, so the effective sample size is "
        f"well below the nominal row count and per-bin counts are small.",
        f"Sets with zero proteins above T={threshold:g}C: "
        f"{', '.join(empty) if empty else 'none'}. Calibration metrics there are "
        f"undefined and reported as null, not zero.",
        f"Independent-set actual Tm reaches {max_indep:.1f}C, beyond the ProMelt fitting "
        f"range; probabilities in that region are extrapolation.",
        "ProMelt is dominated by Meltome/TPP cell-lysate measurements whereas the "
        "independent sets are purified-protein assays. See diagnostics.transport for "
        "whether the residual distribution actually holds across this shift.",
        "Sequences are truncated to 512 tokens at inference, so calibration is optimistic "
        "for proteins longer than that.",
        "The cumulative table is the hit rate for screening a library distributed like the "
        "named set. It is a property of that library as much as of the model, and it "
        "overstates the odds for any individual protein near the low edge of the selection.",
    ]


def evaluate_set(
    calib, y_true: np.ndarray, y_pred: np.ndarray, thresholds: List[float]
) -> Dict[float, dict]:
    """Calibration report for one set at every threshold."""
    out = {}
    for t in thresholds:
        labels = (y_true >= t).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            out[t] = {"n": int(len(labels)), "n_pos": int(labels.sum()),
                      "base_rate": float(labels.mean()), "undefined": True,
                      "mean_pred": float(calib.local_prob(y_pred, t).mean()),
                      "ece": float("nan"), "mce": float("nan"), "brier": float("nan"),
                      "brier_baserate": float("nan"), "log_loss": float("nan"),
                      "cal_in_large": float("nan"), "frac_dropped": float("nan"),
                      "bins": []}
            continue
        rep = calibration_report(calib.local_prob(y_pred, t), labels)
        rep["undefined"] = False
        out[t] = rep
    return out


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)
    thresholds = [float(t) for t in args.thresholds]
    focus = args.focus_threshold

    fit_true, fit_pred = load_predictions(args.fit_csv)
    rep_true, rep_pred = load_predictions(args.report_csv)
    per_set = load_independent(args.independent_dir)
    ind_true = np.concatenate([a for a, _ in per_set.values()])
    ind_pred = np.concatenate([p for _, p in per_set.values()])

    print(f"[calibration] fit n={len(fit_true)}  report n={len(rep_true)}  "
          f"independent n={len(ind_true)} across {len(per_set)} sets")

    # ---- grids ----------------------------------------------------------------
    pred_grid = build_grid(**PRED_GRID)
    thr_grid = build_grid(**THRESHOLD_GRID)
    cut_grid = build_grid(**CUTOFF_GRID)

    # ---- select the calibrator family ----------------------------------------
    # Selection is cross-validated on the fitting set alone, so the held-out set is
    # never consulted when choosing and its reported error stays honest.
    if args.method == "auto":
        cv = compare_calibrators(fit_true, fit_pred, pred_grid, thr_grid, thresholds)
        print("[calibration] calibrator selection (5-fold CV on the fit set):")
        print(cv.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        method = str(cv.loc[0, "method"])
    else:
        cv, method = None, args.method
    print(f"[calibration] using: {method}")

    if method == "kernel_conditional":
        calib = fit_kernel_conditional(fit_true, fit_pred, pred_grid, thr_grid)
        print(f"[calibration] bandwidth {calib.bandwidth:.2f}C "
              f"({calib.fit_meta['bandwidth_selected_by']}), "
              f"prediction support {calib.x_data_range[0]:.1f}-{calib.x_data_range[1]:.1f}C")
    else:
        calib = fit_location_scale(fit_true, fit_pred)
        print(f"[calibration] location m(x) = {calib.a:.3f} + {calib.b:.4f}*x   "
              f"(SE {calib.fit_meta['a_se']:.3f}, {calib.fit_meta['b_se']:.4f})")
        print(f"[calibration] log scale  = {calib.c:.4f} + {calib.d:.5f}*x   "
              f"-> s(50)={calib.scale(np.array([50.0]))[0]:.2f}C, "
              f"s(75)={calib.scale(np.array([75.0]))[0]:.2f}C")
        print(f"[calibration] residual t df = {calib.t_df:.2f}")

    local_table = np.round(calib.local_grid(pred_grid, thr_grid), 4)

    cumulative_sets = {
        "promelt_test": {
            "path": args.report_csv, "n": int(len(rep_true)),
            **cumulative_grid(rep_true, rep_pred, cut_grid, thr_grid),
        },
        "independent_combined": {
            "paths": sorted(per_set.keys()), "n": int(len(ind_true)),
            **cumulative_grid(ind_true, ind_pred, cut_grid, thr_grid),
        },
    }

    # ---- validation -----------------------------------------------------------
    validation = {
        "promelt_test": evaluate_set(calib, rep_true, rep_pred, thresholds),
        "independent_combined": evaluate_set(calib, ind_true, ind_pred, thresholds),
        "per_dataset": {
            n: evaluate_set(calib, a, p, thresholds) for n, (a, p) in per_set.items()
        },
    }

    # Residual diagnostics describe the location-scale assumptions, so they are only
    # meaningful when that family was selected. The kernel calibrator makes no such
    # assumptions; a reference location-scale fit is still recorded so the assumption
    # violations that motivated the choice stay visible in the artifact.
    ref = calib if method == "location_scale" else fit_location_scale(fit_true, fit_pred)
    diagnostics = {
        "applies_to": "location_scale reference fit" if method != "location_scale" else method,
        "fit_promelt_val": residual_diagnostics(fit_true, fit_pred, ref),
        "transport": {
            "promelt_test": residual_diagnostics(rep_true, rep_pred, ref)["shape"],
            "independent_combined": residual_diagnostics(ind_true, ind_pred, ref)["shape"],
        },
    }
    if cv is not None:
        diagnostics["calibrator_selection_cv"] = cv.to_dict(orient="records")

    logistic = fit_per_cutoff_logistic(fit_true, fit_pred, thresholds)
    cross = np.linspace(40, 80, 81)
    max_diff = []
    for _, row in logistic.iterrows():
        lp = logistic_prob(row, cross)
        if np.all(np.isnan(lp)):
            max_diff.append(float("nan"))
        else:
            max_diff.append(float(np.nanmax(np.abs(lp - calib.local_prob(cross, row["threshold"])))))
    logistic["max_abs_diff_vs_primary"] = max_diff

    fit_meta = {
        "name": "promelt_val", "path": args.fit_csv, "n": int(len(fit_true)),
        "base_rate": float((fit_true >= focus).mean()),
        "tm_true_range": [float(fit_true.min()), float(fit_true.max())],
        "tm_pred_range": [float(fit_pred.min()), float(fit_pred.max())],
        "base_rate_at": {str(t): float((fit_true >= t).mean()) for t in thresholds},
    }
    caveats = build_caveats(
        fit_meta, float((rep_true >= focus).mean()), float((ind_true >= focus).mean()),
        per_set, focus,
    )

    artifact = {
        "schema_version": "1.0",
        "artifact_id": f"esm2lora_promelt_val_{args.version}",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "model": {
            "display_name": "ESM2-LoRA",
            "base_model": "facebook/esm2_t33_650M_UR50D",
            "adapter_path": args.model_dir,
            "adapter_sha256": sha256_of(os.path.join(args.model_dir, "adapter_model.safetensors")),
            "max_length": 512,
            "truncation": True,
        },
        "fit_set": fit_meta,
        "method": {"selected": method, "selected_by": "5-fold CV log loss on the fit set"
                   if cv is not None else "caller", **calib.to_dict()},
        "grids": {
            "tm_pred_grid": PRED_GRID,
            "threshold_grid": THRESHOLD_GRID,
            "cutoff_grid": CUTOFF_GRID,
        },
        "local": {
            "description": "P(Tm_true >= T | Tm_pred = x). Row-major p[i][j]; i indexes "
                           "threshold_grid, j indexes tm_pred_grid.",
            "p": local_table.tolist(),
        },
        "cumulative": {
            "description": "P(Tm_true >= T | Tm_pred >= x), empirical, per evaluation set. "
                           "Row-major p[i][j]; i indexes threshold_grid, j indexes "
                           "cutoff_grid. Set-dependent: describes screening a library "
                           "distributed like that set.",
            "sets": cumulative_sets,
        },
        "validation": validation,
        "diagnostics": diagnostics,
        "comparison": {"per_cutoff_logistic": logistic.to_dict(orient="records")},
        "interpolation": {
            "method": "bilinear",
            "out_of_range": "clamp_to_edge",
            "clip": [0.0, 1.0],
            "note": "Consumers without scipy should bilinearly interpolate local.p over "
                    "(threshold_grid, tm_pred_grid); see calibration.interp_local.",
        },
        "caveats": caveats,
    }
    artifact_path = os.path.join(
        args.outdir, f"calibration_esm2lora_promelt_val_{args.version}.json"
    )
    save_artifact(artifact, artifact_path)

    # Per-set summary: the ranking and calibration numbers side by side, because a set
    # can post a clean ECE purely by having nothing to get wrong.
    summary = per_set_summary(per_set, validation["per_dataset"], focus)
    write_summary_table(
        summary,
        os.path.join(args.outdir, f"per_set_summary_T{focus:g}_{args.version}.csv"),
        os.path.join(args.outdir, f"per_set_summary_T{focus:g}_{args.version}.md"),
    )
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- plots ----------------------------------------------------------------
    shift_note = (f"Base rate at T={focus:g}C -- fit(val) {fit_meta['base_rate']:.3f}, "
                  f"ProMelt test {(rep_true >= focus).mean():.3f}, "
                  f"independent pool {(ind_true >= focus).mean():.3f}.")
    indep_note = (f"Independent pool n={len(ind_true)} from {len(per_set)} heterogeneous "
                  f"assays; not a random sample, per-bin counts are small.")

    ref_note = ("" if method == "location_scale" else
                f" Reference location-scale fit shown for diagnosis; the deployed "
                f"calibrator is {method}, which makes no location-scale assumption.")
    plot_residual_diagnostics(
        fit_true, fit_pred, ref, os.path.join(args.img_dir, "calibration_residual_diagnostics.png"),
        footnotes=[f"Fitted on ProMelt val (n={len(fit_true)}). "
                   f"Location m(x)={ref.a:.2f}+{ref.b:.3f}x; log s(x)={ref.c:.3f}+{ref.d:.4f}x."
                   + ref_note],
    )
    plot_reliability(
        {"ProMelt test": validation["promelt_test"][focus],
         "Independent combined": validation["independent_combined"][focus]},
        os.path.join(args.img_dir, f"calibration_reliability_T{focus:g}.png"), focus,
        footnotes=[shift_note, indep_note],
    )
    # Support is the count of fit proteins backing each grid location; the curves are
    # only readable next to it, since the calibrator returns a number everywhere.
    bandwidth = getattr(calib, "bandwidth", 0.75)
    support = local_support(fit_pred, pred_grid, bandwidth)

    plot_calibration_table(
        np.asarray(artifact["local"]["p"], dtype=float), pred_grid, thr_grid,
        support=support, thresholds=thresholds, min_support=args.min_support,
        path_save=os.path.join(args.img_dir, "calibration_table_heatmap.png"),
        footnotes=[f"The deployed artifact in full: {len(thr_grid)} thresholds x "
                   f"{len(pred_grid)} predicted values, fitted on ProMelt val "
                   f"(n={len(fit_pred)})."],
    )
    plot_prediction_density(
        fit_true, fit_pred, pred_grid, bandwidth, thresholds,
        path_save=os.path.join(args.img_dir, "calibration_prediction_density.png"),
        min_support=args.min_support,
        footnotes=[f"ProMelt val (n={len(fit_pred)}), the set the calibrator is fitted on."],
    )
    plot_local_curves(
        calib, [t for t in thresholds if 55 <= t <= 70], pred_grid,
        empirical=local_empirical_bins(rep_true, rep_pred, focus),
        focus_threshold=focus,
        path_save=os.path.join(args.img_dir, "calibration_local_curves.png"),
        footnotes=[f"Curves from the val-fitted calibrator. Points are observed "
                   f"frequencies on ProMelt test (n={len(rep_true)}), Wilson 95% bars.",
                   "Markers on the left panel mark where each curve crosses P=0.5."],
        support=support, min_support=args.min_support,
    )
    plot_local_vs_cumulative(
        calib, {"ProMelt test": (rep_true, rep_pred),
                "Independent combined": (ind_true, ind_pred)},
        focus, cut_grid,
        os.path.join(args.img_dir, f"calibration_local_vs_cumulative_T{focus:g}.png"),
        footnotes=["Cumulative sits above local by construction: it averages the local "
                   "curve over the whole upper tail.", shift_note],
    )
    if args.explainer:
        # Methods figure, not a diagnostic: unpacks one cell of the table so the
        # artifact can be explained without a formula.
        plot_calibration_definition(
            fit_true, fit_pred, args.explainer_pred, focus,
            path_save=os.path.join(args.img_dir, "calibration_definition.png"),
        )
    plot_ece_vs_threshold(
        {"ProMelt test": validation["promelt_test"],
         "Independent combined": validation["independent_combined"]},
        os.path.join(args.img_dir, "calibration_ece_vs_threshold.png"),
        footnotes=[indep_note],
    )

    # ---- optional pooled refit ------------------------------------------------
    if args.refit_pooled:
        pooled_true = np.concatenate([fit_true, rep_true])
        pooled_pred = np.concatenate([fit_pred, rep_pred])
        pooled = (fit_kernel_conditional(pooled_true, pooled_pred, pred_grid, thr_grid)
                  if method == "kernel_conditional"
                  else fit_location_scale(pooled_true, pooled_pred))
        pooled_artifact = dict(artifact)
        pooled_artifact.update({
            "artifact_id": f"esm2lora_promelt_valtest_{args.version}",
            "refit_variant": True,
            "fit_set": {"name": "promelt_val_plus_test", "n": int(len(pooled_true)),
                        "paths": [args.fit_csv, args.report_csv],
                        "base_rate_at": {str(t): float((pooled_true >= t).mean())
                                         for t in thresholds}},
            "method": pooled.to_dict(),
            "local": {"description": artifact["local"]["description"],
                      "p": np.round(pooled.local_grid(pred_grid, thr_grid), 4).tolist()},
            "validation": {"note": "No held-out set exists for this refit by construction. "
                                   "Quote quality metrics from the val-fitted artifact only."},
            "caveats": caveats + ["Refitted on val+test pooled; has no held-out set."],
        })
        save_artifact(pooled_artifact, os.path.join(
            args.outdir, f"calibration_esm2lora_promelt_valtest_{args.version}.json"))

    run_sanity_checks(calib, artifact, rep_true, rep_pred, ind_true, ind_pred,
                      cut_grid, focus, logistic, ref=ref, method=method)


def run_sanity_checks(calib, artifact, rep_true, rep_pred, ind_true, ind_pred,
                      cut_grid, focus, logistic, ref=None, method=None) -> None:
    """Print the checks that decide whether the artifact is trustworthy."""
    print("\n" + "=" * 72)
    print("SANITY CHECKS")
    print("=" * 72)

    table = np.asarray(artifact["local"]["p"], dtype=float)
    mono_x = bool(np.all(np.diff(table, axis=1) >= -1e-9))
    mono_t = bool(np.all(np.diff(table, axis=0) <= 1e-9))
    print(f"  monotone in predicted Tm      : {mono_x}")
    print(f"  monotone in threshold         : {mono_t}")

    cross = calib.crossing_point(focus)
    # a/b live on the location-scale family only; fall back to the reference fit that
    # main() keeps for exactly this purpose when the kernel calibrator was selected.
    ls = calib if method == "location_scale" else ref
    if ls is None:
        why = ""
    elif ls is calib:
        why = f"(expected near {focus:g} only if b~1, a~0; here a={ls.a:.2f}, b={ls.b:.3f})"
    else:
        why = f"(reference location-scale fit: a={ls.a:.2f}, b={ls.b:.3f})"
    print(f"  P=0.5 crossing at T={focus:g}          : {cross:.2f}C {why}")

    for label, (a, p) in [("ProMelt test", (rep_true, rep_pred)),
                          ("independent", (ind_true, ind_pred))]:
        cum = cumulative_precision(a, p, cut_grid, focus)
        loc = calib.local_prob(cut_grid, focus)
        ok = (cum["n_selected"] >= 30).to_numpy()
        viol = int((cum["rate"].to_numpy()[ok] < loc[ok] - 0.02).sum())
        print(f"  cumulative >= local, {label:<12s}: {viol} violations of {int(ok.sum())} cells (n>=30)")

    for name, key in [("ProMelt test", "promelt_test"), ("Independent", "independent_combined")]:
        rep = artifact["validation"][key][focus]
        print(f"  {name:<14s} @T={focus:g}       : ECE {rep['ece']:.4f}  Brier {rep['brier']:.4f} "
              f"(base-rate {rep['brier_baserate']:.4f})  cal-in-large {rep['cal_in_large']:+.4f} "
              f"[base rate {rep['base_rate']:.3f}, mean p {rep['mean_pred']:.3f}]")

    d = logistic["max_abs_diff_vs_primary"].to_numpy(dtype=float)
    print(f"  vs per-cutoff logistic        : max abs diff {np.nanmax(d):.4f} "
          f"over x in [40,80] (want < 0.05)")

    rng = np.random.default_rng(0)
    xs = rng.uniform(25, 110, 1000)
    ts = rng.uniform(40, 100, 1000)
    err = max(abs(interp_local(artifact, float(x), float(t)) - float(calib.local_prob(np.array([x]), float(t))[0]))
              for x, t in zip(xs, ts))
    print(f"  grid round-trip vs closed form: max abs diff {err:.5f} (want < 0.002)")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit_csv", required=True, help="Prediction CSV to fit the calibrator on")
    p.add_argument("--report_csv", required=True, help="Held-out prediction CSV to report on")
    p.add_argument("--independent_dir", default="models/esm2_lora",
                   help="Directory holding the independent evaluation-set prediction CSVs")
    p.add_argument("--model_dir", default="models/esm2_lora", help="Adapter directory, for provenance")
    p.add_argument("--outdir", default="models/esm2_lora/calibration")
    p.add_argument("--img_dir", default="images")
    p.add_argument("--method", choices=["auto", "kernel_conditional", "location_scale"],
                   default="auto",
                   help="Calibrator family. 'auto' cross-validates both on the fit set.")
    p.add_argument("--thresholds", nargs="+", type=float, default=[50, 55, 60, 65, 70, 75])
    p.add_argument("--focus_threshold", type=float, default=60.0,
                   help="Threshold used for the headline figures")
    p.add_argument("--refit_pooled", action="store_true",
                   help="Also emit a variant refitted on fit+report pooled")
    p.add_argument("--explainer", action="store_true",
                   help="Also write the methods figure unpacking one table cell")
    p.add_argument("--explainer_pred", type=float, default=45.0,
                   help="Predicted Tm the --explainer figure unpacks")
    p.add_argument("--min_support", type=int, default=30,
                   help="Fit proteins within one bandwidth below which a prediction "
                        "value is treated as unsupported and drawn faded")
    p.add_argument("--version", default="v1")
    return p.parse_args()


if __name__ == "__main__":
    main()
