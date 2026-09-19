### Run.py ###


# REMEMBER! Windows doesn't see cyrillic directly, if u have problems
# bc u are on WINDOWS, copy and paste 
# ""python -X utf8 run.py"" in powershell. (Long live to linux).


"""
HILNAR — full analysis of the Belarusian prototype corpus (2019 vs 2025).

    python run.py                 # observed indices + audits + figures
    python run.py --full          # adds bootstrap, permutation, spec grid

Outputs (./out/):
    results.json            observed indices, audits, inference
    spec_grid.csv           every index under the full specification cross
    fig_indices.png         NSI bars + TDI trajectory, with bootstrap CIs
    fig_robustness.png      specification curve for the TDI change
"""

import argparse
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

import hilnar_initial_def_for_run as H
from corpus import DOCUMENTS

# Matrices here are small (~10^3 x 10^3); beyond 4 threads the SVD is
# dominated by synchronisation and gets slower.  
# Fixed for reproducible timing.
torch.set_num_threads(min(4, os.cpu_count() or 1))

OUT = "out"
YEARS = (2019, 2025)


def _fmt(x, nd=3):
    return "  n/a" if x is None or (isinstance(x, float) and math.isnan(x)) \
        else f"{x:+.{nd}f}"


def observed(cfg):
    r = H.analyse(DOCUMENTS, cfg, YEARS)
    seg = r["preprocessing"]["segmentation"]
    print("=" * 74)
    print("HILNAR — Belarusian prototype corpus, 2019 vs 2025")
    print("=" * 74)
    print(f"documents            : {sum(r['docs_per_slice'].values())} "
          f"in {len(r['docs_per_slice'])} carrier-year slices "
          f"({r['docs_per_slice']})")
    print(f"exact / near dupes   : {r['preprocessing']['dedup']['exact_removed']}"
          f" / {r['preprocessing']['dedup']['near_removed']}"
          f"   (borderline pairs to inspect: "
          f"{len(r['preprocessing']['dedup']['borderline'])})")
    print(f"BPE merges learned   : {seg['n_merges_learned']} "
          f"-> {seg['n_types']} types, {seg['mean_pieces_per_word']} "
          f"pieces/word, continuation rate {seg['continuation_rate']:.3f}")
    print(f"analysis vocabulary  : {r['vocab']['size']} tokens "
          f"(min_count={cfg.min_count}, filter='{cfg.vocab_filter}') "
          f"{r['vocab']['composition']}")
    print(f"projection           : dim={cfg.dim}, basis='{cfg.basis}', "
          f"rotate={cfg.rotate}, standard={tuple(cfg.standard)}")

    print("-" * 74)
    print("(A) NSI — diachronic narrative shift 2019->2025   [0 none .. 1 inversion]")
    print(f"    {'carrier':<18}{'NSI':>8}{'shared':>8}{'eff.n':>8}")
    for c in H.CARRIERS:
        w = r["nsi"][c]
        print(f"    {H.CARRIER_NAMES[c]:<18}{w.value:>8.3f}"
              f"{w.n_shared:>8d}{w.effective_n:>8.1f}")
    print(f"    {'STATE (doc-weighted)':<18}{r['nsi_state']:>8.3f}")

    print("-" * 74)
    print("(B) SAI vs government + TDI trajectory            [-1 anti .. +1 mirror]")
    print(f"    {'sector':<18}{'2019':>9}{'2025':>9}{'change':>9}"
          f"{'shared19':>10}{'shared25':>10}")
    for s in H.NON_GOV:
        a, b = r["sai"][s][2019], r["sai"][s][2025]
        print(f"    {H.CARRIER_NAMES[s]:<18}{_fmt(a.value):>9}{_fmt(b.value):>9}"
              f"{_fmt(b.value - a.value):>9}{a.n_shared:>10d}{b.n_shared:>10d}")
    d = r["tdi"][2025] - r["tdi"][2019]
    print(f"    {'TDI composite':<18}{_fmt(r['tdi'][2019]):>9}"
          f"{_fmt(r['tdi'][2025]):>9}{_fmt(d):>9}")

    audit = H.weight_audit(r["_reduced"], r["_slices"], YEARS)
    print("-" * 74)
    print("(C) weight audit — share of each index carried by non-content tokens")
    for k, v in audit.items():
        print(f"    {k:<14} shared={v['n_shared']:>4d} content={v['n_content']:>4d}"
              f" non-content weight={v['non_content_weight_share']:.1%}"
              f" eff.n={v['effective_n']:.1f}")
    return r, audit


def figures(r, boot=None):
    os.makedirs(OUT, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    names = [H.CARRIER_NAMES[c] for c in H.CARRIERS] + ["STATE"]
    vals = [r["nsi"][c].value for c in H.CARRIERS] + [r["nsi_state"]]
    err = None
    if boot:
        keys = [f"nsi_{c}" for c in H.CARRIERS] + ["nsi_state"]
        lo = [max(0.0, vals[i] - boot[k]["lo95"]) for i, k in enumerate(keys)]
        hi = [max(0.0, boot[k]["hi95"] - vals[i]) for i, k in enumerate(keys)]
        err = [lo, hi]
    cols = ["#4878a8"] * 4 + ["#a83233"]
    ax1.bar(names, vals, color=cols, yerr=err, capsize=4, ecolor="0.3")
    ax1.set_ylabel("NSI  [0, 1]")
    ax1.set_title("(A) Narrative shift 2019$\\rightarrow$2025")
    ax1.tick_params(axis="x", rotation=20)
    ax1.set_ylim(0, max(1.0, max(vals) * 1.6))

    for s, col in zip(H.NON_GOV, ["#4878a8", "#3a9a5c", "#a83233"]):
        ys = [r["sai"][s][y].value for y in YEARS]
        ax2.plot(YEARS, ys, "o-", color=col, label=H.CARRIER_NAMES[s])
        if boot:
            for y in YEARS:
                b = boot[f"sai_{s}_{y}"]
                ax2.plot([y, y], [b["lo95"], b["hi95"]], color=col,
                         alpha=0.45, lw=6, solid_capstyle="butt")
    ax2.plot(YEARS, [r["tdi"][y] for y in YEARS], "s--", color="0.25",
             label="TDI composite")
    ax2.axhline(0, color="0.6", ls=":", lw=1)
    ax2.set_xticks(list(YEARS))
    ax2.set_ylabel("SAI  [-1, +1]")
    ax2.set_title("(B) Alignment with government (bars: 95% bootstrap CI)")
    ax2.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_indices.png", dpi=180)
    return fig


def robustness_figure(rows):
    """Specification curve: TDI change under every specification, sorted."""
    vals = sorted((row["tdi_delta"], row) for row in rows
                  if not math.isnan(row["tdi_delta"]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 5.6),
                                   gridspec_kw={"height_ratios": [2, 1.5]},
                                   sharex=True)
    xs = range(len(vals))
    ys = [v for v, _ in vals]
    cols = ["#a83232" if y < 0 else "#4878a8" for y in ys]
    ax1.bar(xs, ys, color=cols, width=1.0)
    ax1.axhline(0, color="0.2", lw=1)
    ax1.set_ylabel("TDI(2025) - TDI(2019)")
    ax1.set_title("Specification curve: sign of the alignment change is not "
                  "invariant (red = decreasing alignment)")

    marks = [("rotate=True", lambda r: r["rotate"]),
             ("basis=standard", lambda r: r["basis"] == "standard"),
             ("filter=content", lambda r: r["vocab_filter"] == "content"),
             ("dim>=30", lambda r: r["dim"] >= 30),
             ("min_count=3", lambda r: r["min_count"] == 3)]
    for k, (lab, pred) in enumerate(marks):
        pts = [i for i, (_, row) in enumerate(vals) if pred(row)]
        ax2.plot(pts, [k] * len(pts), "|", color="0.15", markersize=7)
    ax2.set_yticks(range(len(marks)))
    ax2.set_yticklabels([m[0] for m in marks], fontsize=8)
    ax2.set_ylim(-0.6, len(marks) - 0.4)
    ax2.set_xlabel(f"specification, ordered by TDI change (n={len(vals)})")
    ax2.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_robustness.png", dpi=180)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="run bootstrap, permutation tests and spec grid in-process")
    ap.add_argument("--collect", action="store_true",
                    help="assemble out/{boot,perm,grid}.json written by stage.py "
                         "instead of recomputing them (the reproducible route: "
                         "each stage is checkpointed, so a long inference run "
                         "can be resumed rather than restarted)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cfg = H.Config()
    r, audit = observed(cfg)

    out = {
        "config": r["config"],
        "preprocessing": r["preprocessing"],
        "vocab": r["vocab"],
        "docs_per_slice": r["docs_per_slice"],
        "nsi": {c: {"value": w.value, "n_shared": w.n_shared,
                    "effective_n": round(w.effective_n, 2)}
                for c, w in r["nsi"].items()},
        "nsi_state": r["nsi_state"],
        "sai": {f"{s}-{y}": {"value": r["sai"][s][y].value,
                             "n_shared": r["sai"][s][y].n_shared,
                             "effective_n": round(r["sai"][s][y].effective_n, 2)}
                for s in H.NON_GOV for y in YEARS},
        "tdi": {str(y): r["tdi"][y] for y in YEARS},
        "tdi_delta": r["tdi"][YEARS[1]] - r["tdi"][YEARS[0]],
        "weight_audit": audit,
        "top_movers": {c: H.top_movers(r["_reduced"], r["_slices"], c, YEARS, 12)
                       for c in H.CARRIERS},
        "clusters": {c: H.cluster_shifts(r["_reduced"], r["_slices"], c,
                                         *YEARS, n_clusters=5)
                     for c in H.CARRIERS},
        "sources": [{"carrier": d["carrier"], "year": d["year"],
                     "source": d["source"], "note": d["note"]}
                    for d in DOCUMENTS],
    }

    boot = None
    if args.full:
        print("-" * 74)
        print(f"(D) document bootstrap, B={args.n_boot}")
        boot = H.bootstrap(DOCUMENTS, cfg, YEARS, args.n_boot)
        for k in ["nsi_g", "nsi_m", "nsi_e", "nsi_p", "nsi_state",
                  "tdi_2019", "tdi_2025", "tdi_delta"]:
            b = boot[k]
            print(f"    {k:<12} median={_fmt(b['median'])} "
                  f"95% CI [{_fmt(b['lo95'])}, {_fmt(b['hi95'])}]")
        out["bootstrap"] = boot

        print("-" * 74)
        print(f"(E) permutation tests, B={args.n_perm}")
        pn = H.permutation_nsi(DOCUMENTS, cfg, YEARS, args.n_perm)
        for c, v in pn.items():
            print(f"    NSI {c:<7} obs={_fmt(v['observed'])} "
                  f"null={_fmt(v['null_mean'])}+-{v['null_sd']:.3f} "
                  f"p={v['p_one_sided']:.3f}")
        ps = {y: H.permutation_sai(DOCUMENTS, y, cfg, args.n_perm) for y in YEARS}
        for y in YEARS:
            for s, v in ps[y].items():
                print(f"    SAI {s}-{y} obs={_fmt(v['observed'])} "
                      f"null={_fmt(v['null_mean'])}+-{v['null_sd']:.3f} "
                      f"p={v['p_two_sided']:.3f}")
        out["permutation_nsi"] = pn
        out["permutation_sai"] = {str(y): ps[y] for y in YEARS}

        print("-" * 74)
        print("(F) specification grid")
        rows = H.specification_grid(DOCUMENTS, YEARS)
        neg = sum(1 for x in rows if x["tdi_delta"] < 0)
        print(f"    {len(rows)} specifications | TDI change negative in "
              f"{neg} ({neg / len(rows):.0%}), positive in {len(rows) - neg}")
        for pred, lab in [(lambda x: x["rotate"], "rotate=True"),
                          (lambda x: not x["rotate"], "rotate=False")]:
            sub = [x["tdi_delta"] for x in rows if pred(x)]
            print(f"      {lab:<14} n={len(sub):>3} median TDI change="
                  f"{_fmt(sorted(sub)[len(sub) // 2])} "
                  f"range [{_fmt(min(sub))}, {_fmt(max(sub))}]")
        with open(f"{OUT}/spec_grid.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        out["spec_grid_summary"] = {
            "n": len(rows), "n_negative_tdi_delta": neg,
            "share_negative": neg / len(rows)}
        robustness_figure(rows)

    if args.collect:
        missing = [s for s in ("boot", "perm", "grid")
                   if not os.path.exists(f"{OUT}/{s}.json")]
        if missing:
            raise SystemExit(
                f"missing checkpoints: {missing}. Run `python stage.py "
                f"{missing[0]}` (and the others) first.")
        for s in ("boot", "perm", "grid"):
            with open(f"{OUT}/{s}.json") as f:
                out.update(json.load(f))
        boot = out["bootstrap"]
        rows = out["spec_grid"]
        neg = sum(1 for x in rows if x["tdi_delta"] < 0)
        print("-" * 74)
        print(f"(D-F) collected checkpoints: bootstrap B={out['n_boot']}, "
              f"permutation B={out['n_perm']}, {len(rows)} specifications")
        print(f"    TDI change negative in {neg}/{len(rows)} ({neg/len(rows):.0%}); "
              f"bootstrap 95% CI [{_fmt(boot['tdi_delta']['lo95'])}, "
              f"{_fmt(boot['tdi_delta']['hi95'])}] "
              f"{'contains' if boot['tdi_delta']['lo95'] < 0 < boot['tdi_delta']['hi95'] else 'excludes'} zero")
        with open(f"{OUT}/spec_grid.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        out["spec_grid_summary"] = {
            "n": len(rows), "n_negative_tdi_delta": neg,
            "share_negative": neg / len(rows)}
        robustness_figure(rows)

    figures(r, boot)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print("=" * 74)
    print(f"wrote {OUT}/results.json, {OUT}/fig_indices.png"
          + (f", {OUT}/spec_grid.csv, {OUT}/fig_robustness.png"
             if args.full else ""))


if __name__ == "__main__":
    main()
