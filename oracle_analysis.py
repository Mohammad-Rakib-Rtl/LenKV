"""
Turn budget_sweep.py output into the one number your proposal needs:
how much is there to gain by choosing the KV budget PER REQUEST instead of
using one fixed budget for everything?

  python oracle_analysis.py results_sweep/qasper_*.csv

Prints, and writes table_headroom.csv + fig_headroom.png:

  fixed@B      : score when every request uses budget B
  oracle@avgB  : score when each request gets its own budget, but the AVERAGE
                 budget across requests equals B (so memory/throughput match)
  headroom     : oracle - best fixed  <-- this is what the thesis tries to capture
  probe@avgB   : score of a cheap predictor trained on the prompt embedding,
                 i.e. how much of the headroom is reachable in practice
"""
import csv, os, sys
from collections import defaultdict

import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:                      # table still prints without figures
    HAVE_PLT = False
    print("matplotlib not installed — printing the table only "
          "(pip install matplotlib for the figures)")


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    tasks = sorted({r["task"] for r in rows})
    data = {}                      # task -> (idx, budget) -> (score, out_tokens)
    for t in tasks:
        d = {}
        for r in rows:
            if r["task"] == t:
                d[(int(r["idx"]), int(r["budget"]))] = (float(r["score"]),
                                                        int(r["out_tokens"]))
        data[t] = d
    return data


# def oracle_alloc(scores, budgets, avg_budget):
#     """Greedy allocation: start everyone at the smallest budget, then spend the
#     remaining budget where it buys the most score. Classic knapsack greedy —
#     good enough and easy to explain in a defense."""
#     n = len(scores)
#     total = avg_budget * n
#     alloc = {i: budgets[0] for i in range(n)}
#     spent = budgets[0] * n
#     nxt = {a: b for a, b in zip(budgets, budgets[1:])}   # next budget up

#     # Repeated passes: each pass applies the single best affordable upgrade.
#     # A single sorted sweep is wrong here, because an upgrade 256->512 can be
#     # cheaper per point than 128->256 and would then be skipped for good.
#     while True:
#         best = None                     # (gain per token, gain, cost, sample)
#         for i in range(n):
#             a = alloc[i]
#             b = nxt.get(a)
#             if b is None:
#                 continue
#             cost = b - a
#             gain = scores[i][b] - scores[i][a]
#             if gain > 0 and spent + cost <= total:
#                 key = gain / cost
#                 if best is None or key > best[0]:
#                     best = (key, gain, cost, i)
#         if best is None:
#             break
#         _, _, cost, i = best
#         alloc[i] = nxt[alloc[i]]
#         spent += cost
#     return alloc, sum(scores[i][alloc[i]] for i in range(n)) / n

def oracle_alloc(scores, budgets, avg_budget):
    """Greedy allocation: start everyone at the smallest budget, then spend the
    remaining budget where it buys the most score. Checks ALL possible upgrades."""
    n = len(scores)
    total = avg_budget * n
    alloc = {i: budgets[0] for i in range(n)}
    spent = budgets[0] * n

    while True:
        best = None
        for i in range(n):
            a = alloc[i]
            # Check ALL possible larger budgets, not just the next one
            for b in budgets:
                if b <= a: 
                    continue
                cost = b - a
                gain = scores[i][b] - scores[i][a]
                if gain > 0 and spent + cost <= total:
                    key = gain / cost
                    if best is None or key > best[0]:
                        best = (key, gain, cost, i, b)
        if best is None:
            break
        _, _, cost, i, b = best
        spent += (b - alloc[i]) # Pay the cost difference
        alloc[i] = b            # Apply the new highest budget
        
    return alloc, sum(scores[i][alloc[i]] for i in range(n)) / n

def ridge_kfold(X, y, alpha=10.0, k=5):
    """Cross-validated ridge regression, closed form, no sklearn needed.
    Returns out-of-fold predictions, so nothing is scored on data it saw."""
    n = len(y)
    idx = np.arange(n)
    folds = np.array_split(idx, min(k, n))
    yhat = np.zeros(n)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        if len(tr) < 2:
            continue
        Xt = np.c_[X[tr], np.ones(len(tr))]
        A = Xt.T @ Xt + alpha * np.eye(Xt.shape[1])
        w = np.linalg.solve(A, Xt.T @ y[tr])
        yhat[f] = np.c_[X[f], np.ones(len(f))] @ w
    return yhat


def probe_alloc(scores, budgets, avg_budget, feats):
    """Can a cheap model predict the right budget from the prompt alone?
    Cross-validated ridge regression on the prompt representation predicts the
    score at each budget; the same greedy allocation then runs on those
    PREDICTED scores and is evaluated with the TRUE ones."""
    n = len(scores)
    pred = {i: {} for i in range(n)}
    for b in budgets:
        y = np.array([scores[i][b] for i in range(n)], dtype=float)
        yhat = ridge_kfold(feats, y)
        for i in range(n):
            pred[i][b] = float(yhat[i])
    alloc, _ = oracle_alloc(pred, budgets, avg_budget)
    true = sum(scores[i][alloc[i]] for i in range(n)) / n
    return alloc, true


def main(paths):
    data = load(paths)
    out_dir = os.path.dirname(paths[0]) or "."
    table = []
    for task, d in data.items():
        idxs = sorted({i for i, _ in d})
        budgets = sorted({b for _, b in d if b > 0})
        idxs = [i for i in idxs if all((i, b) in d for b in budgets + [0])]
        scores = {k: {b: d[(i, b)][0] for b in budgets} for k, i in enumerate(idxs)}
        full = np.mean([d[(i, 0)][0] for i in idxs])
        lens = {b: np.mean([d[(i, b)][1] for i in idxs]) for b in budgets + [0]}

        feats_path = os.path.join(out_dir, f"{task}_feats.npy")
        feats = np.load(feats_path)[idxs] if os.path.exists(feats_path) else None
        if feats is not None:
            feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

        print(f"\n=== {task}   n={len(idxs)}   full-KV score={full:.3f}   "
              f"full-KV mean output={lens[0]:.0f} tokens")
        print(f"{'avg budget':>11} {'fixed':>8} {'oracle':>8} {'headroom':>9} "
              f"{'probe':>8} {'captured':>9} {'len ratio':>10}")
        for b in budgets:
            fixed = np.mean([scores[k][b] for k in scores])
            _, orc = oracle_alloc(scores, budgets, b)
            pr = probe_alloc(scores, budgets, b, feats)[1] if feats is not None else float("nan")
            head = orc - fixed
            cap = (pr - fixed) / head if feats is not None and head > 1e-9 else float("nan")
            print(f"{b:>11} {fixed:>8.3f} {orc:>8.3f} {head:>9.3f} {pr:>8.3f} "
                  f"{cap:>9.2f} {lens[b]/max(1,lens[0]):>10.2f}")
            table.append(dict(task=task, avg_budget=b, fixed=round(fixed, 4),
                              oracle=round(orc, 4), headroom=round(head, 4),
                              probe=round(pr, 4),
                              captured_fraction=round(cap, 3) if cap == cap else "",
                              length_ratio=round(lens[b] / max(1, lens[0]), 3),
                              full_kv=round(full, 4), n=len(idxs)))

    with open(os.path.join(out_dir, "table_headroom.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)

    if not HAVE_PLT:
        return

    # one figure: fixed vs oracle vs probe, per task
    tasks = sorted({r["task"] for r in table})
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        rs = [r for r in table if r["task"] == t]
        x = [r["avg_budget"] for r in rs]
        ax.plot(x, [r["fixed"] for r in rs], "o-", label="one fixed budget")
        ax.plot(x, [r["oracle"] for r in rs], "s-", label="per-request oracle")
        if rs[0]["probe"] == rs[0]["probe"]:
            ax.plot(x, [r["probe"] for r in rs], "^--", label="cheap predictor")
        ax.axhline(rs[0]["full_kv"], color="k", lw=.8, ls=":", label="full KV")
        ax.set_xscale("log", base=2); ax.set_xlabel("average KV budget (tokens)")
        ax.set_ylabel("task score"); ax.set_title(t); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig_headroom.png"), dpi=200)
    print("\nwrote table_headroom.csv and fig_headroom.png")


if __name__ == "__main__":
    main(sys.argv[1:])
