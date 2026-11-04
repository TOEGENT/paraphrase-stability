"""Grouped cross-validation of the MLP against baselines on train+val (test stays untouched).

Models: constant, ridge on [q,a], MLP on [q,a] / q only / a only.
--group-by root merges equations with the same true root, so no root is shared across folds.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from slm_answer_stability.models.embeddings import load_splits
from slm_answer_stability.models.mlp import build_model

D = 768


def root(question: str):
    m = re.match(r"Solve (.*) for (\w+)\.", question)
    if not m:
        return None
    eq, v = m.groups()
    left, right = eq.split("=")

    def f(x):
        sub = lambda s: re.sub(rf"\b{v}\b", f"F({x})", s)
        return eval(sub(left), {"F": F}) - eval(sub(right), {"F": F})  # shortcut: eval on dataset-generated text only

    a = f(0)
    b = f(1) - a
    return -a / b if b else None


def load(splits_dir: Path, features_dir: Path):
    rows = load_splits(splits_dir)
    meta, q, a = [], [], []
    for s in ("train", "val"):
        z = np.load(features_dir / f"{s}.npz", allow_pickle=False)
        assert z["question_id"].tolist() == [r["question_id"] for r in rows[s]]
        meta += rows[s]
        q.append(z["question"])
        a.append(z["answer"])
    return meta, np.concatenate(q), np.concatenate(a)


def answer_value(text: str):
    m = re.findall(r"=\s*(-?\d+(?:/\d+)?(?:\.\d+)?)", text)
    return float(F(m[-1])) if m else 0.0


def hand(row) -> list[float]:
    """Complexity features from question and answer text only (no equation solving)."""
    q, ans = row["question"], row["answer"]
    nums = [abs(int(n)) for n in re.findall(r"\d+", q)] or [0]
    v = answer_value(ans)
    return [len(q), len(re.findall(r"\d+", q)), max(nums), float(np.log10(1 + sum(nums))),
            q.count("("), q.count("*"), q.count("-"), q.count("+"), len(re.findall(r"[A-Za-z_]\b", q.split("Solve")[-1].split(" for ")[0])),
            len(ans), abs(v), float(np.log10(1 + abs(v))), float(v < 0), float(v == int(v)), float(abs(v) > max(nums))]


def correct_flag(row) -> float:
    r = root(row["question"])
    return float(r is not None and answer_value(row["answer"]) == float(r))


def folds(keys, k, seed):
    uniq = sorted(set(keys))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {u: i % k for i, u in enumerate(uniq)}
    return np.array([fold_of[x] for x in keys])


def fit_mlp(xtr, ytr, xva, yva, cfg, seed, dev):
    torch.manual_seed(seed)
    m = build_model(D, cfg["hidden_dim"], cfg["dropout"]).to(dev) if xtr.shape[1] == 2 * D else \
        torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], cfg["hidden_dim"]), torch.nn.ReLU(),
                            torch.nn.Dropout(cfg["dropout"]), torch.nn.Linear(cfg["hidden_dim"], 1)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    X, Y = torch.as_tensor(xtr, device=dev), torch.as_tensor(ytr, dtype=torch.float32, device=dev)
    XV, YV = torch.as_tensor(xva, device=dev), torch.as_tensor(yva, dtype=torch.float32, device=dev)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    best, state, stale = 1e9, None, 0
    for _ in range(cfg["max_epochs"]):
        m.train()
        for idx in torch.randperm(len(X), device=dev).split(cfg["batch_size"]):
            opt.zero_grad()
            bce(m(X[idx]).squeeze(-1), Y[idx]).backward()
            opt.step()
        m.eval()
        with torch.inference_mode():
            vl = bce(m(XV).squeeze(-1), YV).item()
        if vl < best - 1e-6:
            best, state, stale = vl, copy.deepcopy(m.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    m.load_state_dict(state)
    m.eval()
    return lambda x: torch.sigmoid(m(torch.as_tensor(x, device=dev)).squeeze(-1)).detach().cpu().numpy()


def ridge(xtr, ytr, alpha=10.0):
    mu = xtr.mean(0)
    xc = xtr - mu
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(xc.shape[1]), xc.T @ (ytr - ytr.mean()))
    return lambda x: np.clip((x - mu) @ w + ytr.mean(), 0, 1)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if a.std() and b.std() else float("nan")


def score(y, p, groups):
    e = np.abs(p - y)
    per = defaultdict(list)
    for g, x in zip(groups, e):
        per[g].append(x)
    return {"mae": e.mean(), "rmse": float(np.sqrt(((p - y) ** 2).mean())),
            "mae_group": float(np.mean([np.mean(v) for v in per.values()])), "spearman": spearman(y, p)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/mlp.example.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--group-by", choices=("source", "root"), default="source")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--extended", action="store_true", help="variant C: interaction/hand features, GBM; skips the plain models")
    a = ap.parse_args()
    if a.output.exists():
        sys.exit(f"output exists: {a.output}")
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    meta, Q, A = load(a.splits_dir, a.features_dir)
    y = np.array([r["target"] for r in meta], dtype=np.float32)
    src = [r["source_question_id"] for r in meta]
    if a.group_by == "root":
        root_of = {r["source_question_id"]: str(root(r["question"])) for r in meta if r["anchor_id"] == "original"}
        keys = [root_of[s] for s in src]
    else:
        keys = src
    print(f"{len(meta)} anchors, {len(set(src))} equations, {len(set(keys))} {a.group_by} groups", flush=True)
    feats = {"qa": np.concatenate([Q, A], 1), "q_only": Q, "a_only": A}
    results = defaultdict(list)
    if a.extended:
        H = np.array([hand(r) for r in meta], dtype=np.float32)
        C = np.array([correct_flag(r) for r in meta], dtype=np.float32)  # diagnostic ceiling only, never a real feature
        inter = np.concatenate([Q, A, Q * A, np.abs(Q - A)], 1)
    for seed in a.seeds:
        f = folds(keys, a.folds, seed)
        for fold in range(a.folds):
            te, rest = np.where(f == fold)[0], np.where(f != fold)[0]
            rg = sorted({keys[i] for i in rest})
            rng = np.random.default_rng(seed * 100 + fold)
            val_g = set(rng.choice(rg, max(1, len(rg) // 8), replace=False))
            va = np.array([i for i in rest if keys[i] in val_g])
            tr = np.array([i for i in rest if keys[i] not in val_g])
            preds = {"constant": np.full(len(te), y[tr].mean())}
            if a.extended:
                mu, sd = H[tr].mean(0), H[tr].std(0) + 1e-6
                Hs = ((H - mu) / sd).astype(np.float32)
                sets = {"mlp_hand": Hs, "mlp_qa_inter": inter, "mlp_qa_hand": np.concatenate([feats["qa"], Hs], 1)}
                for name, x in sets.items():
                    preds[name] = fit_mlp(x[tr], y[tr], x[va], y[va], cfg, seed, a.device)(x[te])
                for name, x in (("gbm_hand", H), ("gbm_hand+DIAG_correct", np.c_[H, C])):
                    g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed).fit(x[tr], y[tr])
                    preds[name] = np.clip(g.predict(x[te]), 0, 1)
            else:
                preds["ridge_qa"] = ridge(feats["qa"][tr], y[tr])(feats["qa"][te])
                for name, x in feats.items():
                    preds["mlp_" + name] = fit_mlp(x[tr], y[tr], x[va], y[va], cfg, seed, a.device)(x[te])
            for name, p in preds.items():
                results[name].append(score(y[te], np.asarray(p), [src[i] for i in te]))
            print(f"seed {seed} fold {fold}: " + " ".join(f"{n}={results[n][-1]['mae']:.3f}" for n in preds), flush=True)
    summary = {}
    for name, rs in results.items():
        summary[name] = {m: [float(np.mean([r[m] for r in rs])), float(np.std([r[m] for r in rs]))] for m in rs[0]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"group_by": a.group_by, "folds": a.folds, "seeds": a.seeds, "summary": summary}, indent=2), encoding="utf-8")
    for name, s in summary.items():
        print(f"{name:10s} " + " ".join(f"{m}={v[0]:.3f}±{v[1]:.3f}" for m, v in s.items()))


if __name__ == "__main__":
    main()
