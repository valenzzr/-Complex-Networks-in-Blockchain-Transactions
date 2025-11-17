import os
import glob
import argparse
import json
import random
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import networkx as nx
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import GConvGRU


# ----------------- utils -----------------
def set_seed(seed: int = 42):
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def latest_csv(pattern: str):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def to_tensor_2d(x):
    """Ensure (N,F) float32 torch.Tensor."""
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[:, None]
    return torch.tensor(arr, dtype=torch.float32)


def build_graph_from_edges(df_edges: pd.DataFrame) -> nx.DiGraph:
    """df_edges: columns [from,to,value]. Aggregate weight per (u,v)."""
    G = nx.DiGraph()
    for _, r in df_edges.iterrows():
        s, t, v = str(r["from"]), str(r["to"]), float(r["value"])
        if s == t:
            continue
        if G.has_edge(s, t):
            G[s][t]["weight"] += v
            G[s][t]["count"] = G[s][t].get("count", 1) + 1
        else:
            G.add_edge(s, t, weight=v, count=1)
    return G


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Temporal GNN runner (GConvGRU) with richer eval/plots")
    ap.add_argument("--dyn_csv", default="outputs_22wdata/temporal/dynamic_centrality_timeseries.csv",
                    help="Path to dynamic centrality time-series CSV")
    ap.add_argument("--tx_csv", default=None,
                    help="Path to ethereum_transactions CSV; if None, pick latest in outputs/")
    ap.add_argument("--out_dir", default="outputs_22wdata/temporal",
                    help="Directory to save GNN outputs")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    # visualization choices
    ap.add_argument("--topk", type=int, default=10, help="Top-K hubs to visualize in last window")
    ap.add_argument("--timeseries_nodes", type=int, default=3, help="How many representative nodes to plot over time")
    return ap.parse_args()


# ----------------- main -----------------
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) load dynamic centrality series (already computed upstream)
    if not os.path.exists(args.dyn_csv):
        raise FileNotFoundError(f"Cannot find {args.dyn_csv}. Please run earlier steps first.")
    df_dyn = pd.read_csv(args.dyn_csv)
    for c in ["node", "window_start", "window_end", "snapshot_index",
              "degree", "in_degree", "out_degree", "pagerank"]:
        if c not in df_dyn.columns:
            raise ValueError(f"Column '{c}' missing in {args.dyn_csv}")
    df_dyn["node"] = df_dyn["node"].astype(str)

    # 2) load transactions CSV to rebuild edges per snapshot window
    if args.tx_csv is None:
        tx_csv = latest_csv("outputs_22wdata/ethereum_transactions_*.csv") or latest_csv("outputs/ethereum_transactions_*.csv")
        if tx_csv is None:
            raise FileNotFoundError("No ethereum_transactions_*.csv found in outputs_/outputs_22wdata.")
    else:
        tx_csv = args.tx_csv
    df_all = pd.read_csv(tx_csv)
    for c in ["from", "to", "value", "timeStamp"]:
        if c not in df_all.columns:
            raise ValueError(f"Column '{c}' missing in {tx_csv}")
    df_all["timeStamp"] = pd.to_datetime(df_all["timeStamp"], errors="coerce", utc=True)
    df_all["date"] = df_all["timeStamp"].dt.date

    # 3) build node universe and snapshot metadata
    nodes_all = sorted(df_dyn["node"].unique())
    node2idx = {n: i for i, n in enumerate(nodes_all)}
    N = len(nodes_all)

    snaps_meta = (
        df_dyn[["snapshot_index", "window_start", "window_end"]]
        .drop_duplicates("snapshot_index")
        .sort_values("snapshot_index")
    )

    edge_indices, edge_weights, features = [], [], []
    # Feature columns: [degree, in_degree, out_degree, pagerank]
    for _, row in snaps_meta.iterrows():
        idx = int(row["snapshot_index"])
        ws = pd.to_datetime(row["window_start"]).date()
        we = pd.to_datetime(row["window_end"]).date()

        mask = (df_all["date"] >= ws) & (df_all["date"] <= we)
        df_win = df_all.loc[mask, ["from", "to", "value"]]
        if df_win.empty:
            continue

        Gw = build_graph_from_edges(df_win)

        edges, weights = [], []
        for u, v, data in Gw.edges(data=True):
            if (u in node2idx) and (v in node2idx):
                edges.append((node2idx[u], node2idx[v]))
                weights.append(float(data.get("weight", 1.0)))
        if not edges:
            continue

        edge_index_t = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight_t = torch.tensor(weights, dtype=torch.float32)

        sub = df_dyn[df_dyn["snapshot_index"] == idx].set_index("node")
        x = np.zeros((N, 4), dtype=np.float32)
        for n in nodes_all:
            if n in sub.index:
                x[node2idx[n], 0] = float(sub.loc[n, "degree"])
                x[node2idx[n], 1] = float(sub.loc[n, "in_degree"])
                x[node2idx[n], 2] = float(sub.loc[n, "out_degree"])
                x[node2idx[n], 3] = float(sub.loc[n, "pagerank"])
        X_t = to_tensor_2d(x)

        features.append(X_t)
        edge_indices.append(edge_index_t)
        edge_weights.append(edge_weight_t)

    # target: next snapshot pagerank
    pr_by_snap = []
    for idx in snaps_meta["snapshot_index"].astype(int).tolist():
        sub = df_dyn[df_dyn["snapshot_index"] == idx].set_index("node")
        y = np.zeros((N,), dtype=np.float32)
        for n in nodes_all:
            if n in sub.index:
                y[node2idx[n]] = float(sub.loc[n, "pagerank"])
        pr_by_snap.append(y)

    min_len = min(len(features), len(pr_by_snap))
    if min_len < 3:
        print("Too few snapshots for temporal GNN (need >=3). Exit.")
        return

    features     = features[:min_len-1]
    edge_indices = edge_indices[:min_len-1]
    edge_weights = edge_weights[:min_len-1]
    targets      = pr_by_snap[1:min_len]   # list of (N,) numpy arrays

    dataset = DynamicGraphTemporalSignal(
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        features=features,
        targets=targets
    )

    # ----- standardization on training segment -----
    T = len(dataset.features)
    split = int(0.7 * T)
    X_train = torch.stack(dataset.features[:split], dim=0)  # (T1, N, 4)
    y_train = torch.tensor(np.stack(dataset.targets[:split]), dtype=torch.float32)  # (T1, N)

    x_mean = X_train.mean(dim=(0, 1), keepdim=True)      # (1,1,4)
    x_std  = X_train.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean()
    y_std  = y_train.std().clamp_min(1e-6)

    def norm_x(x: torch.Tensor) -> torch.Tensor:
        return (x - x_mean.squeeze(0)) / x_std.squeeze(0)

    def norm_y(y_np: np.ndarray) -> torch.Tensor:
        y = torch.tensor(y_np, dtype=torch.float32)
        return (y - y_mean) / y_std

    def denorm_y(y_t: torch.Tensor) -> torch.Tensor:
        return y_t * y_std + y_mean

    # ----- model -----
    class GCRURegressor(nn.Module):
        def __init__(self, in_feat=4, hidden=32, out_feat=1, dropout=0.2):
            super().__init__()
            self.rnn = GConvGRU(in_channels=in_feat, out_channels=hidden, K=2)
            self.drop = nn.Dropout(dropout)
            self.lin = nn.Linear(hidden, out_feat)
        def forward(self, x, edge_index, edge_weight=None):
            h = self.rnn(x, edge_index, edge_weight)  # (N, hidden)
            h = self.drop(h)
            out = self.lin(h).squeeze(-1)             # (N,)
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCRURegressor(in_feat=4, hidden=args.hidden, out_feat=1, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loss_fn = nn.L1Loss()

    # ----- training loop with early stopping -----
    best_val, stale, patience = float("inf"), 0, 6
    train_hist, val_hist = [], []

    def step_loop(start, end, train=True):
        model.train(mode=train)
        total, count = 0.0, 0
        for t in range(start, end):
            x  = norm_x(dataset.features[t]).to(device)                    # (N,4)
            ei = dataset.edge_indices[t].to(device)                        # (2,E)
            ew = dataset.edge_weights[t].to(device)                        # (E,)
            y  = norm_y(dataset.targets[t]).to(device)                     # (N,)

            pred = model(x, ei, ew)                                        # (N,)
            loss = loss_fn(pred, y)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                opt.step()

            total += float(loss.item()); count += 1
        return total / max(1, count)

    print(f"Training with T={T} snapshots, split={split}/{T-split} (train/val)")
    for ep in range(1, args.epochs + 1):
        tr = step_loop(0, split, train=True)
        va = step_loop(split, T, train=False)
        train_hist.append(tr); val_hist.append(va)

        scheduler.step(va)
        if va < best_val - 1e-4:
            best_val = va; stale = 0
        else:
            stale += 1

        if ep % 5 == 0:
            cur_lr = opt.param_groups[0]["lr"]
            print(f"[TemporalGNN] Epoch {ep:02d} | train {tr:.4f} | val {va:.4f} | lr {cur_lr:.2e}")

        if stale >= patience:
            print(f"Early stop at epoch {ep} (best val={best_val:.4f})")
            break

    # ----- save loss curves -----
    plt.figure(figsize=(7, 4))
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val")
    plt.title("Temporal GNN (GConvGRU) L1-loss (standardized)")
    plt.xlabel("Epoch"); plt.ylabel("L1")
    plt.legend(); plt.tight_layout()
    loss_png = os.path.join(args.out_dir, "temporal_gnn_loss.png")
    plt.savefig(loss_png, dpi=300, bbox_inches="tight"); plt.close()
    print("Saved:", loss_png)

    # ----- predict all validation windows -----
    val_rows = []
    for t in range(split, T):
        x  = norm_x(dataset.features[t]).to(device)
        ei = dataset.edge_indices[t].to(device)
        ew = dataset.edge_weights[t].to(device)
        y_true = torch.tensor(dataset.targets[t], dtype=torch.float32)

        with torch.no_grad():
            yhat = denorm_y(model(x, ei, ew).cpu()).numpy()
        ytru = y_true.numpy()
        for nid, n in enumerate(nodes_all):
            val_rows.append({"t": int(t), "node": n,
                             "y_true_pagerank": float(ytru[nid]),
                             "y_pred_pagerank": float(yhat[nid])})

    df_val_pred = pd.DataFrame(val_rows)
    val_csv = os.path.join(args.out_dir, "temporal_gnn_val_predictions.csv")
    df_val_pred.to_csv(val_csv, index=False)
    print("Saved:", val_csv)

    # ----- window metrics (MAE & Spearman) -----
    win_metrics = []
    for t, g in df_val_pred.groupby("t"):
        ytru = g["y_true_pagerank"].to_numpy()
        yhat = g["y_pred_pagerank"].to_numpy()
        mae = np.mean(np.abs(yhat - ytru))
        try:
            rho = spearmanr(ytru, yhat).statistic
        except Exception:
            rho = np.nan
        win_metrics.append({"t": int(t), "mae": float(mae), "spearman_rho": float(rho)})
    df_metrics = pd.DataFrame(win_metrics).sort_values("t")
    df_metrics.to_csv(os.path.join(args.out_dir, "temporal_gnn_val_metrics.csv"), index=False)

    # plot metrics
    plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.plot(df_metrics["t"], df_metrics["mae"], label="MAE")
    ax.set_xlabel("Validation window index"); ax.set_ylabel("MAE")
    ax2 = ax.twinx()
    ax2.plot(df_metrics["t"], df_metrics["spearman_rho"], label="Spearman ρ", linestyle="--")
    ax2.set_ylabel("Spearman ρ")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.title("Validation metrics per window")
    plt.tight_layout()
    metrics_png = os.path.join(args.out_dir, "temporal_gnn_val_metrics.png")
    plt.savefig(metrics_png, dpi=300, bbox_inches="tight"); plt.close()
    print("Saved:", metrics_png)

    # ----- last validation snapshot scatter (True vs Pred) -----
    last_t = df_val_pred["t"].max()
    g_last = df_val_pred[df_val_pred["t"] == last_t]
    plt.figure(figsize=(5.5, 5))
    plt.scatter(g_last["y_true_pagerank"], g_last["y_pred_pagerank"], s=12, alpha=0.5)
    mn = min(g_last["y_true_pagerank"].min(), g_last["y_pred_pagerank"].min())
    mx = max(g_last["y_true_pagerank"].max(), g_last["y_pred_pagerank"].max())
    plt.plot([mn, mx], [mn, mx], "r-", linewidth=1.5, label="Ideal 1:1")
    plt.xlabel("True PageRank (t)"); plt.ylabel("Predicted PageRank (t)")
    plt.title(f"Last-val snapshot scatter (t={last_t})")
    plt.legend(); plt.grid(True); plt.tight_layout()
    scat_png = os.path.join(args.out_dir, "temporal_gnn_last_val_scatter.png")
    plt.savefig(scat_png, dpi=300, bbox_inches="tight"); plt.close()
    print("Saved:", scat_png)

    # ----- last window: Top-K precision/Jaccard & bar comparison -----
    K_list = [5, 10, 20]
    # ranks descending
    g_last = g_last.copy()
    g_last["rank_true"] = g_last["y_true_pagerank"].rank(method="first", ascending=False)
    g_last["rank_pred"] = g_last["y_pred_pagerank"].rank(method="first", ascending=False)

    rows = []
    for K in K_list:
        top_true = set(g_last.nsmallest(K, "rank_true")["node"])
        top_pred = set(g_last.nsmallest(K, "rank_pred")["node"])
        inter = len(top_true & top_pred)
        union = len(top_true | top_pred)
        precision = inter / max(1, len(top_pred))
        jacc = inter / max(1, union)
        rows.append({"K": K, "precision_at_k": precision, "jaccard": jacc})
    df_k = pd.DataFrame(rows)
    df_k.to_csv(os.path.join(args.out_dir, "temporal_gnn_last_val_topk_metrics.csv"), index=False)

    plt.figure(figsize=(6,4))
    width = 0.35
    xs = np.arange(len(K_list))
    plt.bar(xs - width/2, df_k["precision_at_k"], width, label="Precision@K")
    plt.bar(xs + width/2, df_k["jaccard"], width, label="Jaccard")
    plt.xticks(xs, [str(k) for k in K_list])
    plt.ylim(0, 1.0)
    plt.xlabel("K"); plt.ylabel("Score")
    plt.title(f"Last-val Top-K overlap (t={last_t})")
    plt.legend(); plt.tight_layout()
    topk_png = os.path.join(args.out_dir, "temporal_gnn_last_val_topk.png")
    plt.savefig(topk_png, dpi=300, bbox_inches="tight"); plt.close()
    print("Saved:", topk_png)

    # bar compare for Top-10 true nodes
    top_true10 = g_last.nsmallest(args.topk, "rank_true")
    plt.figure(figsize=(9,5))
    idx = np.arange(len(top_true10))
    plt.bar(idx - 0.2, top_true10["y_true_pagerank"], width=0.4, label="True")
    plt.bar(idx + 0.2, top_true10["y_pred_pagerank"], width=0.4, label="Pred")
    labels = [n[:10] + "..." for n in top_true10["node"]]
    plt.xticks(idx, labels, rotation=45, ha="right")
    plt.ylabel("PageRank"); plt.title(f"Last-val Top-{args.topk} nodes: true vs pred")
    plt.legend(); plt.tight_layout()
    bar_png = os.path.join(args.out_dir, "temporal_gnn_last_val_top10_bar.png")
    plt.savefig(bar_png, dpi=300, bbox_inches="tight"); plt.close()
    print("Saved:", bar_png)

    # ----- representative nodes time series (true vs pred) over validation -----
    # pick nodes with highest average true PageRank across validation windows
    avg_true = (
        df_val_pred.groupby("node")["y_true_pagerank"]
        .mean()
        .sort_values(ascending=False)
        .head(args.timeseries_nodes)
        .index.tolist()
    )
    plt.figure(figsize=(10,5))
    for n in avg_true:
        g_n = df_val_pred[df_val_pred["node"] == n].sort_values("t")
        plt.plot(g_n["t"], g_n["y_true_pagerank"], linewidth=1.2, label=f"{n[:8]}.. (true)")
        plt.plot(g_n["t"], g_n["y_pred_pagerank"], linewidth=1.2, linestyle="--", label=f"{n[:8]}.. (pred)")
    plt.xlabel("Validation window index"); plt.ylabel("PageRank")
    plt.title(f"Representative nodes time series (val segment, top {args.timeseries_nodes})")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    ts_png = os.path.join(args.out_dir, "temporal_gnn_val_timeseries_nodes.png")
    plt.savefig(ts_png, dpi=300, bbox_inches="tight"); plt.close()
    print("Saved:", ts_png)

    # ----- meta -----
    meta = dict(
        epochs=len(train_hist),
        best_val=float(best_val),
        hidden=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        dyn_csv=os.path.abspath(args.dyn_csv),
        tx_csv=os.path.abspath(tx_csv),
        snapshots=T,
        train_split=split
    )
    save_json(meta, os.path.join(args.out_dir, "temporal_gnn_meta.json"))
    print("Done. Artifacts saved to:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
