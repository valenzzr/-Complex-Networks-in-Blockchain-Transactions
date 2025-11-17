import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
import networkx as nx
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import GConvGRU

# ----------------- utils -----------------
def build_graph_from_edges(df_edges):
    """df_edges: columns [from,to,value] (ETH). Aggregate weight per (u,v)."""
    G = nx.DiGraph()
    for _, r in df_edges.iterrows():
        s, t, v = r["from"], r["to"], float(r["value"])
        if s == t:
            continue
        if G.has_edge(s, t):
            G[s][t]["weight"] += v
            G[s][t]["count"] = G[s][t].get("count", 1) + 1
        else:
            G.add_edge(s, t, weight=v, count=1)
    return G

def latest_csv(pattern):
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

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Temporal GNN runner (GConvGRU)")
    ap.add_argument("--dyn_csv", default="outputs_22wdata/temporal/dynamic_centrality_timeseries.csv",
                    help="Path to dynamic centrality time-series CSV")
    ap.add_argument("--tx_csv", default=None,
                    help="Path to ethereum_transactions CSV; if None, pick latest in outputs/")
    ap.add_argument("--out_dir", default="outputs_22wdata/temporal",
                    help="Directory to save GNN outputs")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    return ap.parse_args()

# ----------------- main -----------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) load dynamic centrality series
    if not os.path.exists(args.dyn_csv):
        raise FileNotFoundError(f"Cannot find {args.dyn_csv}. Please run earlier steps first.")
    df_dyn = pd.read_csv(args.dyn_csv)
    # Check required columns
    for c in ["node","window_start","window_end","snapshot_index","degree","in_degree","out_degree","pagerank"]:
        if c not in df_dyn.columns:
            raise ValueError(f"Column '{c}' missing in {args.dyn_csv}")

    df_dyn["node"] = df_dyn["node"].astype(str)

    # Load transactions CSV (latest if not provided)
    if args.tx_csv is None:
        tx_csv = latest_csv("outputs_22wdata/ethereum_transactions_*.csv")
        if tx_csv is None:
            raise FileNotFoundError("No ethereum_transactions_*.csv found in outputs/.")
    else:
        tx_csv = args.tx_csv
    df_all = pd.read_csv(tx_csv)
    # Check required columns
    for c in ["from","to","value","timeStamp"]:
        if c not in df_all.columns:
            raise ValueError(f"Column '{c}' missing in {tx_csv}")

    # Preprocess timestamp
    df_all["timeStamp"] = pd.to_datetime(df_all["timeStamp"], errors="coerce", utc=True)
    df_all["date"] = df_all["timeStamp"].dt.date

    # Build complete node set and index
    nodes_all = sorted(df_dyn["node"].unique())
    node2idx = {n:i for i,n in enumerate(nodes_all)}
    N = len(nodes_all)

    # Build (edge_index, edge_weight, X_t) for each snapshot; target is next window's pagerank
    # For each snapshot_index, take the first record of window_start/window_end after grouping
    snaps_meta = (df_dyn[["snapshot_index","window_start","window_end"]]
                  .drop_duplicates("snapshot_index")
                  .sort_values("snapshot_index"))
    edge_indices, edge_weights, features = [], [], []
    # Feature columns: 4 dimensions [degree, in_degree, out_degree, pagerank]
    for _, row in snaps_meta.iterrows():
        idx = int(row["snapshot_index"])
        ws = pd.to_datetime(row["window_start"]).date()
        we = pd.to_datetime(row["window_end"]).date()

        mask = (df_all["date"] >= ws) & (df_all["date"] <= we)
        df_win = df_all.loc[mask, ["from","to","value"]]
        if df_win.empty:
            continue

        Gw = build_graph_from_edges(df_win)
        # 边集
        edges, weights = [], []
        for u, v, data in Gw.edges(data=True):
            if (u in node2idx) and (v in node2idx):
                edges.append((node2idx[u], node2idx[v]))
                weights.append(float(data.get("weight", 1.0)))
        if not edges:
            continue
        edge_index_t = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight_t = torch.tensor(weights, dtype=torch.float32)

        # X_t
        sub = df_dyn[df_dyn["snapshot_index"] == idx].set_index("node")
        x = np.zeros((N, 4), dtype=np.float32)
        for n in nodes_all:
            if n in sub.index:
                x[node2idx[n], 0] = sub.loc[n, "degree"]
                x[node2idx[n], 1] = sub.loc[n, "in_degree"]
                x[node2idx[n], 2] = sub.loc[n, "out_degree"]
                x[node2idx[n], 3] = sub.loc[n, "pagerank"]
        X_t = to_tensor_2d(x)

        features.append(X_t)
        edge_indices.append(edge_index_t)
        edge_weights.append(edge_weight_t)

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

    # 构造 Dataset
    dataset = DynamicGraphTemporalSignal(
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        features=features,
        targets=targets
    )

    # 5) model definition
    class GCRURegressor(nn.Module):
        def __init__(self, in_feat=4, hidden=16, out_feat=1):
            super().__init__()
            self.rnn = GConvGRU(in_channels=in_feat, out_channels=hidden, K=2)
            self.lin = nn.Linear(hidden, out_feat)
        def forward(self, x, edge_index, edge_weight=None):
            # x: (N,F)
            h = self.rnn(x, edge_index, edge_weight)  # (N, hidden)
            out = self.lin(h).squeeze(-1)             # (N,)
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCRURegressor(in_feat=4, hidden=args.hidden, out_feat=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    # 6) training loop
    T = len(dataset.features)
    split = int(0.7 * T)
    def step_loop(start, end, train=True):
        model.train(mode=train)
        total, count = 0.0, 0
        for t in range(start, end):
            x  = dataset.features[t].to(device)                 # (N,4)
            ei = dataset.edge_indices[t].to(device)             # (2,E)
            ew = dataset.edge_weights[t].to(device)             # (E,)
            y  = torch.tensor(dataset.targets[t], dtype=torch.float32, device=device)  # (N,)

            pred = model(x, ei, ew)                             # (N,)
            loss = loss_fn(pred, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); count += 1
        return total / max(1, count)

    train_hist, val_hist = [], []
    for ep in range(1, args.epochs + 1):
        tr = step_loop(0, split, train=True)
        va = step_loop(split, T, train=False)
        train_hist.append(tr); val_hist.append(va)
        if ep % 5 == 0:
            print(f"[TemporalGNN] Epoch {ep:02d} | train {tr:.4f} | val {va:.4f}")

    # 7) visualize loss curves
    plt.figure(figsize=(7,4))
    plt.plot(train_hist, label="train"); plt.plot(val_hist, label="val")
    plt.title("Temporal GNN (GConvGRU) L1-loss"); plt.xlabel("Epoch"); plt.ylabel("L1")
    plt.legend(); plt.tight_layout()
    loss_png = os.path.join(args.out_dir, "temporal_gnn_loss.png")
    plt.savefig(loss_png, dpi=300, bbox_inches="tight"); plt.close()
    print("Saved:", loss_png)

    # 8) last snapshot prediction
    x  = dataset.features[-1].to(device)
    ei = dataset.edge_indices[-1].to(device)
    ew = dataset.edge_weights[-1].to(device)
    y  = torch.tensor(dataset.targets[-1], dtype=torch.float32, device=device)
    with torch.no_grad():
        yhat = model(x, ei, ew).cpu().numpy()
    df_pred = pd.DataFrame({
        "node": nodes_all,
        "y_true_pagerank": y.cpu().numpy(),
        "y_pred_pagerank": yhat
    }).sort_values("y_true_pagerank", ascending=False)
    pred_csv = os.path.join(args.out_dir, "temporal_gnn_last_snapshot_prediction.csv")
    df_pred.to_csv(pred_csv, index=False)
    print("Saved:", pred_csv)

if __name__ == "__main__":
    main()