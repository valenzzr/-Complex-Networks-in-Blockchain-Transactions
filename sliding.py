#!/usr/bin/env python
# draw_dynamic_timeseries.py
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "outputs_22wdata/temporal/dynamic_centrality_timeseries.csv"   # change if needed
OUT = "outputs_22wdata/temporal"
TOPK_PLOT = 10

os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV, parse_dates=["window_start","window_end"])
# rank nodes by mean PageRank
mean_pr = df.groupby("node")["pagerank"].mean().sort_values(ascending=False)
# keep nodes that appear in enough snapshots
presence = df.groupby("node")["snapshot_index"].nunique().sort_values(ascending=False)
eligible = presence[presence >= max(3, int(0.2 * df["snapshot_index"].nunique()))].index
focus_nodes = [n for n in mean_pr.index if n in eligible][:TOPK_PLOT]

all_snaps = sorted(df["snapshot_index"].unique())
snap_to_date = df.groupby("snapshot_index")["window_end"].first().to_dict()

for metric in ["pagerank", "betweenness", "kcore"]:
    plt.figure(figsize=(11, 6))
    for n in focus_nodes:
        sub = df[df["node"] == n].set_index("snapshot_index").sort_index()
        sub = sub.reindex(all_snaps)
        # fill gaps
        sub[metric] = sub[metric].fillna(0.0)
        x = [snap_to_date[s] for s in all_snaps]
        y = sub[metric].to_numpy()
        plt.plot(x, y, marker="o", markersize=2, linewidth=1.2, alpha=0.9, label=(n[:8]+"…"))
    plt.title(f"Top-{TOPK_PLOT} nodes: {metric} over sliding windows")
    plt.xlabel("Window end date"); plt.ylabel(metric)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if metric == "pagerank":
        plt.legend(ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"timeseries_{metric}.png"), dpi=300, bbox_inches="tight")
    plt.close()

print("Saved plots to", OUT)
