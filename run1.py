# =====================================================
# Ethereum Transaction Network Analysis (Enhanced + Temporal)
# Adds: structural metrics, rigorous power-law tests,
#       multi-dimensional centralization (value flows),
#       [NEW] temporal slicing & dynamic centrality,
#       [NEW] optional temporal GNN baseline (GConvGRU)
# =====================================================

import os
import sys
import json
import math
import requests
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from datetime import datetime, timedelta

# Optional: rigorous power-law analysis
try:
    import powerlaw
    HAS_POWERLAW = True
except Exception:
    HAS_POWERLAW = False

# Optional: Temporal GNN (PyTorch Geometric Temporal)
try:
    import torch
    from torch import nn
    from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
    from torch_geometric_temporal.nn.recurrent import GConvGRU
    HAS_PGT = True
    print(f"Using PyTorch version: {torch.__version__}")
except Exception:
    HAS_PGT = False
    print("PyTorch Geometric Temporal not available; skipping temporal GNN parts.")

# =========================
# 0. CONFIGURATION
# =========================

# Strongly require env var for key (fail fast)
try:
    API_KEY = os.getenv("ETHERSCAN_API_KEY", "7E3QBKVNRYBITR1IYWG4XK3VQ21DQNE3PS")
except KeyError:
    sys.exit("Missing env var ETHERSCAN_API_KEY. Please set it before running.")

# Known active/visible wallets as seeds (same as before)
seed_addresses_level1 = [
    "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Binance hot wallet
    "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",  # Vitalik Buterin
    "0xE592427A0AEce92De3Edee1F18E0157C05861564",  # Uniswap V3 Router
    "0x564286362092D8e7936f0549571a803B203aAceD",  # Tether Treasury
]

MAX_TX = 3000         # per-address max transactions
DO_SECOND_HOP = True  # expand to top counterparties

# Betweenness can be slow on large graphs; put a soft guard
BETWEENNESS_MAX_NODES = 15000
BETWEENNESS_K_SAMPLE = None  # e.g. set to 500 for approximation

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# -------- [NEW - Temporal] time slicing params --------
TEMP_DIR = os.path.join(OUTDIR, "temporal")
os.makedirs(TEMP_DIR, exist_ok=True)

WINDOW_DAYS = 30            # 滑动窗口大小（天）
STEP_DAYS = 7               # 步长（天）
TOPK_PLOT = 10              # 时序图中展示的Top-K节点（以全局PR均值排序）
BETWEENNESS_SAMPLE_K = 400  # 在大图上对每个快照近似计算介数（None=精确；推荐采样以提速）


# =========================
# Utility functions
# =========================

def gini(x):
    """Gini coefficient for a 1-D array-like (non-negative)."""
    arr = np.array(x, dtype=float)
    arr = arr[arr >= 0]
    if arr.size == 0:
        return float("nan")
    arr.sort()
    n = arr.size
    cumx = np.cumsum(arr)
    if cumx[-1] == 0:
        return 0.0
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def topk_share(x, k_ratio=0.1):
    """Share of total held by top-k_ratio fraction (e.g. top 10%)."""
    arr = np.array(x, dtype=float)
    arr = arr[arr >= 0]
    if arr.size == 0:
        return float("nan")
    arr.sort()
    k = max(1, int(math.ceil(k_ratio * arr.size)))
    return np.sum(arr[-k:]) / np.sum(arr) if np.sum(arr) > 0 else float("nan")

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_graph_from_edges(df_edges):
    """Given df with columns [from,to,value], build DiGraph with weight=count aggregation."""
    G = nx.DiGraph()
    for _, row in df_edges.iterrows():
        s, t, v = row["from"], row["to"], float(row["value"])
        if s == t:
            continue
        if G.has_edge(s, t):
            G[s][t]["weight"] += v
            G[s][t]["count"] = G[s][t].get("count", 1) + 1
        else:
            G.add_edge(s, t, weight=v, count=1)
    return G


# =========================
# 1. ETHERSCAN DATA FETCH
# =========================

def fetch_transactions_for_address(address, api_key, max_tx=MAX_TX):
    """
    Fetch normal txs for an address on Ethereum mainnet via Etherscan v2 API.
    Returns DataFrame with columns: from, to, value(ETH), timeStamp(datetime), hash
    """
    url = (
        "https://api.etherscan.io/v2/api"
        f"?chainid=1&module=account&action=txlist"
        f"&address={address}"
        f"&startblock=0&endblock=9999999999"
        f"&page=1&offset={max_tx}"
        f"&sort=asc"
        f"&apikey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
    except Exception:
        return pd.DataFrame(columns=["from", "to", "value", "timeStamp", "hash"])

    if data.get("status") == "1" and "result" in data:
        df = pd.DataFrame(data["result"])
        if df.empty:
            return pd.DataFrame(columns=["from","to","value","timeStamp","hash","contractAddress"])

        keep_cols = ["from","to","value","timeStamp","hash","contractAddress"]
        for c in keep_cols:
            if c not in df.columns: df[c] = None
        df = df[keep_cols].copy()

        # 统一清洗
        df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0) / 1e18
        df["timeStamp"] = pd.to_datetime(pd.to_numeric(df["timeStamp"], errors="coerce"), unit="s", errors="coerce")
        for c in ("from","to","contractAddress"):
            df[c] = df[c].astype(str).str.strip()

        # 合约创建：to=="" 用 contractAddress 回填
        mask_create = (df["to"] == "") & df["contractAddress"].str.match(r"^0x[0-9a-fA-F]{40}$", na=False)
        df.loc[mask_create, "to"] = df.loc[mask_create, "contractAddress"]

        # 显式把 "" 变 NaN 后再丢弃
        df.replace({"": np.nan}, inplace=True)
        df = df.dropna(subset=["from","to","timeStamp"])

        # 强校验：只保留合法 42 字符地址
        addr_ok = r"^0x[0-9a-fA-F]{40}$"
        df = df[df["from"].str.match(addr_ok, na=False) & df["to"].str.match(addr_ok, na=False)]

        # 最后裁列
        df = df[["from","to","value","timeStamp","hash"]]
        return df
    return pd.DataFrame(columns=["from", "to", "value", "timeStamp", "hash"])


print("=== Fetching seed addresses (1-hop) ===")
level1_list = []
for addr in tqdm(seed_addresses_level1):
    df_addr = fetch_transactions_for_address(addr, API_KEY)
    df_addr["seed_source"] = addr
    level1_list.append(df_addr)

df_level1 = pd.concat(level1_list, ignore_index=True).drop_duplicates(subset=["hash"])

# 2-hop expansion: pick top frequent counterparties from level1
if DO_SECOND_HOP:
    print("=== Selecting top neighbor addresses for 2-hop expansion ===")
    neighbors_all = pd.concat([df_level1["from"], df_level1["to"]], ignore_index=True)
    top_neighbors = neighbors_all.value_counts().head(30).index.tolist()

    level2_list = []
    for addr in tqdm(top_neighbors):
        if addr in seed_addresses_level1:
            continue
        df_addr = fetch_transactions_for_address(addr, API_KEY)
        df_addr["seed_source"] = addr
        level2_list.append(df_addr)
    df_level2 = pd.concat(level2_list, ignore_index=True).drop_duplicates(subset=["hash"]) if level2_list else pd.DataFrame(columns=df_level1.columns)
else:
    df_level2 = pd.DataFrame(columns=df_level1.columns)

df_all = pd.concat([df_level1, df_level2], ignore_index=True).drop_duplicates(subset=["hash"])
print(f"Total unique transactions collected: {len(df_all)}")

all_nodes = pd.unique(pd.concat([df_all["from"], df_all["to"]], ignore_index=True))
print(f"Unique addresses involved: {len(all_nodes)}")

# Save raw tx CSV
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_name = os.path.join(OUTDIR, f"ethereum_transactions_{ts}.csv")
df_all.to_csv(csv_name, index=False)
print(f"💾 Saved raw data to {csv_name}")


# =========================
# 2. BUILD DIRECTED GRAPH (cumulative)
# =========================

print("\n=== Building transaction graph ===")
G = build_graph_from_edges(df_all[["from", "to", "value"]])

print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
UG = G.to_undirected()


# =========================
# 3. STATIC NETWORK ANALYSIS (original + extended)
# =========================

print("\n=== Computing network metrics ===")
deg_total = dict(G.degree())
deg_in    = dict(G.in_degree())
deg_out   = dict(G.out_degree())

pagerank = nx.pagerank(G, alpha=0.85)
clustering = nx.clustering(UG)

avg_deg   = float(np.mean(list(deg_total.values()))) if deg_total else float("nan")
avg_clust = float(np.mean(list(clustering.values()))) if clustering else float("nan")
print(f"Average degree: {avg_deg:.4f}")
print(f"Average clustering coefficient: {avg_clust:.4f}")

# Top 10 by PageRank
top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 wallets by PageRank:")
for i, (node, val) in enumerate(top_pr, 1):
    print(f"{i}. {node} | PR={val:.6f}")

# Giant component fraction (on UG)
components = sorted(nx.connected_components(UG), key=len, reverse=True)
if components:
    giant_nodes = components[0]
    Gc = G.subgraph(giant_nodes)
    frac_gc = len(Gc.nodes()) / G.number_of_nodes() if G.number_of_nodes() else 0.0
    print(f"\nGiant component fraction: {frac_gc:.2%}")
else:
    frac_gc = 0.0
    print("\nGiant component fraction: N/A")

# ====== Extended structural metrics ======
print("\n=== Extended structural metrics ===")
core_numbers = nx.core_number(UG) if UG.number_of_nodes() > 0 else {}
try:
    richclub = nx.rich_club_coefficient(UG, normalized=False)
except Exception:
    richclub = {}

try:
    assort_undirected = nx.degree_pearson_correlation_coefficient(UG) if UG.number_of_edges() else float("nan")
except Exception:
    assort_undirected = float("nan")

try:
    assort_out_in = nx.degree_assortativity_coefficient(G, x='out', y='in') if G.number_of_edges() else float("nan")
except Exception:
    assort_out_in = float("nan")

try:
    hubs, authorities = nx.hits(G, max_iter=1000, normalized=True)
except Exception:
    hubs, authorities = ({n: 0.0 for n in G.nodes()},
                         {n: 0.0 for n in G.nodes()})

# ---- Betweenness: k-sampling on medium/large graphs ----
N = G.number_of_nodes()
M = G.number_of_edges()
BETWEENNESS_AVAILABLE = False
betweenness = {}

if N == 0:
    pass
else:
    # 自动选择 k（你也可以手动改）
    if N <= 5000:
        k_for_btw = None               # 精确
    elif N <= 30000:
        k_for_btw = min(1000, int(np.ceil(0.05 * N)))
    else:
        k_for_btw = min(2000, int(np.ceil(0.03 * N)))

    print(f"Betweenness centrality: N={N}, M={M}, k={k_for_btw if k_for_btw is not None else 'exact'}")
    try:
        betweenness = nx.betweenness_centrality(
            G,
            k=k_for_btw,           # None=精确；整数=采样
            normalized=True,
            weight=None,
            endpoints=False,
            seed=0 if k_for_btw is not None else None
        )
        # 数值清洗（确保都是有限数）
        for n in list(betweenness.keys()):
            v = betweenness[n]
            betweenness[n] = float(v) if np.isfinite(v) else 0.0
        BETWEENNESS_AVAILABLE = True
    except Exception as e:
        print(f"[warn] betweenness failed ({type(e).__name__}): {e}. Will skip exporting this field.")
        betweenness = {}   # 留空，表示不可用

print(f"Assortativity (UG, degree Pearson): {assort_undirected:.4f}")
print(f"Assortativity (DiGraph, out→in):   {assort_out_in:.4f}")

# =========================
# 4. DEGREE DISTRIBUTION (Histogram + Rigorous Power-law)
# =========================

print("\n=== Degree distribution & power-law tests ===")
deg_arr = np.array(list(deg_total.values()), dtype=int)
deg_arr = deg_arr[deg_arr > 0]

plt.figure(figsize=(8,5))
plt.hist(deg_arr, bins=100, log=True)
plt.xlabel("Degree k")
plt.ylabel("Frequency (log scale)")
plt.title("Degree Distribution of Ethereum Transaction Network")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "figure_degree_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()

powerlaw_summary = {}
if HAS_POWERLAW and deg_arr.size >= 50:
    fit = powerlaw.Fit(deg_arr, discrete=True, verbose=False)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    D = fit.power_law.KS()
    R, p = fit.distribution_compare('power_law', 'lognormal')
    powerlaw_summary = {
        "n_samples": int(deg_arr.size),
        "alpha": float(alpha),
        "xmin": int(xmin),
        "KS_statistic": float(D),
        "LR_powerlaw_vs_lognormal": float(R),
        "p_value": float(p),
        "interpretation": "R>0 & p<0.05 → power-law better; R<0 & p<0.05 → lognormal better; else inconclusive"
    }
    plt.figure(figsize=(6,4))
    fit.plot_pdf(label="Empirical")
    fit.power_law.plot_pdf(linestyle="--", label=f"Power-law fit (alpha={alpha:.2f}, xmin={xmin})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figure_powerlaw_pdf.png"), dpi=300, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6,4))
    fit.plot_ccdf(label="Empirical CCDF")
    fit.power_law.plot_ccdf(linestyle="--", label="Power-law fit")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figure_powerlaw_ccdf.png"), dpi=300, bbox_inches="tight"); plt.close()
else:
    if not HAS_POWERLAW:
        print("powerlaw not installed; skip rigorous power-law tests")
    else:
        print("Sample too small for reliable power-law fitting; skip.")

# =========================
# 5. CENTRALIZATION (Gini + Value Concentration)
# =========================

print("\n=== Centralization: value flows (Gini/Top-k) ===")
value_in  = {n: 0.0 for n in G.nodes()}
value_out = {n: 0.0 for n in G.nodes()}
for u, v, data in G.edges(data=True):
    w = float(data.get("weight", 0.0))
    value_out[u] += w
    value_in[v]  += w
value_net = {n: value_in[n] - value_out[n] for n in G.nodes()}

gini_in   = gini(list(value_in.values()))
gini_out  = gini(list(value_out.values()))
gini_net  = gini([abs(x) for x in value_net.values()])
top10_in  = topk_share(list(value_in.values()), 0.10)
top10_out = topk_share(list(value_out.values()), 0.10)
print(f"Gini (inflow):   {gini_in:.3f} | Top10% share: {top10_in:.2%}")
print(f"Gini (outflow):  {gini_out:.3f} | Top10% share: {top10_out:.2%}")
print(f"Gini (|net|):    {gini_net:.3f}")

print("Detecting communities for cross-community flow share...")
try:
    from networkx.algorithms.community import greedy_modularity_communities
    comms = list(greedy_modularity_communities(UG))
    node2comm = {}
    for cid, nodes in enumerate(comms):
        for n in nodes: node2comm[n] = cid
    total_w = 0.0; inter_w = 0.0
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 0.0)); total_w += w
        if node2comm.get(u, -1) != node2comm.get(v, -1):
            inter_w += w
    cross_community_share = (inter_w / total_w) if total_w > 0 else float("nan")
except Exception:
    comms = []; node2comm = {}; cross_community_share = float("nan")
print(f"Cross-community flow share (by weight): {cross_community_share:.2%}")


# =========================
# 6. ANOMALY DETECTION (same as before, features enriched)
# =========================

print("\n=== Anomaly detection (IsolationForest) ===")
nodes_list = list(G.nodes())
X = np.array([
    [
        deg_total.get(n, 0),
        deg_in.get(n, 0),
        deg_out.get(n, 0),
        pagerank.get(n, 0.0),
        core_numbers.get(n, 0),
        hubs.get(n, 0.0),
        authorities.get(n, 0.0),
        betweenness.get(n, 0.0),
        value_in.get(n, 0.0),
        value_out.get(n, 0.0),
        value_net.get(n, 0.0),
    ]
    for n in nodes_list
])
clf = IsolationForest(contamination=0.02, random_state=42)
labels = clf.fit_predict(X)
anomalies = [nodes_list[i] for i, lab in enumerate(labels) if lab == -1]
print(f"Detected {len(anomalies)} anomalous wallets (~2%)")
for a in anomalies[:10]:
    print(" -", a)


# =========================
# 7. TEMPORAL DYNAMICS (original cumulative curves)
# =========================

print("\n=== Temporal dynamics (cumulative overview) ===")
df_all["date"] = df_all["timeStamp"].dt.date
dates_sorted = sorted(df_all["date"].dropna().unique())

daily_tx_count = df_all.groupby("date").size()
daily_unique_addr = df_all.groupby("date")[["from","to"]].nunique().sum(axis=1)

plt.figure(figsize=(10,5))
plt.plot(daily_tx_count.index, daily_tx_count.values, label="Transactions/day")
plt.plot(daily_unique_addr.index, daily_unique_addr.values, label="Unique addresses/day")
plt.legend(); plt.title("Temporal Activity in Sampled Ethereum Subnetwork")
plt.xlabel("Date"); plt.ylabel("Count"); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "figure_temporal_activity.png"), dpi=300, bbox_inches="tight"); plt.close()

gc_ratio_time = []
Gt = nx.DiGraph()
for d in dates_sorted:
    todays_rows = df_all[df_all["date"] == d]
    for _, row in todays_rows.iterrows():
        s, t, v = row["from"], row["to"], row["value"]
        if s == t: continue
        if Gt.has_edge(s, t): Gt[s][t]["weight"] += float(v)
        else:                 Gt.add_edge(s, t, weight=float(v))
    Ug2 = Gt.to_undirected()
    comps2 = list(nx.connected_components(Ug2))
    frac = len(max(comps2, key=len)) / max(1, Gt.number_of_nodes()) if comps2 else 0.0
    gc_ratio_time.append(frac)

plt.figure(figsize=(10,5))
plt.plot(dates_sorted, gc_ratio_time, marker="o", linewidth=1)
plt.title("Giant Component Fraction Over Time")
plt.xlabel("Date"); plt.ylabel("Giant component fraction"); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "figure_giant_component_over_time.png"), dpi=300, bbox_inches="tight"); plt.close()


# =========================
# 7.1 Sliding-window snapshots & dynamic centralities
# =========================

print("\n=== [Temporal] Sliding-window snapshots & dynamic centralities ===")

# Prepare window grid
if len(dates_sorted) > 0:
    start_date = dates_sorted[0]
    end_date = dates_sorted[-1]
    windows = []
    cur = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())
    while cur <= end_dt:
        w_start = cur.date()
        w_end = (cur + timedelta(days=WINDOW_DAYS - 1)).date()
        windows.append((w_start, w_end))
        cur += timedelta(days=STEP_DAYS)
else:
    windows = []

# Containers for time series
records = []          # node, date_end, PR, betweenness, kcore, deg, in_deg, out_deg, tx_count
topk_candidates = set([n for n, _ in top_pr])  # 初始：全局PR前列
snap_idx = 0
for (ws, we) in windows:
    snap_idx += 1
    mask = (df_all["date"] >= ws) & (df_all["date"] <= we)
    df_win = df_all.loc[mask, ["from","to","value"]]
    if df_win.empty:
        continue
    Gw = build_graph_from_edges(df_win)
    UGw = Gw.to_undirected()

    # core / degree / PR / betweenness (sampled)
    deg_w  = dict(Gw.degree())
    in_w   = dict(Gw.in_degree())
    out_w  = dict(Gw.out_degree())
    pr_w   = nx.pagerank(Gw, alpha=0.85) if Gw.number_of_nodes() else {}
    try:
        core_w = nx.core_number(UGw) if UGw.number_of_nodes() else {}
    except Exception:
        core_w = {}
    if Gw.number_of_nodes() <= BETWEENNESS_MAX_NODES:
        try:
            btw_w = nx.betweenness_centrality(
                Gw,
                k=BETWEENNESS_SAMPLE_K,
                normalized=True,
                weight=None,
                endpoints=False,
                seed=0
            )
        except Exception:
            btw_w = {n: float("nan") for n in Gw.nodes()}
    else:
        btw_w = {n: float("nan") for n in Gw.nodes()}

    # collect rows
    for n in Gw.nodes():
        records.append({
            "node": n,
            "window_start": ws,
            "window_end": we,
            "snapshot_index": snap_idx,
            "pagerank": pr_w.get(n, 0.0),
            "betweenness": btw_w.get(n, 0.0),
            "kcore": core_w.get(n, 0),
            "degree": deg_w.get(n, 0),
            "in_degree": in_w.get(n, 0),
            "out_degree": out_w.get(n, 0),
            "tx_count": int(sum(1 for _ in Gw.edges(n)))
        })

    topk_candidates.update([n for n, _ in sorted(pr_w.items(), key=lambda x: x[1], reverse=True)[:TOPK_PLOT]])

# output dynamic centrality timeseries
if records:
    df_dyn = pd.DataFrame(records)
    dyn_csv = os.path.join(TEMP_DIR, "dynamic_centrality_timeseries.csv")
    df_dyn.to_csv(dyn_csv, index=False)

    # choose focus nodes by mean PR
    mean_pr = df_dyn.groupby("node")["pagerank"].mean().sort_values(ascending=False)
    focus_nodes = list(mean_pr.head(TOPK_PLOT).index)

    # draw time series plots
    for metric in ["pagerank", "betweenness", "kcore"]:
        plt.figure(figsize=(11,6))
        for n in focus_nodes:
            sub = df_dyn[df_dyn["node"] == n].sort_values("window_end")
            plt.plot(sub["window_end"], sub[metric], label=n[:8]+"…")
        plt.title(f"Top-{TOPK_PLOT} nodes: {metric} over sliding windows")
        plt.xlabel("Window end date"); plt.ylabel(metric)
        if metric == "pagerank":
            plt.legend(ncol=2, fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(TEMP_DIR, f"timeseries_{metric}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # easy spike detection on PR, save to JSON
    spikes = []
    for n in df_dyn["node"].unique():
        sub = df_dyn[df_dyn["node"] == n].sort_values("snapshot_index")
        x = sub["pagerank"].values
        if len(x) >= 5:
            dif = np.diff(x)
            thr = np.nanmean(dif) + 3*np.nanstd(dif)
            for i, d in enumerate(dif, start=1):
                if d > thr:
                    spikes.append({
                        "node": n, "from_index": int(sub.iloc[i-1]["snapshot_index"]),
                        "to_index": int(sub.iloc[i]["snapshot_index"]),
                        "from_end": str(sub.iloc[i-1]["window_end"]),
                        "to_end": str(sub.iloc[i]["window_end"]),
                        "delta_pr": float(d)
                    })
    if spikes:
        save_json(spikes, os.path.join(TEMP_DIR, "pagerank_spikes.json"))



# =========================
# 7.2 Temporal GNN baseline (GConvGRU)
# =========================

print("\n=== [Temporal] Temporal GNN baseline (optional) ===")
if not HAS_PGT:
    print("PyTorch Geometric Temporal not available; skip temporal GNN demo.")
    print("To enable: pip install torch torch_geometric torch_geometric_temporal (env-specific).")
else:

    if not records:
        print("No temporal snapshots available; skip temporal GNN.")
    else:
        df_dyn = pd.read_csv(os.path.join(TEMP_DIR, "dynamic_centrality_timeseries.csv"))
        df_dyn["node"] = df_dyn["node"].astype(str)

        # sequence construction
        snaps = sorted(df_dyn["snapshot_index"].unique())
        nodes_all = sorted(df_dyn["node"].unique())
        node2idx = {n:i for i,n in enumerate(nodes_all)}
        N = len(nodes_all)

        edge_indices = []
        edge_weights = []   # every snapshot's edge weights
        features = []
        targets  = []

        for (ws, we, idx) in df_dyn.groupby(["window_start","window_end","snapshot_index"]).groups.keys():
            idx = int(idx)
            mask = (df_all["date"] >= pd.to_datetime(ws).date()) & (df_all["date"] <= pd.to_datetime(we).date())
            df_win = df_all.loc[mask, ["from","to","value"]]
            if df_win.empty:
                continue
            Gw = build_graph_from_edges(df_win)

            # --- edge_index (2, E) & edge_weight (E,)
            edges, weights = [], []
            for u, v, data in Gw.edges(data=True):
                if (u in node2idx) and (v in node2idx):
                    edges.append((node2idx[u], node2idx[v]))
                    # use 'weight' attribute as edge weight; default to 1.0
                    weights.append(float(data.get("weight", 1.0)))
            if not edges:
                continue
            edge_index_t = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weight_t = torch.tensor(weights, dtype=torch.float32)  # NEW

            # --- X_t: (N, 4) torch.float32
            sub = df_dyn[df_dyn["snapshot_index"] == idx].set_index("node")
            x = np.zeros((N, 4), dtype=np.float32)
            for n in nodes_all:
                if n in sub.index:
                    x[node2idx[n], 0] = sub.loc[n, "degree"]
                    x[node2idx[n], 1] = sub.loc[n, "in_degree"]
                    x[node2idx[n], 2] = sub.loc[n, "out_degree"]
                    x[node2idx[n], 3] = sub.loc[n, "pagerank"]
            X_t = torch.tensor(x, dtype=torch.float32)

            features.append(X_t)
            edge_indices.append(edge_index_t)
            edge_weights.append(edge_weight_t)  # NEW

        # y_t uses pagerank at t
        # align features[0..T-2] → targets[1..T-1]
        if len(features) < 3:
            print("Too few snapshots for temporal GNN; need >=3.")
        else:
            # --- targets: list of (N,) numpy
            pr_by_snap = []
            for idx in sorted(df_dyn["snapshot_index"].unique()):
                sub = df_dyn[df_dyn["snapshot_index"] == idx].set_index("node")
                y = np.zeros((N,), dtype=np.float32)  # 1D 更稳妥
                for n in nodes_all:
                    if n in sub.index:
                        y[node2idx[n]] = float(sub.loc[n, "pagerank"])
                pr_by_snap.append(y)

            # length alignment
            # --- sequence length T
            min_len = min(len(features), len(pr_by_snap))
            features    = features[:min_len-1]
            edge_indices = edge_indices[:min_len-1]
            edge_weights = edge_weights[:min_len-1]
            targets     = pr_by_snap[1:min_len]

            # === Dataset ready ===
            dataset = DynamicGraphTemporalSignal(
                edge_indices=edge_indices,
                edge_weights=edge_weights,   
                features=features,
                targets=targets              # numpy
            )

            # define simple GConvGRU-based regressor
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
            model = GCRURegressor().to(device)
            optim = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.L1Loss()

            # easy train/val split
            T = len(dataset.features)
            split = int(0.7 * T)

            def step_loop(start, end, train=True):
                model.train(mode=train)
                total = 0.0; count = 0
                for t in range(start, end):
                    x = dataset.features[t].to(device)  # node features
                    ei = dataset.edge_indices[t].to(device)  # edge_index 
                    ew = dataset.edge_weights[t].to(device)  # edge_weight 
                    y = torch.tensor(dataset.targets[t], dtype=torch.float32).to(device)  # targets

                    # forward + loss
                    pred = model(x, ei, ew)  # tensor (N,)
                    loss = loss_fn(pred, y)
                    
                    if train:
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                    
                    total += float(loss.item())
                    count += 1
                
                return total / max(1, count)

            EPOCHS = 20
            train_hist, val_hist = [], []
            for ep in range(1, EPOCHS+1):
                train_loss = step_loop(0, split, train=True)
                val_loss   = step_loop(split, T, train=False)
                train_hist.append(train_loss)
                val_hist.append(val_loss)
                
                if ep % 5 == 0:
                    print(f"[TemporalGNN] Epoch {ep:02d} | train {train_loss:.4f} | val {val_loss:.4f}")

            # save loss curve
            plt.figure(figsize=(7,4))
            plt.plot(train_hist, label="train")
            plt.plot(val_hist, label="val")
            plt.title("Temporal GNN (GConvGRU) L1-loss")
            plt.xlabel("Epoch")
            plt.ylabel("L1")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(TEMP_DIR, "temporal_gnn_loss.png"), dpi=300, bbox_inches="tight")
            plt.close()

            # export the last snapshot prediction
            x = dataset.features[-1].to(device)
            ei = dataset.edge_indices[-1].to(device)
            ew = dataset.edge_weights[-1].to(device)
            y = torch.tensor(dataset.targets[-1], dtype=torch.float32).to(device)

            with torch.no_grad():
                yhat = model(x, ei, ew).cpu().numpy()

            df_pred = pd.DataFrame({
                "node": nodes_all,
                "y_true_pagerank": y.cpu().numpy(),
                "y_pred_pagerank": yhat
            }).sort_values("y_true_pagerank", ascending=False)

            df_pred.to_csv(os.path.join(TEMP_DIR, "temporal_gnn_last_snapshot_prediction.csv"), index=False)




# =========================
# 8. VISUALIZATION SNAPSHOTS (unchanged)
# =========================

print("\n=== Exporting sampled subgraph & hub ego ===")
if G.number_of_nodes() > 200:
    sample_nodes = np.random.choice(list(G.nodes()), 200, replace=False)
else:
    sample_nodes = list(G.nodes())
H_random = G.subgraph(sample_nodes)
pos_rand = nx.spring_layout(H_random, k=0.2, seed=0)
plt.figure(figsize=(8,8))
nx.draw_networkx_nodes(H_random, pos_rand, node_size=40)
nx.draw_networkx_edges(H_random, pos_rand, alpha=0.3, arrows=False)
plt.title("Ethereum Transaction Subgraph (sampled nodes)")
plt.axis("off"); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "figure_subgraph_random.png"), dpi=300, bbox_inches="tight"); plt.close()

hub_node = max(pagerank.items(), key=lambda x: x[1])[0] if pagerank else None
if hub_node is not None:
    ego_nodes = set([hub_node]); ego_nodes.update(G.predecessors(hub_node)); ego_nodes.update(G.successors(hub_node))
    H_ego = G.subgraph(ego_nodes).copy()
    node_sizes = [300 + 2000 * pagerank.get(n, 0) for n in H_ego.nodes()]
    pos_ego = nx.spring_layout(H_ego, k=0.3, seed=1)
    plt.figure(figsize=(8,8))
    nx.draw_networkx_nodes(H_ego, pos_ego, node_size=node_sizes, node_color="lightblue", edgecolors="k", linewidths=0.5)
    nx.draw_networkx_edges(H_ego, pos_ego, alpha=0.3, arrows=True, arrowsize=10, width=0.5)
    plt.title("Ego Network of Top-PageRank Hub")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figure_hub_ego_network.png"), dpi=300, bbox_inches="tight"); plt.close()

# enrich GEXF attributes
for n in G.nodes():
    G.nodes[n]["degree"]      = deg_total.get(n, 0)
    G.nodes[n]["in_degree"]   = deg_in.get(n, 0)
    G.nodes[n]["out_degree"]  = deg_out.get(n, 0)
    G.nodes[n]["pagerank"]    = pagerank.get(n, 0.0)
    G.nodes[n]["core"]        = core_numbers.get(n, 0)
    G.nodes[n]["hub"]         = hubs.get(n, 0.0)
    G.nodes[n]["authority"]   = authorities.get(n, 0.0)
    G.nodes[n]["betweenness"] = betweenness.get(n, 0.0)
    G.nodes[n]["value_in"]    = value_in.get(n, 0.0)
    G.nodes[n]["value_out"]   = value_out.get(n, 0.0)
    G.nodes[n]["value_net"]   = value_net.get(n, 0.0)

gexf_path = os.path.join(OUTDIR, "ethereum_network.gexf")
nx.write_gexf(G, gexf_path)

nodes_metrics = pd.DataFrame({
    "address": list(G.nodes()),
    "degree": [deg_total.get(n, 0) for n in G.nodes()],
    "in_degree": [deg_in.get(n, 0) for n in G.nodes()],
    "out_degree": [deg_out.get(n, 0) for n in G.nodes()],
    "pagerank": [pagerank.get(n, 0.0) for n in G.nodes()],
    "core": [core_numbers.get(n, 0) for n in G.nodes()],
    "hub": [hubs.get(n, 0.0) for n in G.nodes()],
    "authority": [authorities.get(n, 0.0) for n in G.nodes()],
    "betweenness": [betweenness.get(n, 0.0) for n in G.nodes()],
    "value_in": [value_in.get(n, 0.0) for n in G.nodes()],
    "value_out": [value_out.get(n, 0.0) for n in G.nodes()],
    "value_net": [value_net.get(n, 0.0) for n in G.nodes()],
})
nodes_metrics.to_csv(os.path.join(OUTDIR, "nodes_metrics.csv"), index=False)

edges_metrics = pd.DataFrame([
    {"source": u, "target": v, "weight": float(data.get("weight",0.0)), "count": int(data.get("count",0))}
    for u, v, data in G.edges(data=True)
])
edges_metrics.to_csv(os.path.join(OUTDIR, "edges_metrics.csv"), index=False)

summary = {
    "graph": {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": avg_deg,
        "avg_clustering": avg_clust,
        "giant_component_fraction": frac_gc,
    },
    "assortativity": {
        "undirected_degree_pearson": assort_undirected,
        "directed_out_in": assort_out_in,
    },
    "centralization_value": {
        "gini_in": gini_in,
        "gini_out": gini_out,
        "gini_abs_net": gini_net,
        "top10_share_in": top10_in,
        "top10_share_out": top10_out,
        "cross_community_flow_share": cross_community_share,
        "n_communities": len(comms),
    },
    "powerlaw": powerlaw_summary if powerlaw_summary else {"note": "skipped or not available"},
    "temporal": {
        "window_days": WINDOW_DAYS,
        "step_days": STEP_DAYS,
        "timeseries_csv": os.path.join(TEMP_DIR, "dynamic_centrality_timeseries.csv"),
        "topk_plots": {
            "pagerank": os.path.join(TEMP_DIR, "timeseries_pagerank.png"),
            "betweenness": os.path.join(TEMP_DIR, "timeseries_betweenness.png"),
            "kcore": os.path.join(TEMP_DIR, "timeseries_kcore.png"),
        },
        "pr_spikes_json": os.path.join(TEMP_DIR, "pagerank_spikes.json"),
        "temporal_gnn": {
            "enabled": HAS_PGT,
            "loss_curve": os.path.join(TEMP_DIR, "temporal_gnn_loss.png") if HAS_PGT else None,
            "last_snapshot_pred": os.path.join(TEMP_DIR, "temporal_gnn_last_snapshot_prediction.csv") if HAS_PGT else None,
        }
    },
    "top_pagerank": top_pr,
    "anomalies_sample": anomalies[:20],
}
save_json(summary, os.path.join(OUTDIR, "summary.json"))

print("\nAll figures & files saved in:", OUTDIR)
print(" - figure_degree_distribution.png")
print(" - figure_powerlaw_pdf.png / figure_powerlaw_ccdf.png (if generated)")
print(" - figure_temporal_activity.png")
print(" - figure_giant_component_over_time.png")
print(" - figure_subgraph_random.png")
print(" - figure_hub_ego_network.png")
print(" - nodes_metrics.csv / edges_metrics.csv")
print(" - ethereum_network.gexf")
print(" - summary.json")
print("Temporal outputs in:", TEMP_DIR)
print("DONE.")
