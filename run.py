# =====================================================
# Ethereum Transaction Network Analysis (Final Version)
# Complexity Science Final Project
# =====================================================

import requests
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # 非交互式后端：不弹窗，直接保存成 PNG
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from datetime import datetime

# =========================
# 0. CONFIGURATION
# =========================

API_KEY = "GEAB5WHJYW76B5SFUUVGQR2FE85755BYUH"  # <--- 换成你自己的 Etherscan API key

# 高活跃/高可见度的已知钱包地址，交易丰富（公开信息）
seed_addresses_level1 = [
    "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Binance hot wallet
    "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",  # Vitalik Buterin
    "0xE592427A0AEce92De3Edee1F18E0157C05861564",  # Uniswap V3 Router
    "0x564286362092D8e7936f0549571a803B203aAceD",  # Tether Treasury
]

MAX_TX = 3000         # 每个地址最多抓多少笔交易
DO_SECOND_HOP = True  # True = 抓二跳（扩展对手方地址）；False = 只抓上述4个种子


# =========================
# 1. ETHERSCAN DATA FETCH
# =========================

def fetch_transactions_for_address(address, api_key, max_tx=MAX_TX):
    """
    用 Etherscan V2 API 获取某个地址的交易历史（主网）。
    API 形如:
      https://api.etherscan.io/v2/api?chainid=1&module=account&action=txlist&address=...
    返回列: from, to, value(ETH), timeStamp(datetime), hash
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

    resp = requests.get(url)
    data = resp.json()

    # status == "1" 表示成功；否则可能是没交易/速率限制/错误
    if data.get("status") == "1" and "result" in data:
        df = pd.DataFrame(data["result"])
        if df.empty:
            return pd.DataFrame(columns=["from", "to", "value", "timeStamp", "hash"])

        keep_cols = ["from", "to", "value", "timeStamp", "hash"]
        df = df[keep_cols].copy()

        # wei -> ETH
        df["value"] = df["value"].astype(float) / 1e18

        # timestamp -> pandas datetime
        # 先astype(int)可以避免 pandas 的 future warning
        df["timeStamp"] = pd.to_datetime(df["timeStamp"].astype(int), unit="s")

        # 丢掉没有 to 的奇怪 internal 记录
        df = df[(df["from"] != "") & (df["to"] != "")]
        return df

    return pd.DataFrame(columns=["from", "to", "value", "timeStamp", "hash"])


print("=== Fetching seed addresses (1-hop) ===")
level1_list = []
for addr in tqdm(seed_addresses_level1):
    df_addr = fetch_transactions_for_address(addr, API_KEY)
    df_addr["seed_source"] = addr  # 标记来源（谁引进的这个交易）
    level1_list.append(df_addr)

df_level1 = pd.concat(level1_list, ignore_index=True).drop_duplicates(subset=["hash"])


# ---------- 二跳扩展 ----------
if DO_SECOND_HOP:
    print("=== Selecting top neighbor addresses for 2-hop expansion ===")

    # 从一跳交易里收集所有出现过的地址（from / to）
    neighbors_all = pd.concat(
        [df_level1["from"], df_level1["to"]],
        ignore_index=True
    )

    # 出现次数最高的地址往往是“最活跃/最中心的对手方”
    # 我们抓这些地址的交易历史，网络会变得明显更大
    top_neighbors = (
        neighbors_all.value_counts()
        .head(30)          # 取前30个最频繁对手方
        .index
        .tolist()
    )

    level2_list = []
    for addr in tqdm(top_neighbors):
        # 避免重复抓种子地址
        if addr in seed_addresses_level1:
            continue
        df_addr = fetch_transactions_for_address(addr, API_KEY)
        df_addr["seed_source"] = addr
        level2_list.append(df_addr)

    if len(level2_list) > 0:
        df_level2 = pd.concat(level2_list, ignore_index=True).drop_duplicates(subset=["hash"])
    else:
        df_level2 = pd.DataFrame(columns=df_level1.columns)
else:
    df_level2 = pd.DataFrame(columns=df_level1.columns)

# 合并所有抓到的交易
df_all = pd.concat([df_level1, df_level2], ignore_index=True).drop_duplicates(subset=["hash"])

print(f"Total unique transactions collected: {len(df_all)}")

all_nodes = pd.unique(
    pd.concat([df_all["from"], df_all["to"]], ignore_index=True)
)
print(f"Unique addresses involved: {len(all_nodes)}")


# 保存交易数据到 CSV，用时间戳保证不会和打开的旧文件冲突
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_name = f"ethereum_transactions_{ts}.csv"
df_all.to_csv(csv_name, index=False)
print(f"💾 Saved raw data to {csv_name}")


# =========================
# 2. BUILD DIRECTED GRAPH
# =========================

print("\n=== Building transaction graph ===")
G = nx.DiGraph()

# 每条交易变成一条有向边 (from -> to)，边的权重累计转账ETH金额
for _, row in df_all.iterrows():
    s, t, v = row["from"], row["to"], row["value"]
    if s == t:
        continue
    if G.has_edge(s, t):
        G[s][t]["weight"] += v
        G[s][t]["count"] = G[s][t].get("count", 1) + 1
    else:
        G.add_edge(s, t, weight=v, count=1)

print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")


# =========================
# 3. STATIC NETWORK ANALYSIS
# =========================

print("\n=== Computing network metrics ===")
deg_total = dict(G.degree())       # 节点的总度数 (入度+出度)
deg_in    = dict(G.in_degree())    # 入度
deg_out   = dict(G.out_degree())   # 出度

# PageRank: 资金流网络里“重要性高/被指向多/路径中转多”的节点
pagerank = nx.pagerank(G, alpha=0.85)

# 聚类系数：我们把图转成无向后测 clustering
clustering = nx.clustering(G.to_undirected())

avg_deg   = np.mean(list(deg_total.values()))
avg_clust = np.mean(list(clustering.values()))

print(f"Average degree: {avg_deg:.4f}")
print(f"Average clustering coefficient: {avg_clust:.4f}")

# PageRank Top-10 节点（往往是交易所/DeFi路由/稳定币金库等关键角色）
print("\nTop 10 wallets by PageRank:")
top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
for i, (node, val) in enumerate(top_pr, 1):
    print(f"{i}. {node} | PR={val:.6f}")

# 巨型连通分量 (giant component)：把图看成无向图后，最大的连通块占多少比例
UG = G.to_undirected()
components = sorted(nx.connected_components(UG), key=len, reverse=True)
if len(components) > 0:
    giant_nodes = components[0]
    Gc = G.subgraph(giant_nodes)
    frac_gc = len(Gc.nodes()) / G.number_of_nodes()
    print(f"\nGiant component fraction: {frac_gc:.2%}")
else:
    frac_gc = 0.0
    print("\nGiant component fraction: N/A (empty graph?)")


# =========================
# 4. DEGREE DISTRIBUTION (SCALE-FREENESS)
# =========================

print("\n=== Saving degree distribution figure ===")
deg_arr = np.array(list(deg_total.values()))

plt.figure(figsize=(8,5))
plt.hist(deg_arr, bins=100, log=True)
plt.xlabel("Degree k")
plt.ylabel("Frequency (log scale)")
plt.title("Degree Distribution of Ethereum Transaction Network")
plt.tight_layout()
plt.savefig("figure_degree_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# 可选：拟合幂律，显示长尾（中心化的金融中枢）
try:
    import powerlaw
    fit = powerlaw.Fit(deg_arr[deg_arr > 0])
    fit.plot_pdf(label="Empirical")
    fit.power_law.plot_pdf(linestyle="--", label="Power-law fit")
    plt.legend()
    plt.title(f"Power-law exponent α ≈ {fit.alpha:.2f}")
    plt.tight_layout()
    plt.savefig("figure_powerlaw_fit.png", dpi=300, bbox_inches="tight")
    plt.close()
except ImportError:
    print("powerlaw not installed; skip power-law plot")


# =========================
# 5. CENTRALIZATION (GINI)
# =========================

print("\n=== Measuring centralization (Gini) ===")
def gini(x):
    x = np.sort(np.array(x))
    n = len(x)
    if n == 0:
        return 0.0
    cumx = np.cumsum(x)
    return (n + 1 - 2*np.sum(cumx)/cumx[-1]) / n

gini_coeff = gini(list(deg_total.values()))
print(f"Gini coefficient of node degree: {gini_coeff:.3f}")
# Gini 越高 -> 连接度越集中在极少数节点手中 -> 资金流越“中心化”


# =========================
# 6. ANOMALY DETECTION
# =========================

print("\n=== Anomaly detection (IsolationForest) ===")
nodes_list = list(G.nodes())
X = np.array([
    [deg_total[n], deg_in[n], deg_out[n], pagerank[n]]
    for n in nodes_list
])

clf = IsolationForest(contamination=0.02, random_state=42)
labels = clf.fit_predict(X)

# IsolationForest 输出：-1 = 异常，1 = 正常
anomalies = [nodes_list[i] for i, lab in enumerate(labels) if lab == -1]

print(f"Detected {len(anomalies)} anomalous wallets (~2%)")
print("Top suspicious / high-impact wallets:")
for a in anomalies[:10]:
    print(" -", a)

# 这些往往是：
# - 资金中转/洗钱枢纽
# - 大型分发合约/空投/ICO合约
# - 交易所清算/热钱包
# - 攻击/诈骗集资地址 (短时间收到大量小额转账)


# =========================
# 7. TEMPORAL DYNAMICS
# =========================

print("\n=== Temporal dynamics ===")

# 把交易按天聚合，研究活跃度和连通性的演化
df_all["date"] = df_all["timeStamp"].dt.date
dates_sorted = sorted(df_all["date"].unique())

# (1) 每天的交易数 & 每天参与过交易的唯一地址数
daily_tx_count = df_all.groupby("date").size()
daily_unique_addr = (
    df_all.groupby("date")[["from", "to"]]
    .nunique()
    .sum(axis=1)
)

plt.figure(figsize=(10,5))
plt.plot(daily_tx_count.index, daily_tx_count.values, label="Transactions/day")
plt.plot(daily_unique_addr.index, daily_unique_addr.values, label="Unique addresses/day")
plt.legend()
plt.title("Temporal Activity in Sampled Ethereum Subnetwork")
plt.xlabel("Date")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figure_temporal_activity.png", dpi=300, bbox_inches="tight")
plt.close()

# (2) Giant component fraction over time
# 思路：随着时间往前推进，把每天新增的交易不断加进一张"累积图"
# 然后测这张图的 giant component 占比 (网络凝聚程度)
gc_ratio_time = []
Gt = nx.DiGraph()

for d in dates_sorted:
    todays_rows = df_all[df_all["date"] == d]
    for _, row in todays_rows.iterrows():
        s, t, v = row["from"], row["to"], row["value"]
        if s == t:
            continue
        if Gt.has_edge(s, t):
            Gt[s][t]["weight"] += v
        else:
            Gt.add_edge(s, t, weight=v)

    Ug2 = Gt.to_undirected()
    comps2 = list(nx.connected_components(Ug2))
    if len(comps2) > 0:
        biggest = max(comps2, key=len)
        frac = len(biggest) / max(1, Gt.number_of_nodes())
    else:
        frac = 0.0
    gc_ratio_time.append(frac)

plt.figure(figsize=(10,5))
plt.plot(dates_sorted, gc_ratio_time, marker="o", linewidth=1)
plt.title("Giant Component Fraction Over Time")
plt.xlabel("Date")
plt.ylabel("Giant component fraction")
plt.tight_layout()
plt.savefig("figure_giant_component_over_time.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# 8. VISUALIZATION SNAPSHOTS
# =========================

# 8a. 随机采样子图 (可能很稀疏，作为“随机局部视角”)
print("\n=== Exporting random subgraph (200 nodes) ===")

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
plt.axis("off")
plt.tight_layout()
plt.savefig("figure_subgraph_random.png", dpi=300, bbox_inches="tight")
plt.close()

# 注意：如果网络是高度hub-and-spoke型，随机抽样经常只抽到“叶子节点”且几乎不互相连接，
# 所以这张图很可能只是一圈点。这本身就是一个现象：大多数地址几乎只连向中心化hub。


# 8b. 以 PageRank 最高的地址为中心，画它的1跳ego网络
print("=== Exporting ego network of top hub ===")
hub_node = max(pagerank.items(), key=lambda x: x[1])[0]
print(f"Hub node for ego visualization: {hub_node}")

ego_nodes = set([hub_node])
ego_nodes.update(G.predecessors(hub_node))
ego_nodes.update(G.successors(hub_node))

H_ego = G.subgraph(ego_nodes).copy()

# 节点大小用 PageRank 体现“重要性”
node_sizes = [300 + 2000 * pagerank.get(n, 0) for n in H_ego.nodes()]

pos_ego = nx.spring_layout(H_ego, k=0.3, seed=1)
plt.figure(figsize=(8,8))
nx.draw_networkx_nodes(
    H_ego,
    pos_ego,
    node_size=node_sizes,
    node_color="lightblue",
    edgecolors="k",
    linewidths=0.5,
)
nx.draw_networkx_edges(
    H_ego,
    pos_ego,
    alpha=0.3,
    arrows=True,
    arrowsize=10,
    width=0.5,
)
plt.title("Ego Network of Top-PageRank Hub")
plt.axis("off")
plt.tight_layout()
plt.savefig("figure_hub_ego_network.png", dpi=300, bbox_inches="tight")
plt.close()

# 解释：这张hub ego图就是“中心化资金中枢 + 它的所有一跳连接”。
# 这通常能画出‘太阳放射状/蜘蛛’结构，是展示去中心化系统里“中心化流动枢纽”的最好证据。


# 8c. 导出全图给 Gephi 做高质量可视化（力导布局、着色、标签等）
nx.write_gexf(G, "ethereum_network.gexf")


# =========================
# 9. DONE
# =========================

print("\nAll figures saved:")
print(" - figure_degree_distribution.png")
print(" - figure_powerlaw_fit.png (if generated)")
print(" - figure_temporal_activity.png")
print(" - figure_giant_component_over_time.png")
print(" - figure_subgraph_random.png")
print(" - figure_hub_ego_network.png")
print(f"Raw tx data saved to {csv_name}")
print("Full graph exported to ethereum_network.gexf")
print("DONE.")
