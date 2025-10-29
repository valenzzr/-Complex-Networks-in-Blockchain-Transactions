# -Complex-Networks-in-Blockchain-Transactions
> *Author: Zhang Zhirui*
>
> *Description: PC5253 final project *
>
> 内容包括：
>
> 项目简介（你们在做什么，研究问题是什么）
>
> 环境配置（Python 版本、依赖、虚拟环境）
>
> Etherscan API_KEY 配置方法
>
> 脚本怎么跑、会输出什么文件
>
> 每个输出文件的意义（CSV / PNG / GEXF）
>
> 代码结构分章节解释（数据获取 / 建图 / 指标分析 / 时间演化 / 异常检测 / 可视化）

------

# Ethereum Transaction Network Analysis

Final Project – Complexity Science

## 1. 项目简介

本项目的目标是用**复杂网络的视角**来研究以太坊主网上的真实资金流动结构。
 我们想回答的问题包括：

- 以太坊的资金流网络是不是“去中心化”的？还是由少数超级节点控制？
- 资金流网络的结构是否呈现典型的复杂网络特征（例如：幂律度分布、巨型连通分量、中心化骨干）？
- 这些关键节点是谁（交易所钱包？DeFi 路由合约？创始人地址？），它们在网络中扮演什么角色？
- 网络连通性如何随时间演化？是不是越来越集中到同一批“金融枢纽”中？

为此我们做了几件事：

1. 从 Etherscan v2 API 抓取主网真实交易数据（Ethereum mainnet）。
2. 构建一个“以太坊交易网络”（节点=地址，边=资金转移）。
3. 分析网络的拓扑性质（平均度、聚类系数、Gini 系数、PageRank、巨型连通分量等）。
4. 用 IsolationForest 检测“异常”/“高影响力”地址。
5. 研究网络随时间的演化（网络凝聚度如何增长，活动是否呈爆发式）。
6. 输出可视化图，包括一个可以在 Gephi 中进一步渲染的 `.gexf` 网络文件，用于展示资金流的中心化骨干。

> 这个项目直接对应课程大作业要求里的：
>
> - "complex transaction networks where nodes can be wallet addresses and directed edges represent fund transfers"
> - "investigate emergent features such as centralization or anomaly detection"
> - "examine network properties temporal dynamics"

------

## 2. 环境配置

建议使用 Conda 或 venv 创建独立虚拟环境，避免污染系统 Python。

### 2.1 Python 版本

推荐使用 **Python 3.9+**。
 我们在运行时使用了如下主要依赖：

- `requests` （访问 Etherscan API）
- `pandas` （清洗和操作交易表格）
- `numpy`
- `networkx` （构建和分析交易网络）
- `matplotlib` （画图）
- `scikit-learn`（IsolationForest 异常检测）
- `tqdm`（进度条显示）
- `powerlaw`（可选，用于拟合幂律分布。如果没安装会自动跳过）
- （可选）`gephi` 不是 Python 包，是一个桌面应用，我们导出的 `.gexf` 文件会在 Gephi 里打开

### 2.2 创建虚拟环境示例（Conda）

```bash
conda create -n ethnet python=3.10 -y
conda activate ethnet
pip install requests pandas numpy networkx matplotlib scikit-learn tqdm powerlaw
```

如果安装 `powerlaw` 出错，可以暂时不装，它只影响幂律拟合那一步；脚本会自动忽略。

### 2.3 Windows 上注意

- 如果你之前打开过 `ethereum_transactions_....csv` 文件（比如用 Excel 打开），再次运行脚本可能会在保存 CSV 时报 “PermissionError: file in use”。
   解决办法：
  - 关掉那个 CSV，
  - 或者我们现在的脚本已经用时间戳生成新文件名（`ethereum_transactions_2025xxxx_xxxxxx.csv`），所以基本不会再撞同名。

------

## 3. 准备 Etherscan API Key

我们使用的是 **Etherscan v2 API**。

### 步骤：

1. 去 [https://etherscan.io](https://etherscan.io/) 注册账号并登录。
2. 在你的账户里创建一个 API Key。
3. 复制这个 Key（看起来会是类似 `ABCD1234...` 的一段字符串）。

### 配置到脚本里

在脚本开头有一段：

```python
API_KEY = "YOUR_API_KEY_HERE"
```

请把 `"YOUR_API_KEY_HERE"` 替换成你自己的真实 key，例如：

```python
API_KEY = "GEAB5WHJYW76B5SFUUVGQR2FE85755BYUH"
```

⚠️ 注意：

- 如果 API_KEY 没改，Etherscan 返回的 `status` 不是 `"1"`，脚本会抓不到任何交易，整张图就会是空的。
- 免费 key 有速率限制（大约 5 次请求/秒 + 每日请求总量限制）。我们的脚本是串行请求几十个地址，正常不会超标。如果后续扩展抓更多地址，可以在循环里加 `time.sleep(0.25)` 来限速。

------

## 4. 运行脚本

假设脚本文件叫 `final_run.py`，在虚拟环境里运行：

```bash
python final_run.py
```

脚本会依次完成：

1. 抓链上交易数据（Etherscan API）
2. 合并和清洗
3. 构建交易网络（NetworkX）
4. 计算网络指标与中心化指标
5. 时间演化分析
6. 异常检测（IsolationForest）
7. 画图并保存
8. 导出给 Gephi 的 `.gexf` 文件

------

## 5. 输出文件说明

运行结束后会在当前目录生成这些文件：

### 5.1 交易数据（CSV）

```
ethereum_transactions_YYYYMMDD_HHMMSS.csv
```

- 这是原始的交易明细（合并后去重）。
- 列包括：
  - `from`：付款地址
  - `to`：收款地址
  - `value`：交易金额（ETH, 已经从 wei 转换过）
  - `timeStamp`：转账时间（UTC）
  - `hash`：交易哈希（唯一 ID）
  - `seed_source`：我们最初是从哪个“种子地址/邻居地址”抓到它的

这是你“数据来源透明性”的证据，报告里可以附上。

------

### 5.2 分析图（PNG）

脚本会生成多张图，常见包括：

- `figure_degree_distribution.png`
   节点度分布（log 纵轴）。显示是不是 heavy-tailed / scale-free 风格。
- `figure_powerlaw_fit.png`（如果安装了 powerlaw）
   用 powerlaw 包拟合度分布，给出幂律指数 α。可以用来支持“长尾分布，说明有极少数超级枢纽”。
- `figure_temporal_activity.png`
   纵轴：每天交易数 / 每天活跃唯一地址数。
   可以观察链上活动是高爆发的，而不是均匀的（典型 ICO、空投、清算日）。
- `figure_giant_component_over_time.png`
   x 轴是时间，y 轴是 giant component fraction（巨型连通分量占全图节点的比例）。
   解释“以太坊资金流随着时间变得越来越凝聚，几乎所有地址都被同一批中心化枢纽连到一起”。
- `figure_subgraph_random.png`
   我们随机抽 ~200 个节点画出来的小子图。
   通常会发现这些节点几乎彼此不连，体现“绝大多数地址只是叶子”，并不是互相之间转来转去。
- `figure_hub_ego_network.png`
   把 PageRank 最高的那个超级枢纽节点（通常是交易所热钱包/路由合约）作为中心，画它和所有一跳邻居的 ego network。
   这图往往是一颗“放射状大太阳”：中心是巨型hub，周围是一圈一圈向它连的地址。
   这个图极其适合放在 PPT 里当“中心化证据”。

这些图基本就是你的“结果章节”。

------

### 5.3 网络结构（GEXF）

```
ethereum_network.gexf
```

- 这是完整的交易网络（节点=地址，边=资金流），以 GEXF (Graph Exchange XML Format) 格式保存。
- 你可以用 **Gephi** 打开它，然后：
  - 使用 ForceAtlas2 布局
  - 根据 PageRank 给节点设大小
  - 根据社群划分(Modularity)或 in-degree/out-degree 给节点上色
  - 调整边透明度
  - 导出高分辨率可视化图（PNG / SVG / PDF）

这张 Gephi 图是展示“资金流骨干网络”的最直观方式，适合放报告封面或者最后一页。

------

## 6. 脚本主要逻辑解释（逐段解析）

下面对应 `final_run.py` 的核心段落。

### 6.1 配置区

```python
API_KEY = "YOUR_API_KEY_HERE"

seed_addresses_level1 = [
    "0x742d35...f44e",  # Binance hot wallet
    "0xd8dA6B...6045",  # Vitalik
    "0xE59242...1564",  # Uniswap V3 Router
    "0x564286...AceD",  # Tether Treasury
]

MAX_TX = 3000
DO_SECOND_HOP = True
```

- 这些 `seed_addresses_level1` 是公开已知的高影响力地址（交易所热钱包、DeFi 路由、稳定币金库等）。
   之所以用这种地址当“起点”，是因为它们涉及大量真实交易，我们可以获得一个有代表性的子图。
- `DO_SECOND_HOP = True` 表示我们不仅抓这 4 个地址的交易，还会抓它们交易对手中最活跃的一批地址（top 30），进一步扩大网络规模。这就是“2-hop 采样”。

### 6.2 抓交易数据

```python
def fetch_transactions_for_address(address, api_key, max_tx=MAX_TX):
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

    # 解析返回结果
    # status == "1" 表示成功
    # result 是交易数组，每一行是一次转账
    ...
```

这里做了几步清洗：

- 把 `value` 从 wei 转成 ETH；
- 把 `timeStamp` 转成可读的 `datetime`;
- 丢弃 `from` 或 `to` 为空的交易（有些内部合约调用没有正常 to 地址）；
- 给每笔交易打上 `seed_source`，记录这笔交易是从哪条“扩展边”抓到的，方便追踪来源。

### 6.3 一跳 & 二跳

1. 先抓种子地址本身（1-hop）。
2. 然后看这些交易里所有出现过的地址（from/to），统计谁出现频率最高。
3. 取出现频率最高的 Top 30 作为“二跳邻居”，继续抓他们的交易。

```python
neighbors_all = pd.concat([df_level1["from"], df_level1["to"]], ignore_index=True)
top_neighbors = (
    neighbors_all.value_counts()
    .head(30)
    .index
    .tolist()
)
```

这相当于构建了一个“局部以太坊经济圈”图：
 不是整条链，而是围绕几个关键流动性枢纽的巨大子网络。

### 6.4 构建交易网络 (NetworkX)

```python
G = nx.DiGraph()
for _, row in df_all.iterrows():
    s, t, v = row["from"], row["to"], row["value"]
    if s == t:
        continue
    if G.has_edge(s, t):
        G[s][t]["weight"] += v
        G[s][t]["count"] = G[s][t].get("count", 1) + 1
    else:
        G.add_edge(s, t, weight=v, count=1)
```

- 我们把每一笔交易看成一条有向边 `from -> to`。
- 边的 `weight` 是累计转账金额（ETH）。
- `count` 是交易次数。
- 这相当于是资金流网络（谁给了谁多少钱）。

NetworkX `DiGraph` 让我们之后能算 PageRank、in_degree、out_degree 等网络指标。

### 6.5 静态网络分析

```python
deg_total = dict(G.degree())
pagerank  = nx.pagerank(G, alpha=0.85)
clustering = nx.clustering(G.to_undirected())
```

我们测：

- 平均度（平均每个地址跟多少个地址有资金往来）
- 平均聚类系数（是不是“朋友的朋友也互相转钱”？）
- PageRank（谁是资金流里最关键的“枢纽节点”）
- Giant component fraction（几乎所有地址是否属于同一个资金流大生态）

这些是经典复杂网络指标，能支持“结构是否中心化”“系统是否被少数节点绑定在一起”等论点。

### 6.6 中心化度量（Gini 系数）

```python
def gini(x):
    ...
gini_coeff = gini(list(deg_total.values()))
```

我们把“谁连接了多少人”当成一种“资源/权力”，然后用 Gini 系数衡量它有多不平等。

Gini ~0.51 非常高，表示极强不平等：

> 绝大多数地址没什么连接度，资金全在少数超级节点之间流动。

这是你们在报告里可以当作“定量证明中心化”的关键数字。

### 6.7 异常检测（IsolationForest）

```python
X = np.array([
    [deg_total[n], deg_in[n], deg_out[n], pagerank[n]]
    for n in nodes_list
])
clf = IsolationForest(contamination=0.02, random_state=42)
labels = clf.fit_predict(X)
```

我们对每个地址生成 4 个特征：

- 总度（它到底接触了多少不同地址）
- 入度、出度（是一直在收钱，还是一直在打钱）
- PageRank（它是否是路径中转的关键）

IsolationForest 会把“过于极端”的点标成异常（-1）。

这些异常节点往往就是：

- 交易所钱包（同时跟成千上万地址打钱/收钱）
- DeFi 路由合约
- 巨额分发/募资合约
- 诈骗/钓鱼/集资类地址
- 甚至是创始人级别钱包

我们不是在做合规审查；我们是在说明：**极端结构位置 = 系统性重要性**。

### 6.8 时间演化分析

我们按照日期滚动构建子图，记录巨型连通分量（giant component）占整个图的比例随时间怎么变。

直觉上：

- 如果早期网络是分散的（很多孤立团），巨型分量占比会比较低。
- 随着时间推移，大家都开始把钱通过同一批枢纽流动，整个网络“连成一体”，巨型分量占比就逼近 1。

对应的图 `figure_giant_component_over_time.png` 就是你们的“系统逐渐集中化”证据图。

### 6.9 可视化输出

我们做了两个特别有解释力的视角：

1. `figure_subgraph_random.png`:
   - 随机抽一批节点画出来的子图。通常是一圈一圈的孤立点/小星星，很稀疏。
   - 说明“典型的普通地址”并没有互相强连接。
2. `figure_hub_ego_network.png`:
   - 取 PageRank 最高的那个地址（比如交易所热钱包），画它和所有一跳邻居。
   - 这张图通常长得像一颗巨大的太阳/刺猬：一个超巨大中心 + 大量射线。
   - 这是最有冲击力的“中心化骨干”图，非常适合 PPT。

同时，我们还导出了：

```python
nx.write_gexf(G, "ethereum_network.gexf")
```

方便用 Gephi 生成最终极高清的网络骨架图（ForceAtlas2 + Node size = PageRank + Community colors）。

------

## 7. 在报告/答辩时可以直接写的总结

> 我们抓取了约 55k 条真实 Ethereum 主网交易，构建了一个 ~21k 地址、~22k 边的资金流网络。
>  我们发现该网络的平均度仅约 2，但 PageRank 和度分布极度偏斜，Gini 系数约 0.51，巨型连通分量覆盖率几乎 100%。
>  这意味着：虽然以太坊在协议层是去中心化的，但实际的资金流高度依赖少数超级枢纽（交易所钱包、路由合约、稳定币金库）。
>  时间分析进一步显示，这种“单一骨干网络”是逐渐形成的 —— 活动在关键历史时期出现极端爆发，随后所有新地址被吸入同一个全球性巨型连通块。
>  使用 IsolationForest 的结构异常检测，我们还能自动识别这些系统性关键节点。

这段基本表示：我们不仅跑了数据，还做了复杂网络分析 + 时间演化 + 异常检测 + 可视化，而且会解释结果。

