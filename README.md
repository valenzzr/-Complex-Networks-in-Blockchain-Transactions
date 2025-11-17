# Ethereum Transaction Network as a Complex System

Final Project – *GROUP 5* 
Topic: **Complex Networks in Blockchain Transactions**

---

## 1. Project Overview

This project studies **Ethereum transaction networks** as a real-world example of a complex system.

We focus on a **local subnetwork** around several well-known wallets (exchanges, protocol contracts, Vitalik, etc.), and analyze:

- Static network structure (degree distributions, clustering, assortativity, cores, communities)
- **Centralization** of value flows (Gini coefficients, top-k concentration, cross-community flows)
- **Temporal dynamics** of connectivity and node centrality
- **Anomaly detection** on wallets using multi-dimensional network features
- An optional **temporal GNN baseline** (GConvGRU) to forecast future PageRank

All analysis is implemented in a single script:

```bash
ethereum_network_analysis.py
```

Running it fetches transactions from Etherscan, builds graphs, computes metrics, and saves all outputs into `outputs/` and `outputs/temporal/`.

------

## 2. Data Source & Sampling Strategy

### 2.1 Data source

We use the [Etherscan v2 API](https://etherscan.io/apidashboard) on Ethereum mainnet:

- Endpoint: `module=account&action=txlist&chainid=1`
- For each address, we retrieve up to `MAX_TX` normal transactions (default: 3,000).
- We keep:
  - `from`, `to` – wallet addresses
  - `value` – in ETH (converted from wei)
  - `timeStamp` – converted to Python `datetime`
  - `hash` – transaction hash
  - plus `contractAddress` to correctly handle contract-creation transactions

Data cleaning:

- Convert `value` to ETH and `timeStamp` to `datetime`.
- Strip and normalize address strings.
- Replace `to == ""` with `contractAddress` for contract-creation txs.
- Drop rows with invalid timestamps or missing addresses.
- Keep only valid Ethereum addresses matching `^0x[0-9a-fA-F]{40}$`.

A full raw export is saved to:

```text
outputs/ethereum_transactions_YYYYMMDD_HHMMSS.csv
```

(Example: `outputs/ethereum_transactions_20251102_223921.csv`.)

### 2.2 Seed selection & 2-hop expansion

We construct a **local Ethereum subnetwork** around several high-visibility addresses:

- Binance hot wallet
- Vitalik Buterin
- Uniswap V3 router
- Tether Treasury

(Exact addresses are listed in `seed_addresses_level1` in the code.)

Sampling strategy:

1. **1-hop expansion**
   - For each seed address, fetch up to `MAX_TX` transactions.
   - Concatenate the results and **deduplicate by transaction hash**.
2. **2-hop expansion** (optional, controlled by `DO_SECOND_HOP`)
   - From the level-1 data, count how often each address appears as a counterparty.
   - Select the **top 200 counterparties** by occurrence frequency.
   - Fetch transactions for each of these neighbors (again up to `MAX_TX`).
   - Merge everything and deduplicate by transaction hash.
3. Final dataset: `df_all`
   - All unique transactions involving the seeds and top neighbors after cleaning.

On one representative run (2025-11-02), we obtain:

- **Total unique transactions collected:** **250,876**
- **Unique addresses involved (nodes):** **61,134**
- **Directed edges in the aggregated graph:** **72,351**

------

## 3. Network Construction

From the cleaned transaction dataframe `df_all`, we build a **directed weighted transaction graph** `G`:

- **Nodes:** all unique wallet addresses (both `from` and `to`)
- **Directed edges:** `from → to`
- **Edge attributes:**
  - `weight`: total ETH value sent along that edge (aggregated over all transactions)
  - `count`: number of transactions between a given pair of addresses

We also construct an **undirected graph** `UG = G.to_undirected()` for metrics that assume undirected edges (e.g., clustering, k-core, community detection).

On the representative run:

- `G.number_of_nodes() = 61,134`
- `G.number_of_edges() = 72,351`
- The largest connected component in `UG` contains essentially **100% of nodes**, i.e. the sampled subnetwork is fully connected.

------

## 4. Static Network Analysis

### 4.1 Basic structural metrics

On the directed graph `G` and its undirected projection `UG`, we compute:

- **Degree statistics**
  - Total degree, in-degree, and out-degree for each node
  - **Average degree:** ≈ **2.37**
- **Clustering coefficient** (on `UG`)
  - **Average clustering coefficient:** ≈ **0.0113**
     → very low triangle density, as expected for a sparse financial network.
- **Giant component**
  - Fraction of nodes in the giant component: ≈ **100%**

These values suggest a **large, sparse, but well-connected** transaction network.

### 4.2 Assortativity

We measure degree assortativity:

- **Undirected degree assortativity (Pearson):** ≈ **−0.52**
- **Directed out→in assortativity:** ≈ **−0.44**

Both are strongly **negative (disassortative)**:

- High-degree hubs tend to connect to many low-degree nodes.
- High out-degree “senders” are more likely to send to addresses with lower in-degree than themselves.

This pattern is typical of infrastructure networks where a small number of large entities (exchanges, routers) interact with a broad set of smaller wallets.

### 4.3 Extended structural metrics

We further compute:

- **k-core decomposition** (`nx.core_number` on `UG`)
   → identifies deeply embedded “core” addresses.
- **Rich-club coefficient** (for high-degree nodes), when computation is numerically stable.
- **HITS scores** (`hubs`, `authorities`) on `G`
  - Highlighting nodes that act as broadcasters vs. information sinks.
- **Betweenness centrality**
  - Computed using exact or sampled algorithms depending on graph size.
  - For this run:
    - `N = 61,134`, `M = 72,351`, sampled with `k = 1,835`.

### 4.4 Top PageRank wallets

We compute **PageRank** on `G` (`alpha = 0.85`). The top-10 highest PageRank wallets in our subnetwork are:

1. `0x876eabf441b2ee5b5b0554fd502a8e0600950cfa` (PR ≈ 0.0609)
2. `0x742d35cc6634c0532925a3b844bc454e4438f44e` (Binance hot wallet, PR ≈ 0.0369)
3. `0xdf6c10f310ef0402a5d4a35b85905cb09ae80994` (PR ≈ 0.0219)
4. `0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45` (PR ≈ 0.0188)
5. `0xff1f2b4adb9df6fc8eafecdcbf96a2b351680455` (PR ≈ 0.0187)
6. `0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b` (PR ≈ 0.0177)
7. `0x32be343b94f860124dc4fee278fdcbd38c102d88` (PR ≈ 0.0142)
8. `0xe592427a0aece92de3edee1f18e0157c05861564` (Uniswap V3 router, PR ≈ 0.0122)
9. `0xd8da6bf26964af9d7eed9e03e53415d37aa96045` (Vitalik Buterin, PR ≈ 0.0114)
10. `0xab5801a7d398351b8be11c439e05c5b3259aec9b` (PR ≈ 0.0104)

Many of these are known exchange / protocol / public figure wallets, which confirms that **network centrality is dominated by a small set of infrastructure nodes**.

------

## 5. Degree Distribution & Power-Law Testing

We analyze the **degree distribution** of `G`:

1. Extract total degree values `k > 0` for all nodes.
2. Plot a log-scaled histogram:
   - `outputs/figure_degree_distribution.png`
3. If the [`powerlaw`](https://github.com/jeffalstott/powerlaw) Python package is installed and the sample size is sufficient, we:
   - Fit a **discrete power-law** to the degree sequence.
   - Estimate the exponent `α`, lower cutoff `xmin`, and KS statistic.
   - Compare **power-law vs lognormal** using the likelihood ratio `R` and p-value.

Additional figures (if computed):

- `outputs/figure_powerlaw_pdf.png` – empirical PDF and fitted power-law.
- `outputs/figure_powerlaw_ccdf.png` – empirical CCDF and fitted power-law.

Numerical fitting summary (α, xmin, KS, R, p) is stored within `summary.json` under `"powerlaw"`.

------

## 6. Centralization of Value Flows

We quantify how **financial flows are concentrated** across nodes.

For each node `n`:

- `value_in[n]` – total ETH received (sum of weights of incoming edges)
- `value_out[n]` – total ETH sent (sum of weights of outgoing edges)
- `value_net[n] = value_in[n] − value_out[n]`

We compute:

- **Gini coefficients:**
  - Gini of inflows
  - Gini of outflows
  - Gini of |net flow|
- **Top-10% concentration:**
  - Share of total inflows received by the top 10% of nodes
  - Share of total outflows sent by the top 10% of nodes

For our run, we obtain:

- **Gini(inflow) ≈ 1.000**, with the **top 10% of addresses receiving ~100% of total inflow**.
- **Gini(outflow) ≈ 0.999**, with the **top 10% of addresses responsible for ~99.98% of total outflow**.
- **Gini(|net flow|) ≈ 0.998**, indicating that net liquidity provision/absorption is also extremely concentrated.

This confirms **extreme centralization** of economic activity in our Ethereum subnetwork.

### 6.1 Cross-community flow

To understand how value moves **between communities**, we:

1. Run **greedy modularity community detection** on the undirected graph `UG`.
2. Assign each node a community ID.
3. For each edge in `G`, accumulate:
   - `total_w` – total ETH transferred over all edges.
   - `inter_w` – ETH transferred between **different** communities.

The **cross-community flow share** is:

```text
cross_community_share = inter_w / total_w
```

On our run:

- **Cross-community flow share ≈ 50.76%**

This suggests that hubs not only concentrate value, but also act as bridges between modular clusters, with roughly half of the total value flowing across community boundaries.

------

## 7. Anomaly Detection on Wallets

We apply an **Isolation Forest** to perform unsupervised anomaly detection on nodes, using a feature vector:

- Degree, in-degree, out-degree
- PageRank
- k-core index
- HITS hub & authority scores
- Betweenness centrality
- Value statistics:
  - total inflow, total outflow, net flow

We set `contamination = 0.02`, so approximately 2% of nodes are flagged as anomalies.

On our run:

- **Detected anomalous wallets:** **1,165 (~2% of all nodes)**

The flagged set includes:

- Highly central infrastructure addresses (e.g. `0x742d35...`, `0x876eab...`), which are **outliers in degree and flow volume**.
- A number of lesser-known addresses (e.g. `0xcbeaec6994...`, `0xf7920b0768...`, `0xbf2179859f...`, `0x86fa049857...`, `0x3893b9422c...`), which may correspond to token contracts, aggregators, or unusual transaction patterns.

**Important:** Here, “anomalous” means **network-structural outlier**, not necessarily malicious activity. The method highlights nodes whose joint features (centralities + value flows) deviate strongly from the majority.

A sample of anomalies is also stored in `summary.json` under `"anomalies_sample"`.

------

## 8. Temporal Dynamics

### 8.1 Daily activity & giant component growth

From `df_all`, we extract a `date` column from `timeStamp` and compute:

- Number of **transactions per day**
- Number of **unique addresses per day** (based on `from` and `to`)

We also build a cumulative graph over time:

1. Initialize an empty directed graph `Gt`.
2. For each day in chronological order:
   - Add all edges from that day to `Gt`.
   - Convert to undirected `Ug2`.
   - Compute the fraction of nodes in the **giant component**.

Outputs:

- `outputs/figure_temporal_activity.png` – daily transactions and unique addresses.
- `outputs/figure_giant_component_over_time.png` – giant component fraction over time.

This shows how the sampled subnetwork becomes connected and how activity fluctuates over the observation window.

### 8.2 Sliding-window snapshots & dynamic centralities

We perform **sliding-window analysis** to study time-varying centralities.

Parameters:

- Window length: `WINDOW_DAYS` (default: 30 days)
- Step size: `STEP_DAYS` (default: 7 days)

For each time window `[w_start, w_end]`:

1. Filter transactions whose dates fall within the window.
2. Build a subgraph `Gw` and its undirected version `UGw`.
3. Compute **window-level metrics**:
   - Degree, in-degree, out-degree per node
   - PageRank
   - k-core index
   - Betweenness centrality (sampled on larger graphs)

We record all metrics for all nodes across windows, and save:

```text
outputs/temporal/dynamic_centrality_timeseries.csv
```

We then:

- Compute average PageRank across windows for each node.
- Select **Top-K nodes** (default: 10) by mean PageRank.
- Plot time series for:
  - PageRank
  - Betweenness
  - k-core index

Figures:

- `outputs/temporal/timeseries_pagerank.png`
- `outputs/temporal/timeseries_betweenness.png`
- `outputs/temporal/timeseries_kcore.png`

** You can see how sliding-window's results on large scale data by running the script `sliding.py` **

### 8.3 PageRank spike detection

For each node, we examine its PageRank time series across windows:

- Compute differences ΔPR between consecutive snapshots.

- Define a spike when:

  ```text
  ΔPR > mean(ΔPR) + 3 * std(ΔPR)
  ```

- Record spikes (with window indices and dates) to:

  ```text
  outputs/temporal/pagerank_spikes.json
  ```

These spikes correspond to **sudden jumps in centrality**, which may indicate large transfers, token launches, or other structural events.

------

## 9. Temporal GNN Baseline (GConvGRU, Optional)

If **PyTorch Geometric Temporal** is available, we build a simple **GConvGRU-based model** to forecast future PageRank.

### 9.1 Sequence construction

For each sliding window:

- **Graph structure**
  - `edge_index_t`: list of directed edges (source, target) as integer node indices.
  - `edge_weight_t`: corresponding edge weights from `Gw`.
- **Node features** `X_t ∈ ℝ^{N×4}`:
  - Degree
  - In-degree
  - Out-degree
  - PageRank at time `t`
- **Targets** `y_t ∈ ℝ^N`:
  - PageRank at time `t` (used with a one-step temporal shift).

We then build a `DynamicGraphTemporalSignal` dataset:

- `features[t] = X_t`
- `edge_indices[t] = edge_index_t`
- `edge_weights[t] = edge_weight_t`
- `targets[t] = y_t`

### 9.2 Model & training

Model:

- GConvGRU with:
  - `in_channels = 4`
  - `out_channels = 16`
  - `K = 2` (Chebyshev polynomial degree)
- Linear layer maps hidden state to scalar PageRank per node.

Loss and optimization:

- Loss: **L1 loss** between predicted and true PageRank.
- Optimizer: Adam with `lr = 1e-3`.
- Temporal train/validation split (e.g. first 70% of snapshots for training).

On our run, the temporal GNN trains stably:

- After 20 epochs, **L1 loss** is on the order of **5×10⁻⁴ – 10⁻³**:
  - e.g. around `0.0006` (train) vs `0.0005` (validation).

This suggests that, at least on this sampled subnetwork and time horizon, **future PageRank is reasonably predictable** from recent structural snapshots.

Outputs:

- `outputs/temporal/temporal_gnn_loss.png` – training/validation loss curves.
- `outputs/temporal/temporal_gnn_last_snapshot_prediction.csv` – table of:
  - `node`
  - `y_true_pagerank`
  - `y_pred_pagerank` (last snapshot)

If the required libraries are missing or there are too few snapshots, this module is automatically skipped.

### 9.3 Model & training on large-scale dataset

Run the script `GNN2.py` model can show the following results.

Model:

- `GCRURegressor` built on **GConvGRU**:
  - `in_channels = 4` (features: degree, in-degree, out-degree, PageRank)
  - `out_channels = hidden` (default `hidden = 32`, configurable via `--hidden`)
  - `K = 2` (Chebyshev polynomial degree)
- A **Dropout** layer with probability `dropout` (default `0.2`, via `--dropout`).
- A final **Linear** layer:
  - Maps hidden state of size `hidden` to a **scalar PageRank** per node.
- Forward pass (per snapshot):
  - Input: node features `x ∈ ℝ^{N×4}`, `edge_index`, `edge_weight`.
  - Output: `pred ∈ ℝ^N` – predicted PageRank for all nodes in that snapshot.

Loss and optimization:

- Before training, **features and targets are standardised** using only the training segment:
  - `x_norm = (x - x_mean) / x_std`
  - `y_norm = (y - y_mean) / y_std`
- Loss: **L1 loss** (`nn.L1Loss`) between **normalised** predicted and true PageRank.
- Optimiser: **AdamW** with:
  - Learning rate `lr` (default `2e-3`, via `--lr`)
  - Weight decay `weight_decay` (default `1e-4`, via `--weight_decay`)
- Learning-rate scheduler:
  - `ReduceLROnPlateau` on **validation loss**, `factor = 0.5`, `patience = 3`.
- Gradient clipping:
  - `clip_grad_norm_(model.parameters(), max_norm=2.0)` each training step.
- Early stopping:
  - Tracks best validation loss.
  - Stops if no improvement > `1e-4` for `patience = 6` epochs.
- Temporal train/validation split:
  - Let `T` be number of snapshots in `DynamicGraphTemporalSignal`.
  - Training indices: first `⌊0.7 T⌋` snapshots.
  - Validation indices: remaining snapshots `[⌊0.7 T⌋, …, T-1]`.
- Temporal supervision:
  - Inputs: snapshots `t = 0 … T-2`
  - Targets: **next-step PageRank** `PR(t+1)` for `t = 0 … T-2`
  - The model learns to predict future PageRank from current graph structure and features.

Training behaviour:

- The script prints periodic logs every 5 epochs:
  - Training loss, validation loss, and current learning rate.
- Training and validation losses are recorded per epoch to support visual inspection of convergence.
- If **aligned feature/target snapshots are fewer than 3** (`min_len < 3`), the script prints a warning:
  - `"Too few snapshots for temporal GNN (need >=3). Exit."`
  - and **returns without training**.

Outputs:

- `outputs_22wdata/temporal/temporal_gnn_loss.png`  
  – Training/validation L1 loss curves over epochs.
- `outputs_22wdata/temporal/temporal_gnn_val_predictions.csv`  
  – Per-validation-snapshot predictions:
  - `t` (validation window index)  
  - `node`  
  - `y_true_pagerank`  
  - `y_pred_pagerank`
- `outputs_22wdata/temporal/temporal_gnn_val_metrics.csv`  
  – Window-level metrics:
  - `t`, `mae`, `spearman_rho`
- `outputs_22wdata/temporal/temporal_gnn_val_metrics.png`  
  – MAE and Spearman ρ vs. validation window index.
- `outputs_22wdata/temporal/temporal_gnn_last_val_scatter.png`  
  – Scatter plot of true vs. predicted PageRank for the **last** validation snapshot.
- `outputs_22wdata/temporal/temporal_gnn_last_val_topk_metrics.csv`  
  – Top-K overlap on last validation snapshot:
  - `K`, `precision_at_k`, `jaccard`
- `outputs_22wdata/temporal/temporal_gnn_last_val_topk.png`  
  – Bar chart of Precision@K and Jaccard for `K ∈ {5, 10, 20}`.
- `outputs_22wdata/temporal/temporal_gnn_last_val_top10_bar.png`  
  – Side-by-side bar plot of true vs. predicted PageRank for top-`--topk` nodes (default 10) in last validation snapshot.
- `outputs_22wdata/temporal/temporal_gnn_val_timeseries_nodes.png`  
  – Time series (over validation windows) of true vs. predicted PageRank for `--timeseries_nodes` representative high-PageRank nodes (default 3).
- `outputs_22wdata/temporal/temporal_gnn_meta.json`  
  – Metadata summary of the run (epochs, best validation loss, hyperparameters, file paths, number of snapshots, train/val split index).


------

## 10. Visualization & Export

We produce a few key visualizations:

1. **Random subgraph**
   - Randomly sample up to 200 nodes.
   - Plot with spring layout.
   - Output: `outputs/figure_subgraph_random.png`.
2. **Ego network of the top PageRank hub**
   - Center node: wallet with highest PageRank.
   - Node size proportional to PageRank.
   - Directed edges with arrows.
   - Output: `outputs/figure_hub_ego_network.png`.

For downstream tools (e.g. **Gephi**), we export:

- `outputs/ethereum_network.gexf` – directed graph with enriched node attributes:
  - `degree`, `in_degree`, `out_degree`
  - `pagerank`, `core` (k-core index)
  - `hub`, `authority`
  - `betweenness`
  - `value_in`, `value_out`, `value_net`

Additional CSVs:

- `outputs/nodes_metrics.csv` – per-node metrics.
- `outputs/edges_metrics.csv` – per-edge weights and counts.
- `outputs/summary.json` – high-level summary with key metrics and file paths.

------

## 11. How to Run the Code

### 11.1 Dependencies

Core Python libraries:

- `requests`
- `pandas`
- `numpy`
- `networkx`
- `matplotlib`
- `scikit-learn` (for IsolationForest)
- `tqdm`

Optional:

- `powerlaw` – rigorous power-law fitting
- `torch`, `torch_geometric_temporal` – temporal GNN baseline

### 11.2 Example environment setup

Using conda (example):

```bash
conda create -n ethnet python=3.10
conda activate ethnet

pip install requests pandas numpy networkx matplotlib scikit-learn tqdm powerlaw
# Optional (for temporal GNN; versions depend on CUDA/CPU setup):
# pip install torch torch_geometric torch_geometric_temporal
```

### 11.3 Etherscan API key

Set your **Etherscan API key** as an environment variable:

```bash
# Linux / macOS
export ETHERSCAN_API_KEY="YOUR_API_KEY_HERE"

# Windows (cmd)
set ETHERSCAN_API_KEY=YOUR_API_KEY_HERE
```

> Note: The script also contains a default key(`API_KEY = os.getenv("ETHERSCAN_API_KEY", "${default key}}")`) for convenience, but you **should override** it with your own key to respect rate limits and ensure reproducibility.

### 11.4 Run the script

```bash
python ethereum_network_analysis.py
```

This will:

1. Fetch transactions for the seed addresses and their top neighbors.
2. Build the transaction network and compute all metrics.
3. Save all figures and CSV/JSON files into `outputs/` and `outputs/temporal/`.


As for the sliding windows and GNN part, run the script below:

```bash
python sliding.py
python GNN2.py
```

This will run the temporal dynamics and GNN baseline part, and save all figures and CSV/JSON files into `outputs_22wdata/` and `outputs_22wdata/temporal/`

------

## 12. Limitations & Future Work

- The dataset is a **local subnetwork** (around several prominent wallets), **not** the full Ethereum graph.
- API limits and `MAX_TX` cap the maximum number of transactions per address.
- Anomaly detection is **unsupervised**; we do not have ground truth labels for malicious or fraudulent addresses.
- The temporal GNN is a **very simple baseline** with a small feature set and limited time horizon.

Possible extensions:

- Use full-blockchain archives or larger, more diverse seed sets.
- Incorporate ERC-20 / ERC-721 token transfer events and contract interactions.
- Validate anomaly detection against curated lists of known scams, mixers, or bridge exploits.
- Use richer temporal GNN architectures (e.g. T-GAT, TGCRN) and more features (token volumes, gas usage, etc.).

------

## 13. High-Level Takeaways

Overall, our empirical results show:

- A **large, sparse**, but almost fully connected Ethereum subnetwork.
- Strong **disassortativity**, consistent with hub-and-spoke–like infrastructure.
- **Extreme centralization** of value flows: almost all inflows and outflows are concentrated in the top 10% of nodes.
- A **non-trivial fraction (~51%)** of total value flowing **between communities**, suggesting that hubs bridge modular clusters.
- Temporal patterns and a simple temporal GNN baseline indicate that **changes in centrality are structured enough to be predictably learned over time**.

This supports the view of blockchain transaction networks as **complex socio-technical systems**, where emergent macroscopic patterns (centralization, modularity, temporal dynamics) arise from simple local rules (peer-to-peer transactions).

