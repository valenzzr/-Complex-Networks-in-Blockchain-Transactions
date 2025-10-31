import pandas as pd, numpy as np, re

df = pd.read_csv("outputs/ethereum_transactions_20251031_161127.csv")  # 改成你的文件名

# 1) 去掉首尾空格
for c in ("from","to"):
    df[c] = df[c].astype(str).str.strip()

# 2) 统计空字符串/NaN 数
print("from=='' rows:", (df["from"] == "").sum())
print("to==''   rows:", (df["to"] == "").sum())
print("from is NaN:", df["from"].isna().sum(), "to is NaN:", df["to"].isna().sum())

# 3) 合约创建（to=='' 但有 contractAddress）
if "contractAddress" in df.columns:
    has_contract_addr = (df["contractAddress"].astype(str).str.strip() != "")
    print("contract-creation rows (to=='', have contractAddress):",
          ((df["to"]=="") & has_contract_addr).sum())

# 4) 非法地址长度（不是 42 个字符的 0x……）
def bad_addr(s):
    s = str(s)
    return not bool(re.fullmatch(r"0x[0-9a-fA-F]{40}", s))

print("bad FROM length:", df["from"].map(bad_addr).sum())
print("bad TO   length:", df["to"].map(bad_addr).sum())

# 5) 看几条具体样例
print(df[(df["to"]=="")].head(5))
