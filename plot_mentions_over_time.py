import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FOLDERS = ["pushshift_22_24_json", "pushshift_25_json"]

rows = []

for folder in FOLDERS:
    for file in Path(folder).glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

df = pd.DataFrame(rows)
print(f"loaded {len(df):,} rows")

df = df[["created_date", "model_detected"]].dropna()

df["created_date"] = pd.to_datetime(df["created_date"])
df["month"] = df["created_date"].dt.to_period("M").dt.to_timestamp()

counts = (
    df.groupby(["month", "model_detected"])
      .size()
      .unstack(fill_value=0)
      .sort_index()
)

plt.figure(figsize=(12, 6))

for model in counts.columns:
    plt.plot(counts.index, counts[model], marker="o", label=model)

plt.xlabel("Month")
plt.ylabel("Mentions")
plt.title("Model Mentions Over Time")
plt.legend()
plt.tight_layout()
plt.show()
