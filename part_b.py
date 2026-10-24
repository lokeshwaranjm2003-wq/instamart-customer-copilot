
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# File paths
CLEAN = Path("data/clean")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Load cleaned data
customers = pd.read_csv(CLEAN / "customers.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Use completed orders only for RFM
completed = orders.loc[
    orders["status"].eq("completed")
].copy()

# Calculate the reference date from the latest completed order
reference_date = completed["order_ts"].max().normalize() + pd.Timedelta(days=1)

# Calculate RFM for each customer
rfm = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

rfm["recency"] = (
    reference_date - rfm["last_order"]
).dt.days

rfm = rfm.drop(columns=["last_order"])

# Scale RFM values before clustering
rfm_features = rfm[["recency", "frequency", "monetary"]]

scaler = StandardScaler()
scaled_rfm = scaler.fit_transform(rfm_features)

# K-Means clustering with four segments
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["segment_id"] = kmeans.fit_predict(scaled_rfm)

# Profile each cluster
profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).round(2)

print("\nRFM segment profile:")
print(profile)

# Premium is the segment with the highest mean monetary value
premium_segment = profile["mean_monetary"].idxmax()

# Give readable names to segments

# Assign business-friendly segment names using cluster profiles
segment_names = {}

for segment_id, row in profile.iterrows():
    if segment_id == premium_segment:
        name = "Premium"
    elif row["mean_recency"] > 60:
        name = "At Risk"
    elif row["mean_frequency"] >= 15:
        name = "Loyal"
    else:
        name = "Regular"

    segment_names[segment_id] = name

rfm["segment_name"] = rfm["segment_id"].map(segment_names)


# Mark customers belonging to the Premium segment
rfm["is_premium"] = (
    rfm["segment_id"] == premium_segment
)


# Save the required segmentation file
segment_output = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]

segment_output.to_csv(
    OUTPUTS / "customer_segments.csv",
    index=False
)

# PCA on the scaled RFM matrix
pca = PCA(n_components=2)
pca.fit(scaled_rfm)

print("\nPCA explained variance ratio:")
print(pca.explained_variance_ratio_.round(3))
print(
    "Total explained variance:",
    round(pca.explained_variance_ratio_.sum(), 3)
)

print("\nPremium segment ID:", premium_segment)
print("Customer segments saved to outputs/customer_segments.csv")


# K-Means comparison: K = 2 to 8
from sklearn.metrics import silhouette_score

print("\nK-MEANS COMPARISON")
print("-" * 45)
print(f"{'K':<5} {'Inertia':<15} {'Silhouette':<15}")

results = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(scaled_rfm)

    inertia = model.inertia_
    silhouette = silhouette_score(scaled_rfm, labels)

    results.append({
        "K": k,
        "inertia": inertia,
        "silhouette_score": silhouette
    })

    print(f"{k:<5} {inertia:<15.2f} {silhouette:<15.4f}")

comparison = pd.DataFrame(results)

comparison.to_csv(
    OUTPUTS / "kmeans_comparison.csv",
    index=False
)

print("\nSelected K: 4")
print("Saved: outputs/kmeans_comparison.csv")
