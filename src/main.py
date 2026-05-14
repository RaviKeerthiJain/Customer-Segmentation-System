import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("../data/Mall_Customers.csv")

print("First 5 Rows:\n")

print(df.head())

# =========================
# DATASET INFO
# =========================

print("\nDataset Shape:", df.shape)

print("\nDataset Columns:\n")

print(df.columns)

# =========================
# GENDER DISTRIBUTION
# =========================

plt.figure(figsize=(6,4))

sns.countplot(
    x='Gender',
    data=df
)

plt.title("Gender Distribution")

plt.savefig("../outputs/gender_distribution.png")

plt.show()

# =========================
# SELECT FEATURES
# =========================

X = df.iloc[:, [3, 4]].values

# Annual Income
# Spending Score

# =========================
# ELBOW METHOD
# =========================

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

# =========================
# PLOT ELBOW GRAPH
# =========================

plt.figure(figsize=(8,5))

plt.plot(range(1, 11), wcss, marker='o')

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.savefig("../outputs/elbow_method.png")

plt.show()

# =========================
# APPLY K-MEANS
# =========================

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

y_kmeans = kmeans.fit_predict(X)

# =========================
# VISUALIZE CLUSTERS
# =========================

plt.figure(figsize=(8,6))

plt.scatter(
    X[y_kmeans == 0, 0],
    X[y_kmeans == 0, 1],
    s=100,
    label='Cluster 1'
)

plt.scatter(
    X[y_kmeans == 1, 0],
    X[y_kmeans == 1, 1],
    s=100,
    label='Cluster 2'
)

plt.scatter(
    X[y_kmeans == 2, 0],
    X[y_kmeans == 2, 1],
    s=100,
    label='Cluster 3'
)

plt.scatter(
    X[y_kmeans == 3, 0],
    X[y_kmeans == 3, 1],
    s=100,
    label='Cluster 4'
)

plt.scatter(
    X[y_kmeans == 4, 0],
    X[y_kmeans == 4, 1],
    s=100,
    label='Cluster 5'
)

# =========================
# CENTROIDS
# =========================

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='black',
    label='Centroids'
)

# =========================
# GRAPH DETAILS
# =========================

plt.title("Customer Segmentation")

plt.xlabel("Annual Income")

plt.ylabel("Spending Score")

plt.legend()

plt.savefig("../outputs/customer_clusters.png")

plt.show()

# =========================
# CLUSTER ANALYSIS
# =========================

df["Cluster"] = y_kmeans

print("\nCluster Counts:\n")

print(df["Cluster"].value_counts())

# =========================
# CUSTOMER SEGMENT LABELS
# =========================

cluster_names = {
    0: "Standard Customers",
    1: "Premium Customers",
    2: "Budget Customers",
    3: "Careful Customers",
    4: "Target Customers"
}

df["Customer_Type"] = df["Cluster"].map(cluster_names)

print("\nCustomer Type Counts:\n")

print(df["Customer_Type"].value_counts())

# =========================
# CLUSTER SUMMARY
# =========================

cluster_summary = df.groupby("Customer_Type")[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].mean()

print("\nCluster Summary:\n")

print(cluster_summary)

print("\nProject Completed Successfully")