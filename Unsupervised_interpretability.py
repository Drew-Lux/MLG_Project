import os
import warnings
warnings.filterwarnings("ignore")
 
# ─── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend — safe for Render / Dash
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
 
import shap
import joblib
 
# ─── Project-level imports (adjust path if needed) ───────────────────────────
# These models must be trained in Role 2 (classification) first.
# If the files do not exist yet, the SHAP-for-classification block is skipped
# gracefully with a clear warning message.
 
DATASET_PATH = os.path.join("data_insight", "Diabetes_scaled_for_modeling.csv")
OUTPUT_DIR     = "outputs/clustering"
MODELS_DIR     = "outputs/models"           
BEST_MODEL_KEY = "xgboost_model.pkl"        
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
 
PALETTE = {
    "Cluster 0 — High Risk":    "#E63946",
    "Cluster 1 — Moderate Risk":"#F4A261",
    "Cluster 2 — Low Risk":     "#2A9D8F",
}
 
CLUSTER_COLORS = ["#E63946", "#F4A261", "#2A9D8F"]
 
 
def save_fig(fig: plt.Figure, name: str) -> str:
    """Save figure to OUTPUT_DIR and return the full path."""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved → {path}")
    return path
 
 
def section(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
 
section("1. Loading & Preprocessing Data")
 
df_raw = pd.read_csv(DATASET_PATH)
print(f"Dataset shape  : {df_raw.shape}")
print(f"Columns        : {list(df_raw.columns)}")
print(f"Missing values :\n{df_raw.isnull().sum()[df_raw.isnull().sum() > 0]}")
 
df = df_raw.copy()
 
# ── Encode target (diabetes_stage) ───────────────────────────────────────────
target_col = "diabetes_stage"
le_target = LabelEncoder()
df["diabetes_stage_enc"] = le_target.fit_transform(df[target_col])
print(f"\nTarget classes : {dict(enumerate(le_target.classes_))}")
 
# ── Separate features ────────────────────────────────────────────────────────
drop_cols = [target_col, "diabetes_stage_enc"]
cat_cols  = df.select_dtypes(include="object").columns.difference(drop_cols)
num_cols  = df.select_dtypes(include=[np.number]).columns.difference(drop_cols)
 
# Encode remaining categoricals
le_dict = {}
df_enc  = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    le_dict[col] = le
 
feature_cols = list(num_cols) + list(cat_cols)
X = df_enc[feature_cols].copy()
y = df["diabetes_stage_enc"].copy()
 
sample_idx = np.random.RandomState(42).choice(len(X), size=min(1000, len(X)), replace=False)

print(f"\nFeature matrix : {X.shape}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — K-MEANS CLUSTERING  (k = 3, as specified)
# ─────────────────────────────────────────────────────────────────────────────
 
section("2. K-Means Clustering (k = 3)")
 
# ── Scale features (mandatory before K-Means) ────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# ── Elbow & Silhouette validation ────────────────────────────────────────────
k_range   = range(2, 9)
inertias  = []
sil_scores = []
 
for k in k_range:
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_tmp = km_tmp.fit_predict(X_scaled)
    inertias.append(km_tmp.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels_tmp))
 
# Plot Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("K-Means Validation — Elbow & Silhouette", fontsize=14, fontweight="bold")
 
axes[0].plot(k_range, inertias, "o-", color="#E63946", linewidth=2)
axes[0].axvline(3, color="#2A9D8F", linestyle="--", label="k=3 (specified)")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Curve")
axes[0].legend()
 
axes[1].plot(k_range, sil_scores, "s-", color="#F4A261", linewidth=2)
axes[1].axvline(3, color="#2A9D8F", linestyle="--", label="k=3 (specified)")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Scores")
axes[1].legend()
 
plt.tight_layout()
save_fig(fig, "kmeans_validation.png")
print(f"  Silhouette score at k=3 : {sil_scores[1]:.4f}")
 
# ── Fit final K-Means (k = 3) ─────────────────────────────────────────────
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
 
df["cluster"]     = cluster_labels
df_enc["cluster"] = cluster_labels
 
# Save model
joblib.dump(
    {"kmeans": kmeans, "scaler": scaler},
    os.path.join(OUTPUT_DIR, "kmeans_pipeline.pkl")
)
print(f"  K-Means model saved.")
 
# ── Cluster size distribution ─────────────────────────────────────────────────
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
print(f"\n  Cluster sizes:\n{cluster_counts.to_string()}")
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CLUSTER PROFILING
# ─────────────────────────────────────────────────────────────────────────────
 
section("3. Cluster Profiling")
 
# Mean of numeric features per cluster
cluster_profile = df.groupby("cluster")[list(num_cols)].mean().round(2)
print(cluster_profile.T.to_string())
 
# ── Diabetes-stage distribution per cluster ───────────────────────────────────
stage_dist = (
    df.groupby(["cluster", target_col])
    .size()
    .reset_index(name="count")
)
stage_pivot = stage_dist.pivot(index="cluster", columns=target_col, values="count").fillna(0)
print(f"\nDiabetes stage distribution per cluster:\n{stage_pivot.to_string()}")
 
# ── Name clusters based on dominant stage ────────────────────────────────────
dominant_stage = stage_pivot.idxmax(axis=1)
cluster_names  = {}
for cid, stage in dominant_stage.items():
    if "No" in str(stage) or "no" in str(stage) or "Healthy" in str(stage):
        cluster_names[cid] = "Low Risk Lifestyle"
    elif "Pre" in str(stage) or "pre" in str(stage) or "Border" in str(stage):
        cluster_names[cid] = "Moderate Risk Lifestyle"
    else:
        cluster_names[cid] = "High Risk Lifestyle"
 
# Fallback: rank by mean target value
if len(set(cluster_names.values())) < 3:
    mean_risk = df.groupby("cluster")["diabetes_stage_enc"].mean()
    sorted_clusters = mean_risk.sort_values().index.tolist()
    risk_labels = ["Low Risk Lifestyle", "Moderate Risk Lifestyle", "High Risk Lifestyle"]
    cluster_names = {c: risk_labels[i] for i, c in enumerate(sorted_clusters)}
 
df["cluster_name"] = df["cluster"].map(cluster_names)
print(f"\nCluster labels : {cluster_names}")
 
# ── Heatmap of cluster profiles ───────────────────────────────────────────────
# Normalise to z-score for comparability
profile_z = cluster_profile.apply(lambda col: (col - col.mean()) / (col.std() + 1e-9))
 
fig, ax = plt.subplots(figsize=(max(10, len(num_cols) * 0.8), 4))
sns.heatmap(
    profile_z.T,
    annot=cluster_profile.T,
    fmt=".1f",
    cmap="RdYlGn_r",
    ax=ax,
    linewidths=0.4,
    cbar_kws={"label": "Z-Score"},
)
ax.set_title("Cluster Feature Profiles (Annotated with Raw Means)", fontsize=13, fontweight="bold")
ax.set_xticklabels([cluster_names.get(i, f"C{i}") for i in range(3)], rotation=15, ha="right")
plt.tight_layout()
save_fig(fig, "cluster_profile_heatmap.png")
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PCA VISUALISATION OF CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
 
section("4. PCA Cluster Visualisation")
 
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_
 
fig, ax = plt.subplots(figsize=(8, 6))
for cid in range(3):
    mask = cluster_labels == cid
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        s=20, alpha=0.6,
        color=CLUSTER_COLORS[cid],
        label=cluster_names.get(cid, f"Cluster {cid}"),
    )
 
# Plot centroids in PCA space
centers_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
           s=250, marker="X", color="black", zorder=5, label="Centroids")
 
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
ax.set_title("Patient Lifestyle Clusters — PCA Projection", fontsize=13, fontweight="bold")
ax.legend(framealpha=0.9)
plt.tight_layout()
save_fig(fig, "cluster_pca.png")
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SHAP FOR CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
 
section("5. SHAP Values — Risk Classification")
 
model_path = os.path.join(MODELS_DIR, BEST_MODEL_KEY)
 
if not os.path.exists(model_path):
    print(f"  ⚠  Model file not found at: {model_path}")
    print("     Ensure Role 2 (classification) has saved the model first.")
    print("     Skipping SHAP classification section.")
    clf_model = None
else:
    clf_model = joblib.load(model_path)
    print(f"  ✔  Loaded model from {model_path}")
 
    # ── SHAP TreeExplainer ────────────────────────────────────────────────────
    explainer_clf = shap.TreeExplainer(clf_model)
    # Use a sample for speed (max 1000 rows)
    sample_idx = np.random.RandomState(42).choice(len(X), size=min(1000, len(X)), replace=False)
    X_sample   = X.iloc[sample_idx]
    shap_vals  = explainer_clf.shap_values(X_sample)
 
    # Handle multi-class: use mean abs across classes
    if isinstance(shap_vals, list):
        shap_mean = np.mean([np.abs(s) for s in shap_vals], axis=0)
    else:
        shap_mean = np.abs(shap_vals)
 
    mean_shap_df = pd.DataFrame({
        "feature":    feature_cols,
        "mean_|SHAP|": shap_mean.mean(axis=0)
    }).sort_values("mean_|SHAP|", ascending=False)
 
    print(f"\n  Top 10 classification drivers:\n{mean_shap_df.head(10).to_string(index=False)}")
    mean_shap_df.to_csv(os.path.join(OUTPUT_DIR, "shap_classification_importance.csv"), index=False)
 
    # ── Bar plot ──────────────────────────────────────────────────────────────
    top_n = min(15, len(mean_shap_df))
    fig, ax = plt.subplots(figsize=(9, 5))
    top_df = mean_shap_df.head(top_n)
    ax.barh(top_df["feature"][::-1], top_df["mean_|SHAP|"][::-1], color="#E63946", alpha=0.85)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"Top {top_n} Features — Diabetes Risk Classification\n(SHAP Feature Importance)",
                 fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    save_fig(fig, "shap_classification_bar.png")
 
    # Beeswarm / Summary plot
fig = plt.figure(figsize=(10, 6))

if isinstance(shap_vals, list):
    # For multi-class, pick the class with highest avg |shap|
    best_class = int(np.argmax([np.abs(s).mean() for s in shap_vals]))
    shap.summary_plot(
        shap_vals[best_class],
        X_sample,
        feature_names=feature_cols,
        show=False,
        plot_type="dot",
        max_display=15
    )
    plt.title(f"SHAP Summary - Class: {le_target.classes_[best_class]}", fontweight="bold")

else:
    shap.summary_plot(
        shap_vals,
        X_sample,
        feature_names=feature_cols,
        show=False,
        plot_type="dot",
        max_display=15
    )
    plt.title("SHAP Summary - Risk Classification", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "shap_classification_beeswarm.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SHAP FOR CLUSTER INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────
 
section("6. SHAP Values — Cluster Interpretation")
 
from sklearn.ensemble import RandomForestClassifier
 
# Surrogate: predict cluster label from original features
surrogate = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
surrogate.fit(X_scaled, cluster_labels)
surrogate_acc = surrogate.score(X_scaled, cluster_labels)
print(f"  Surrogate model accuracy (train): {surrogate_acc:.4f}  "
      f"(higher = clusters are more linearly separable)")
 
# SHAP for surrogate
explainer_clust = shap.TreeExplainer(surrogate)
X_sample_df     = pd.DataFrame(X_scaled, columns=feature_cols)
X_s_sample      = X_sample_df.iloc[sample_idx]

shap_clust = explainer_clust.shap_values(X_s_sample)

# ── FIX: handle both 3D array and list-of-2D-arrays ──────────────────────────
if isinstance(shap_clust, np.ndarray) and shap_clust.ndim == 3:
    # shape: (n_samples, n_features, n_classes) — newer SHAP versions
    shap_clust_mean = np.abs(shap_clust).mean(axis=(0, 2))   # → (n_features,)
elif isinstance(shap_clust, list):
    # list of (n_samples, n_features) arrays — one per class
    shap_clust_mean = np.mean([np.abs(s) for s in shap_clust], axis=0).mean(axis=0)  # → (n_features,)
else:
    shap_clust_mean = np.abs(shap_clust).mean(axis=0)        # binary fallback

print(f"  SHAP output shape check — features: {len(feature_cols)}, shap values: {len(shap_clust_mean)}")

mean_shap_clust_df = pd.DataFrame({
    "feature":     feature_cols,
    "mean_|SHAP|": shap_clust_mean,
}).sort_values("mean_|SHAP|", ascending=False)

print(f"\n  Top 10 cluster-driving features:\n{mean_shap_clust_df.head(10).to_string(index=False)}")
mean_shap_clust_df.to_csv(os.path.join(OUTPUT_DIR, "shap_cluster_importance.csv"), index=False)
 
# ── Bar chart ─────────────────────────────────────────────────────────────────
top_n = min(15, len(mean_shap_clust_df))
fig, ax = plt.subplots(figsize=(9, 5))
top_df = mean_shap_clust_df.head(top_n)
colors = ["#2A9D8F" if i % 2 == 0 else "#F4A261" for i in range(top_n)]
ax.barh(top_df["feature"][::-1], top_df["mean_|SHAP|"][::-1], color=colors[::-1], alpha=0.85)
ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax.set_title(f"Top {top_n} Features — Patient Lifestyle Segments\n(SHAP via Surrogate Model)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save_fig(fig, "shap_cluster_bar.png")
 
# ── Per-cluster SHAP summary ──────────────────────────────────────────────────
for cid in range(3):
    fig = plt.figure(figsize=(10, 5))
    
    # Handle both 3D array and list-of-2D formats
    if isinstance(shap_clust, np.ndarray) and shap_clust.ndim == 3:
        # shape: (n_samples, n_features, n_classes) → slice class cid
        shap_for_class = shap_clust[:, :, cid]
    elif isinstance(shap_clust, list):
        shap_for_class = shap_clust[cid]
    else:
        shap_for_class = shap_clust

    shap.summary_plot(shap_for_class, X_s_sample,
                      feature_names=feature_cols, show=False,
                      plot_type="dot", max_display=12)
    plt.title(f"SHAP Summary — {cluster_names.get(cid, f'Cluster {cid}')}",
              fontweight="bold")
    plt.tight_layout()
    save_fig(fig, f"shap_cluster_{cid}_beeswarm.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — ACTIONABLE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
 
section("7. Actionable Recommendations (Dashboard-Ready)")
 
# Build recommendations from cluster profiles + SHAP
# Top SHAP-driven features per cluster
 
RECOMMENDATIONS = {}
for cid in range(3):
    # Handle 3D array (n_samples, n_features, n_classes) vs list of 2D arrays
    if isinstance(shap_clust, np.ndarray) and shap_clust.ndim == 3:
        shap_for_class = shap_clust[:, :, cid]   # → (n_samples, n_features)
    elif isinstance(shap_clust, list):
        shap_for_class = shap_clust[cid]
    else:
        shap_for_class = shap_clust

    per_cluster_shap = pd.Series(
        np.abs(shap_for_class).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)

    top_features = per_cluster_shap.head(5).index.tolist()
    cluster_data = df[df["cluster"] == cid][list(num_cols)].mean()

    recs = []
    for feat in top_features:
        if feat not in cluster_data.index:
            continue
        val    = cluster_data[feat]
        f_lower = feat.lower()

        if "bmi" in f_lower:
            if val > 27:
                recs.append(f"⚠ BMI is elevated ({val:.1f}). Recommend structured weight-management plan and dietitian referral.")
            else:
                recs.append(f"✔ BMI is within acceptable range ({val:.1f}). Encourage maintenance through regular physical activity.")
        elif "glucose" in f_lower or "blood_glucose" in f_lower:
            if val > 140:
                recs.append(f"⚠ Average blood glucose is high ({val:.1f} mg/dL). Immediate dietary carbohydrate review and HbA1c monitoring advised.")
            elif val > 100:
                recs.append(f"⚠ Blood glucose is borderline ({val:.1f} mg/dL). Recommend reducing refined sugar intake and 30-min daily walks.")
        elif "hba1c" in f_lower or "a1c" in f_lower:
            if val > 6.5:
                recs.append(f"⚠ HbA1c is in diabetic range ({val:.2f}%). Refer to endocrinologist and start medication review.")
            elif val > 5.7:
                recs.append(f"⚠ HbA1c suggests pre-diabetes ({val:.2f}%). Begin lifestyle intervention programme immediately.")
        elif "exercise" in f_lower or "physical" in f_lower or "activity" in f_lower:
            if val < 3:
                recs.append(f"⚠ Low physical activity ({val:.1f} days/week). Target ≥150 min/week moderate aerobic exercise (ADA guidelines).")
            else:
                recs.append(f"✔ Physical activity is adequate ({val:.1f} days/week). Encourage consistency.")
        elif "age" in f_lower:
            if val > 50:
                recs.append(f"ℹ Average age is {val:.0f}. Prioritise annual screening and cardiovascular risk co-assessment.")
        elif "diet" in f_lower:
            recs.append(f"ℹ Diet score is {val:.1f}. Review nutritional counselling access for this group.")
        elif "stress" in f_lower or "sleep" in f_lower:
            if val > 6:
                recs.append(f"⚠ High stress/poor sleep score ({val:.1f}). Introduce mindfulness / sleep-hygiene workshops.")
        elif "smoking" in f_lower:
            if val > 0.3:
                recs.append(f"⚠ Notable smoking prevalence ({val:.2f}). Provide cessation support and counselling resources.")
        elif "alcohol" in f_lower:
            if val > 0.3:
                recs.append(f"⚠ Notable alcohol usage ({val:.2f}). Provide cessation support and counselling resources.")
        else:
            recs.append(f"ℹ Feature '{feat}' (mean={val:.2f}) is influential for this segment. Clinical review recommended.")

    if not recs:
        recs = ["No strong numerical signals detected — review categorical features manually."]

    RECOMMENDATIONS[cid] = {
        "cluster_id":        cid,
        "cluster_name":      cluster_names.get(cid, f"Cluster {cid}"),
        "patient_count":     int((df["cluster"] == cid).sum()),
        "top_shap_features": top_features,
        "recommendations":   recs,
    }
 
# ── Print recommendations ─────────────────────────────────────────────────────
for cid, info in RECOMMENDATIONS.items():
    print(f"\n  ── {info['cluster_name']}  (n={info['patient_count']}) ──")
    print(f"  Key drivers: {', '.join(info['top_shap_features'])}")
    for r in info["recommendations"]:
        print(f"    {r}")
 
# ── Save as CSV ───────────────────────────────────────────────────────────────
rec_rows = []
for cid, info in RECOMMENDATIONS.items():
    for r in info["recommendations"]:
        rec_rows.append({
            "cluster_id":    cid,
            "cluster_name":  info["cluster_name"],
            "patient_count": info["patient_count"],
            "key_drivers":   ", ".join(info["top_shap_features"]),
            "recommendation": r,
        })
 
rec_df = pd.DataFrame(rec_rows)
rec_df.to_csv(os.path.join(OUTPUT_DIR, "actionable_recommendations.csv"), index=False)
print(f"\n  Recommendations saved → {OUTPUT_DIR}/actionable_recommendations.csv")
 
# ── Save full cluster profiles ────────────────────────────────────────────────
cluster_profile.to_csv(os.path.join(OUTPUT_DIR, "cluster_profiles.csv"))
stage_pivot.to_csv(os.path.join(OUTPUT_DIR, "cluster_stage_distribution.csv"))
 
# ── Save cluster assignments ──────────────────────────────────────────────────
df[["cluster", "cluster_name", target_col]].to_csv(
    os.path.join(OUTPUT_DIR, "patient_cluster_assignments.csv"), index=False
)
 
print(f"\n{'═' * 70}")
print("  Pipeline complete. All outputs saved to:", OUTPUT_DIR)
print(f"{'═' * 70}\n")