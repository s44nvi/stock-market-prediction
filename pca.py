# ============================================================
#  Stock Market Trend Prediction using Principal Component Analysis (PCA)
#  Dataset: Stock Market Dataset for Financial Analysis (Kaggle)
#  Team: Saanvi Baraskar, Rugveda Belekar, Abhhijatya Singh
#  Batch: CSBS SY - A1
# ============================================================

# ── Imports ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

# ============================================================
#  PHASE I — DATA ENGINEERING
# ============================================================

# ── Step 1: Data Acquisition ─────────────────────────────────
# Download the dataset from Kaggle and place the CSV file in
# the same folder as this script, then update the path below.
# Dataset: https://www.kaggle.com/datasets/s3programmer/stock-market-dataset-for-financial-analysis

CSV_PATH = "stock_market_data.csv"   # <-- update filename if different

print("=" * 60)
print("PHASE I: DATA ENGINEERING")
print("=" * 60)

print("\n[Step 1] Loading dataset...")
df = pd.read_csv(CSV_PATH)

print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")
print(df.head(3))

# ── Step 2: Data Cleaning ────────────────────────────────────
print("\n[Step 2] Cleaning data...")

# Convert 'Date' column to datetime if present and set as index
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

# Drop non-numeric columns (e.g. ticker symbol, company name)
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"  Dropping non-numeric columns: {non_numeric}")
    df = df.drop(columns=non_numeric)

# Handle missing values using linear interpolation
missing_before = df.isnull().sum().sum()
df = df.interpolate(method="linear").dropna()
print(f"  Missing values fixed: {missing_before} → {df.isnull().sum().sum()}")

# Remove extreme outliers using IQR method
Q1 = df.quantile(0.01)
Q3 = df.quantile(0.99)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 3 * IQR)) | (df > (Q3 + 3 * IQR))).any(axis=1)]
print(f"  Rows after outlier removal: {len(df)}")

# ── Step 3: Feature Scaling (Z-score Standardization) ────────
print("\n[Step 3] Standardizing features...")

# Define target: we predict 'Close' price (or next-day close)
TARGET_COL = "Close"   # change to match your CSV column name

# Separate features (X) and target (y)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Apply Z-score normalization so all features have mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  Features scaled: {X.shape[1]} columns")

# ── Correlation Heatmap (EDA) ─────────────────────────────────
print("\n  Plotting correlation heatmap...")
plt.figure(figsize=(12, 8))
corr_matrix = pd.DataFrame(X_scaled, columns=X.columns).corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap (Before PCA)", fontsize=14)
plt.tight_layout()
plt.savefig("1_correlation_heatmap.png", dpi=150)
plt.show()
print("  Saved: 1_correlation_heatmap.png")

# ============================================================
#  PHASE II — DIMENSIONALITY REDUCTION (PCA)
# ============================================================

print("\n" + "=" * 60)
print("PHASE II: DIMENSIONALITY REDUCTION (PCA)")
print("=" * 60)

# ── Step 4: Covariance Computation ───────────────────────────
print("\n[Step 4] Computing covariance matrix...")

# The covariance matrix shows how each pair of features varies together
cov_matrix = np.cov(X_scaled.T)
print(f"  Covariance matrix shape: {cov_matrix.shape}")

# ── Step 5: Eigen-Decomposition ──────────────────────────────
print("\n[Step 5] Performing Eigen-Decomposition...")

# Eigenvalues tell us how much variance each principal component captures
# Eigenvectors tell us the direction of each principal component
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues in descending order
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[sorted_idx].real
eigenvectors = eigenvectors[:, sorted_idx].real

print(f"  Top 5 Eigenvalues: {np.round(eigenvalues[:5], 4)}")

# ── Step 6: Feature Selection via Scree Plot ─────────────────
print("\n[Step 6] Selecting components that retain 95% variance...")

# Full PCA to determine how many components explain 95% variance
pca_full = PCA()
pca_full.fit(X_scaled)

explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance      = np.cumsum(explained_variance_ratio)

# Find minimum number of components for 95% explained variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"  Components needed for 95% variance: {n_components_95}")

# ── Scree Plot ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Individual explained variance
axes[0].bar(range(1, min(21, len(explained_variance_ratio) + 1)),
            explained_variance_ratio[:20] * 100,
            color="steelblue", edgecolor="white")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance (%)")
axes[0].set_title("Scree Plot — Individual Explained Variance")

# Cumulative explained variance
axes[1].plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance * 100,
             marker="o", color="darkorange", linewidth=2)
axes[1].axhline(y=95, color="red", linestyle="--", label="95% threshold")
axes[1].axvline(x=n_components_95, color="green", linestyle="--",
                label=f"{n_components_95} components")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance (%)")
axes[1].set_title("Cumulative Explained Variance")
axes[1].legend()

plt.suptitle("PCA — Scree Plot Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("2_scree_plot.png", dpi=150)
plt.show()
print("  Saved: 2_scree_plot.png")

# ── Apply PCA with selected components ───────────────────────
pca = PCA(n_components=n_components_95)
X_pca = pca.fit_transform(X_scaled)

print(f"\n  Original feature dimensions : {X_scaled.shape[1]}")
print(f"  Reduced PCA dimensions      : {X_pca.shape[1]}")
print(f"  Total variance retained     : {cumulative_variance[n_components_95-1]*100:.2f}%")

# ── PCA Biplot: First 2 Components ───────────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, s=10, c=y, cmap="viridis")
plt.colorbar(label="Close Price")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA — First Two Principal Components vs Close Price")
plt.tight_layout()
plt.savefig("3_pca_scatter.png", dpi=150)
plt.show()
print("  Saved: 3_pca_scatter.png")

# ── Loading Plot: Which features drive PC1? ───────────────────
plt.figure(figsize=(10, 5))
loadings = pd.Series(pca.components_[0], index=X.columns)
loadings.abs().sort_values(ascending=False)[:15].plot(
    kind="bar", color="teal", edgecolor="white")
plt.title("PCA — Top Feature Loadings on PC1")
plt.ylabel("Absolute Loading")
plt.xlabel("Feature")
plt.tight_layout()
plt.savefig("4_pca_loadings.png", dpi=150)
plt.show()
print("  Saved: 4_pca_loadings.png")

# ============================================================
#  PHASE III — PREDICTION & VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("PHASE III: PREDICTION & VALIDATION")
print("=" * 60)

# ── Step 7: Model Training ────────────────────────────────────
print("\n[Step 7] Splitting data and training models...")

# 80% train, 20% test split (shuffle=False preserves time order)
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, shuffle=False)

print(f"  Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# --- Model 1: Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# --- Model 2: Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# ── Step 8: Performance Evaluation ───────────────────────────
print("\n[Step 8] Evaluating model performance...")

def evaluate(name, y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{name}]")
    print(f"    MSE  : {mse:.4f}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    R²   : {r2:.4f}")
    return mse, rmse, r2

lr_mse, lr_rmse, lr_r2 = evaluate("Linear Regression", y_test, y_pred_lr)
rf_mse, rf_rmse, rf_r2 = evaluate("Random Forest",     y_test, y_pred_rf)

# ── Model Comparison Bar Chart ────────────────────────────────
metrics_df = pd.DataFrame({
    "Model"  : ["Linear Regression", "Random Forest"],
    "MSE"    : [lr_mse,  rf_mse],
    "RMSE"   : [lr_rmse, rf_rmse],
    "R²"     : [lr_r2,   rf_r2],
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["MSE", "RMSE", "R²"]):
    bars = ax.bar(metrics_df["Model"], metrics_df[col],
                  color=["steelblue", "darkorange"], edgecolor="white")
    ax.set_title(col)
    ax.set_ylabel(col)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)

plt.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("5_model_comparison.png", dpi=150)
plt.show()
print("  Saved: 5_model_comparison.png")

# ── Step 9: Predicted vs Actual Visualization ─────────────────
print("\n[Step 9] Plotting predicted vs actual stock trends...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
slice_size = 100

for ax, (name, y_pred) in zip(axes,
        [("Linear Regression", y_pred_lr),
         ("Random Forest",     y_pred_rf)]):
    # Use [-slice_size:] to only plot the last 100 days
    ax.plot(y_test.values[-slice_size:], label="Actual Close Price", color="black", linewidth=2)
    ax.plot(y_pred[-slice_size:], label=f"Predicted ({name})", 
            color="tomato" if "Linear" in name else "steelblue", linewidth=1.5, linestyle="--")
    ax.set_title(f"Actual vs Predicted — {name}", fontsize=12)
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Close Price")
    ax.legend()

plt.suptitle("Stock Price: Actual vs Predicted", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("6_actual_vs_predicted.png", dpi=150)
plt.show()
print("  Saved: 6_actual_vs_predicted.png")

# ── Residual Plot ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, (name, y_pred) in zip(axes,
        [("Linear Regression", y_pred_lr),
         ("Random Forest",     y_pred_rf)]):
    residuals = y_test.values - y_pred
    ax.scatter(range(len(residuals)), residuals, alpha=0.4, s=10)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title(f"Residuals — {name}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Residual (Actual − Predicted)")

plt.suptitle("Residual Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("7_residuals.png", dpi=150)
plt.show()
print("  Saved: 7_residuals.png")

# ============================================================
#  SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Original features           : {X.shape[1]}")
print(f"  PCA components retained     : {n_components_95} (95% variance)")
print(f"  Dimensionality reduction    : {X.shape[1] - n_components_95} features removed")
print(f"\n  Linear Regression → R²={lr_r2:.4f} | RMSE={lr_rmse:.4f}")
print(f"  Random Forest     → R²={rf_r2:.4f} | RMSE={rf_rmse:.4f}")
print("\nAll plots saved as PNG files in the current directory.")
print("=" * 60)