
import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 1. CONFIGURATION
# -----------------------------
DATA_PATH = "data/car_data.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def normalize_column(name):
    """Normalize column names for easier matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# -----------------------------
# 2. LOAD DATASET
# -----------------------------
if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
    raise SystemExit(
        "Dataset is missing or empty. Add your car dataset to "
        "data/car_data.csv and run this script again."
    )

df = pd.read_csv(DATA_PATH)
df.columns = [str(col).strip() for col in df.columns]
df = df.drop_duplicates()

if df.empty:
    raise SystemExit("The CSV contains no usable rows.")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nAvailable columns:", df.columns.tolist())
print("\nFirst five rows:\n", df.head())


# -----------------------------
# 3. IDENTIFY TARGET COLUMN
# -----------------------------
target_aliases = {
    "sellingprice",
    "selling_price",
    "price",
    "sellingpriceinlakhs",
    "sellingpriceinlakh",
    "sellingpriceinr",
    "selling_price_in_lakhs",
    "selling_price_in_lakh",
}

normalized_columns = {
    normalize_column(col): col for col in df.columns
}

target_column = next(
    (
        original
        for normalized, original in normalized_columns.items()
        if normalized in target_aliases
    ),
    None,
)

if target_column is None:
    raise SystemExit(
        "\nCould not identify the selling price column. "
        "Rename the target column to 'Selling_Price' in your CSV."
    )

# Convert the target into numeric values.
df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
df = df.dropna(subset=[target_column])

if len(df) < 5:
    raise SystemExit(
        "At least five valid rows are needed to train the model."
    )


# -----------------------------
# 4. PREPARE FEATURES
# -----------------------------
X = df.drop(columns=[target_column])
y = df[target_column]

# Remove columns that contain no useful information.
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=False) > 1]

# Convert numeric-looking text columns when possible.
for col in X.select_dtypes(include=["object", "string"]).columns:
    converted = pd.to_numeric(X[col], errors="coerce")
    if converted.notna().mean() >= 0.90:
        X[col] = converted

if X.shape[1] == 0:
    raise SystemExit("No useful input features were found.")

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = [
    col for col in X.columns if col not in numeric_features
]

print("\nTarget column:", target_column)
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)


# -----------------------------
# 5. PREPROCESSING
# -----------------------------
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers=transformers)


# -----------------------------
# 6. SPLIT DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# -----------------------------
# 7. TRAIN AND COMPARE MODELS
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
}

results = {}

for name, estimator in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results[name] = {
    "pipeline": pipeline,
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "actual": y_test.tolist(),
    "predictions": predictions.tolist(),
}

    print(f"\n{name}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")


# -----------------------------
# 8. SELECT BEST MODEL
# -----------------------------
best_name = min(
    results,
    key=lambda name: results[name]["mae"],
)
best_result = results[best_name]

joblib.dump(
    best_result["pipeline"],
    os.path.join(MODEL_DIR, "car_price_model.joblib"),
)

# Store feature information for the web app.
feature_info = {
    "feature_columns": X.columns.tolist(),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "defaults": {},
    "options": {},
    "target_column": target_column,
    "model_name": best_name,
    "target_min": float(y.min()),
    "target_max": float(y.max()),
}

for col in numeric_features:
    values = pd.to_numeric(X[col], errors="coerce").dropna()
    feature_info["defaults"][col] = (
        float(values.median()) if not values.empty else 0.0
    )
    feature_info.setdefault("ranges", {})[col] = [
        float(values.min()) if not values.empty else 0.0,
        float(values.max()) if not values.empty else 100.0,
    ]

for col in categorical_features:
    feature_info["options"][col] = (
        X[col].dropna().astype(str).value_counts().index.tolist()[:100]
    )

joblib.dump(
    feature_info,
    os.path.join(MODEL_DIR, "model_info.joblib"),
)

print("\n-----------------------------")
print("TRAINING COMPLETED!")
print("Best model:", best_name)
print(f"Best model MAE: {best_result['mae']:.3f}")
print(f"Best model RMSE: {best_result['rmse']:.3f}")
print(f"Best model R²: {best_result['r2']:.3f}")
print("Saved model to models/car_price_model.joblib")
print("-----------------------------")
