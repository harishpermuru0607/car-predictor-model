
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/quikr_car.csv")
OUTPUT_PATH = Path("data/car_data.csv")

if not INPUT_PATH.exists():
    raise SystemExit(
        "Dataset not found! Place quikr_car.csv inside the data folder."
    )

# Load original dataset
car = pd.read_csv(INPUT_PATH)

# Standardize column names
car.columns = car.columns.str.strip()

required = ["name", "company", "year", "Price", "kms_driven", "fuel_type"]
missing = [col for col in required if col not in car.columns]

if missing:
    raise SystemExit(f"Missing required columns: {missing}")

# Remove rows with missing essential information
car = car.dropna(subset=["name", "company", "year", "Price", "kms_driven", "fuel_type"]).copy()

# Clean year
car["year"] = pd.to_numeric(car["year"], errors="coerce")

# Clean selling price
car["Price"] = (
    car["Price"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)
car.loc[
    car["Price"].str.lower() == "ask for price", "Price"
] = None
car["Price"] = pd.to_numeric(car["Price"], errors="coerce")

# Clean kilometers driven
car["kms_driven"] = (
    car["kms_driven"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace(r"(?i)\s*kms?\s*$", "", regex=True)
    .str.strip()
)
car["kms_driven"] = pd.to_numeric(
    car["kms_driven"], errors="coerce"
)

# Remove invalid rows
car = car.dropna(
    subset=["year", "Price", "kms_driven", "fuel_type"]
)
car = car[
    (car["year"] >= 1900)
    & (car["year"] <= 2100)
    & (car["Price"] > 0)
    & (car["kms_driven"] >= 0)
].copy()

# Keep the first three words of each car name
car["name"] = (
    car["name"].astype(str).str.split().str[:3].str.join(" ")
)

# Normalize text values
for col in ["name", "company", "fuel_type"]:
    car[col] = car[col].astype(str).str.strip()

# Save cleaned data
car = car[required].reset_index(drop=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
car.to_csv(OUTPUT_PATH, index=False)

print("Dataset cleaned successfully!")
print("Rows:", len(car))
print("Columns:", car.columns.tolist())
print("\nFirst five rows:")
print(car.head())
print(f"\nSaved to: {OUTPUT_PATH}")
