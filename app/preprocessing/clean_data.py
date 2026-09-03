import pandas as pd
import re
from pathlib import Path


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

INPUT_FILE = Path("data/raw/cpse_materials_raw.csv")
OUTPUT_FILE = Path("data/processed/cpse_materials_clean.csv")


# --------------------------------------------------
# 2. Text cleaning function
# --------------------------------------------------

def clean_text(text):
    """
    Clean and normalize material text.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Replace common symbols with spaces
    text = text.replace("×", " x ")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Replace ' by ' with ' x '
    text = re.sub(r"\bby\b", " x ", text)

    # Remove unnecessary punctuation
    text = re.sub(r"[^a-z0-9./+\- ]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# 3. Load raw dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} records.")


# --------------------------------------------------
# 4. Clean important text columns
# --------------------------------------------------

text_columns = [
    "material_description",
    "category",
    "material",
    "size",
    "unit",
    "grade"
]

for column in text_columns:
    df[f"{column}_clean"] = df[column].apply(clean_text)


# --------------------------------------------------
# 5. Create combined searchable text
# --------------------------------------------------

df["combined_text"] = (
    df["material_description_clean"]
    + " "
    + df["category_clean"]
    + " "
    + df["material_clean"]
    + " "
    + df["size_clean"]
    + " "
    + df["grade_clean"]
)


# --------------------------------------------------
# 6. Create output directory
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 7. Save processed dataset
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Cleaned dataset saved to: {OUTPUT_FILE}")

print("\nColumns in processed dataset:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())