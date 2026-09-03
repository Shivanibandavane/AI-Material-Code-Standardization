import pandas as pd
import re
from pathlib import Path


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

INPUT_FILE = Path("data/processed/cpse_materials_clean.csv")
OUTPUT_FILE = Path("data/processed/cpse_materials_attributes.csv")


# --------------------------------------------------
# 2. Extract size information
# --------------------------------------------------

def extract_size(text):
    """
    Extract size/dimension information from material text.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Examples:
    # m10 x 50
    # 10 x 50
    # 50 nb
    # 6205
    # 5 hp
    # 3.15 mm

    patterns = [
        r'\bm\d+\s*x\s*\d+\b',
        r'\b\d+\s*x\s*\d+\b',
        r'\b\d+\s*nb\b',
        r'\b\d+\s*mm\b',
        r'\b\d+\s*hp\b',
        r'\b\d+\s*m3\s*/?\s*hr\b',
        r'\b\d{4,5}\b'
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(0)

    return ""


# --------------------------------------------------
# 3. Extract material type
# --------------------------------------------------

def extract_material_type(text):
    """
    Identify common material types.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    material_keywords = {
        "stainless steel": [
            "stainless steel",
            "ss"
        ],
        "mild steel": [
            "mild steel",
            "ms"
        ],
        "carbon steel": [
            "carbon steel",
            "cs"
        ],
        "galvanized steel": [
            "galvanized steel",
            "galvanised steel",
            "gi"
        ],
        "copper": [
            "copper",
            "cu"
        ],
        "aluminium": [
            "aluminium",
            "aluminum",
            "al"
        ],
        "cast iron": [
            "cast iron",
            "ci"
        ],
        "ductile iron": [
            "ductile iron",
            "di"
        ]
    }

    for material, keywords in material_keywords.items():

        for keyword in keywords:

            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                return material

    return ""


# --------------------------------------------------
# 4. Extract grade
# --------------------------------------------------

def extract_grade(text):
    """
    Extract common material/specification grades.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    grade_patterns = [
        r'\bss\s*304\b',
        r'\bss\s*316\b',
        r'\b304\b',
        r'\b316\b',
        r'\bie2\b',
        r'\bie3\b',
        r'\bepdm\b',
        r'\bpn\s*16\b',
        r'\bclass\s*150\b',
        r'\bcl\s*150\b',
        r'\bsch\s*\d+\b',
        r'\bschedule\s*\d+\b',
        r'\be6013\b',
        r'\be308l(?:-\d+)?\b',
        r'\bep\s*2\b',
        r'\bnlgi\s*[- ]?\d+\b'
    ]

    for pattern in grade_patterns:

        match = re.search(pattern, text)

        if match:
            return match.group(0)

    return ""


# --------------------------------------------------
# 5. Load processed dataset
# --------------------------------------------------

print("Loading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} records.")


# --------------------------------------------------
# 6. Extract attributes
# --------------------------------------------------

print("Extracting attributes...")

df["extracted_size"] = df["combined_text"].apply(extract_size)

df["extracted_material"] = df["combined_text"].apply(
    extract_material_type
)

df["extracted_grade"] = df["combined_text"].apply(
    extract_grade
)


# --------------------------------------------------
# 7. Save attribute dataset
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print(
    f"Attribute dataset saved to: {OUTPUT_FILE}"
)

print("\nExtracted attribute columns:")

print(
    df[
        [
            "material_description",
            "extracted_size",
            "extracted_material",
            "extracted_grade"
        ]
    ].head(10)
)

print("\nTotal records:", len(df))

print("\nAttribute extraction completed successfully!")