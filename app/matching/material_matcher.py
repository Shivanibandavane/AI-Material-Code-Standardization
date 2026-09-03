import pandas as pd
from pathlib import Path
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher


# ============================================================
# 1. FILE PATH
# ============================================================

INPUT_FILE = Path(
    "data/processed/cpse_materials_attributes.csv"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 65)
print("AI MATERIAL CODE STANDARDIZATION - MATCHING ENGINE")
print("=" * 65)

print("\n[1/7] Loading material dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} material records.")


# ============================================================
# 3. NORMALIZATION FUNCTIONS
# ============================================================

def normalize_material(value):
    """
    Normalize different names for the same material.
    """

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    replacements = {
        "ss": "stainless steel",
        "stainless": "stainless steel",
        "ms": "mild steel",
        "cs": "carbon steel",
        "gi": "galvanized steel",
        "cu": "copper",
        "al": "aluminium",
        "aluminum": "aluminium",
        "ci": "cast iron",
        "di": "ductile iron"
    }

    for old, new in replacements.items():

        if value == old:
            value = new

    return value


def normalize_size(value):
    """
    Normalize common size formats.

    Examples:
    M10 x 50
    10 x 50
    M10-50

    become a comparable representation.
    """

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Replace M prefix
    value = re.sub(r'\bm(\d+)', r'\1', value)

    # Replace separators
    value = value.replace("-", " x ")
    value = value.replace("×", " x ")

    # Normalize spaces around x
    value = re.sub(r'\s*x\s*', ' x ', value)

    # Remove unnecessary spaces
    value = re.sub(r'\s+', ' ', value)

    return value.strip()


def normalize_grade(value):
    """
    Normalize common grade formats.

    Examples:
    SS304
    SS 304
    304

    become:
    304
    """

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = value.replace(" ", "")
    value = value.replace("-", "")

    # Remove common material prefixes
    value = value.replace("ss", "")

    return value


def normalize_category(value):
    """
    Normalize category names.
    """

    if pd.isna(value):
        return ""

    return str(value).lower().strip()


# ============================================================
# 4. CREATE NORMALIZED ATTRIBUTES
# ============================================================

print("\n[2/7] Normalizing material attributes...")

df["material_normalized"] = (
    df["extracted_material"]
    .apply(normalize_material)
)

df["size_normalized"] = (
    df["extracted_size"]
    .apply(normalize_size)
)

df["grade_normalized"] = (
    df["extracted_grade"]
    .apply(normalize_grade)
)

df["category_normalized"] = (
    df["category"]
    .apply(normalize_category)
)


print("Attribute normalization completed.")


# ============================================================
# 5. LOAD AI EMBEDDING MODEL
# ============================================================

print("\n[3/7] Loading AI embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("AI model loaded successfully.")


# ============================================================
# 6. CREATE SEMANTIC EMBEDDINGS
# ============================================================

print("\n[4/7] Creating semantic embeddings...")

descriptions = (
    df["material_description_clean"]
    .fillna("")
    .tolist()
)

embeddings = model.encode(
    descriptions,
    show_progress_bar=True
)

print("Embeddings created successfully.")
print(f"Embedding shape: {embeddings.shape}")


# ============================================================
# 7. ATTRIBUTE SIMILARITY
# ============================================================

def attribute_similarity(row1, row2):

    scores = []

    # ----------------------------------------
    # Category
    # ----------------------------------------

    category1 = row1["category_normalized"]
    category2 = row2["category_normalized"]

    if category1 and category2:

        if category1 == category2:
            scores.append(1.0)

        else:
            scores.append(0.0)

    # ----------------------------------------
    # Material
    # ----------------------------------------

    material1 = row1["material_normalized"]
    material2 = row2["material_normalized"]

    if material1 and material2:

        if material1 == material2:
            scores.append(1.0)

        else:
            scores.append(0.0)

    # ----------------------------------------
    # Size
    # ----------------------------------------

    size1 = row1["size_normalized"]
    size2 = row2["size_normalized"]

    if size1 and size2:

        if size1 == size2:
            scores.append(1.0)

        else:
            scores.append(0.0)

    # ----------------------------------------
    # Grade
    # ----------------------------------------

    grade1 = row1["grade_normalized"]
    grade2 = row2["grade_normalized"]

    if grade1 and grade2:

        if grade1 == grade2:
            scores.append(1.0)

        else:
            scores.append(0.0)

    # ----------------------------------------
    # Final attribute score
    # ----------------------------------------

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# ============================================================
# 8. LEXICAL SIMILARITY
# ============================================================

def lexical_similarity(text1, text2):

    text1 = str(text1).lower()
    text2 = str(text2).lower()

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


# ============================================================
# 9. TEST FIRST TWO MATERIALS
# ============================================================

print("\n" + "=" * 65)
print("HYBRID MATCHING TEST")
print("=" * 65)


row1 = df.iloc[0]
row2 = df.iloc[1]


# ------------------------------------------------------------
# Semantic score
# ------------------------------------------------------------

semantic_score = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)[0][0]


# ------------------------------------------------------------
# Attribute score
# ------------------------------------------------------------

attribute_score = attribute_similarity(
    row1,
    row2
)


# ------------------------------------------------------------
# Lexical score
# ------------------------------------------------------------

lexical_score = lexical_similarity(
    row1["material_description_clean"],
    row2["material_description_clean"]
)


# ------------------------------------------------------------
# Combined score
# ------------------------------------------------------------

combined_score = (
    0.50 * semantic_score
    + 0.35 * attribute_score
    + 0.15 * lexical_score
)


# ============================================================
# 10. DISPLAY MATERIALS
# ============================================================

print("\nMaterial 1:")
print(row1["material_description"])

print("\nMaterial 2:")
print(row2["material_description"])


# ============================================================
# 11. DISPLAY ATTRIBUTES
# ============================================================

print("\n--- Normalized Attributes ---")

print(
    f"Material 1 → "
    f"{row1['material_normalized']}"
)

print(
    f"Material 2 → "
    f"{row2['material_normalized']}"
)

print(
    f"\nSize 1 → "
    f"{row1['size_normalized']}"
)

print(
    f"Size 2 → "
    f"{row2['size_normalized']}"
)

print(
    f"\nGrade 1 → "
    f"{row1['grade_normalized']}"
)

print(
    f"Grade 2 → "
    f"{row2['grade_normalized']}"
)


# ============================================================
# 12. DISPLAY SCORES
# ============================================================

print("\n--- Similarity Results ---")

print(
    f"Semantic similarity : "
    f"{semantic_score:.4f}"
)

print(
    f"Attribute similarity: "
    f"{attribute_score:.4f}"
)

print(
    f"Lexical similarity  : "
    f"{lexical_score:.4f}"
)

print(
    f"Combined score      : "
    f"{combined_score:.4f}"
)


# ============================================================
# 13. DECISION
# ============================================================

if combined_score >= 0.80:

    decision = "HIGH CONFIDENCE MATCH"

elif combined_score >= 0.65:

    decision = "REVIEW REQUIRED"

else:

    decision = "LOW CONFIDENCE"


print(
    f"\nAI Decision: {decision}"
)


# ============================================================
# 14. COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("HYBRID MATCHING TEST COMPLETED")
print("=" * 65)