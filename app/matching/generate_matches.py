import pandas as pd
import numpy as np
import re

from pathlib import Path
from difflib import SequenceMatcher

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/cpse_materials_attributes.csv"
)

OUTPUT_FILE = Path(
    "data/processed/material_matches.csv"
)

print("=" * 75)
print("AI MATERIAL CODE STANDARDIZATION")
print("FULL CROSS-CPSE MATCHING ENGINE")
print("=" * 75)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/8] Loading material dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} material records.")
print(f"CPSEs found: {df['cpse'].unique().tolist()}")


# ============================================================
# 2. NORMALIZATION FUNCTIONS
# ============================================================

def normalize_material(value):

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

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = re.sub(r"\bm(\d+)", r"\1", value)

    value = value.replace("-", " x ")
    value = value.replace("×", " x ")

    value = re.sub(r"\s*x\s*", " x ", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_grade(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = value.replace(" ", "")
    value = value.replace("-", "")

    value = value.replace("ss", "")

    return value


def normalize_category(value):

    if pd.isna(value):
        return ""

    return str(value).lower().strip()


# ============================================================
# 3. NORMALIZE ATTRIBUTES
# ============================================================

print("\n[2/8] Normalizing material attributes...")

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
# 4. LOAD AI MODEL
# ============================================================

print("\n[3/8] Loading AI embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("AI model loaded successfully.")


# ============================================================
# 5. CREATE EMBEDDINGS
# ============================================================

print("\n[4/8] Creating semantic embeddings...")

descriptions = (
    df["material_description_clean"]
    .fillna("")
    .tolist()
)

embeddings = model.encode(
    descriptions,
    show_progress_bar=True
)

print(
    f"Embedding shape: {embeddings.shape}"
)


# ============================================================
# 6. SIMILARITY FUNCTIONS
# ============================================================

def attribute_similarity(row1, row2):

    scores = []

    # Category
    if (
        row1["category_normalized"]
        and row2["category_normalized"]
    ):
        scores.append(
            1.0
            if row1["category_normalized"]
            == row2["category_normalized"]
            else 0.0
        )

    # Material
    if (
        row1["material_normalized"]
        and row2["material_normalized"]
    ):
        scores.append(
            1.0
            if row1["material_normalized"]
            == row2["material_normalized"]
            else 0.0
        )

    # Size
    if (
        row1["size_normalized"]
        and row2["size_normalized"]
    ):
        scores.append(
            1.0
            if row1["size_normalized"]
            == row2["size_normalized"]
            else 0.0
        )

    # Grade
    if (
        row1["grade_normalized"]
        and row2["grade_normalized"]
    ):
        scores.append(
            1.0
            if row1["grade_normalized"]
            == row2["grade_normalized"]
            else 0.0
        )

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def lexical_similarity(text1, text2):

    text1 = str(text1).lower()
    text2 = str(text2).lower()

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


def get_decision(score):

    if score >= 0.80:
        return "HIGH CONFIDENCE"

    elif score >= 0.65:
        return "REVIEW REQUIRED"

    else:
        return "LOW CONFIDENCE"


def generate_reason(
    row1,
    row2,
    semantic_score,
    attribute_score,
    lexical_score
):

    reasons = []

    if (
        row1["category_normalized"]
        == row2["category_normalized"]
        and row1["category_normalized"]
    ):
        reasons.append("Same category")

    if (
        row1["material_normalized"]
        == row2["material_normalized"]
        and row1["material_normalized"]
    ):
        reasons.append("Same material")

    if (
        row1["size_normalized"]
        == row2["size_normalized"]
        and row1["size_normalized"]
    ):
        reasons.append("Same size")

    if (
        row1["grade_normalized"]
        == row2["grade_normalized"]
        and row1["grade_normalized"]
    ):
        reasons.append("Same grade")

    if semantic_score >= 0.70:
        reasons.append("Strong semantic similarity")

    elif semantic_score >= 0.50:
        reasons.append("Moderate semantic similarity")

    if lexical_score >= 0.70:
        reasons.append("Similar wording")

    if not reasons:
        reasons.append("Limited similarity")

    return "; ".join(reasons)


# ============================================================
# 7. GENERATE CROSS-CPSE MATCHES
# ============================================================

print("\n[5/8] Generating cross-CPSE material matches...")

results = []

total_comparisons = 0

for i in range(len(df)):

    row1 = df.iloc[i]

    for j in range(len(df)):

        row2 = df.iloc[j]

        # Do not compare a material with itself
        if i == j:
            continue

        # Only compare materials belonging to different CPSEs
        if row1["cpse"] == row2["cpse"]:
            continue

        # Candidate filtering:
        # Materials from different categories are less likely
        # to represent the same material.
        if (
            row1["category_normalized"]
            != row2["category_normalized"]
        ):
            continue

        total_comparisons += 1

        # Semantic similarity
        semantic_score = cosine_similarity(
            [embeddings[i]],
            [embeddings[j]]
        )[0][0]

        # Attribute similarity
        attribute_score = attribute_similarity(
            row1,
            row2
        )

        # Lexical similarity
        lexical_score = lexical_similarity(
            row1["material_description_clean"],
            row2["material_description_clean"]
        )

        # Hybrid score
        combined_score = (
            0.50 * semantic_score
            + 0.35 * attribute_score
            + 0.15 * lexical_score
        )

        decision = get_decision(
            combined_score
        )

        reason = generate_reason(
            row1,
            row2,
            semantic_score,
            attribute_score,
            lexical_score
        )

        results.append({

            "cpse_1":
                row1["cpse"],

            "material_code_1":
                row1["material_code"],

            "description_1":
                row1["material_description"],

            "cpse_2":
                row2["cpse"],

            "material_code_2":
                row2["material_code"],

            "description_2":
                row2["material_description"],

            "category":
                row1["category"],

            "semantic_score":
                round(
                    semantic_score,
                    4
                ),

            "attribute_score":
                round(
                    attribute_score,
                    4
                ),

            "lexical_score":
                round(
                    lexical_score,
                    4
                ),

            "combined_score":
                round(
                    combined_score,
                    4
                ),

            "decision":
                decision,

            "reason":
                reason,

            # Ground truth is stored ONLY for evaluation.
            # It is NOT used by the AI for prediction.
            "same_benchmark_group":
                row1["benchmark_group_id"]
                == row2["benchmark_group_id"]
        })


print(
    f"Total cross-CPSE comparisons: "
    f"{total_comparisons}"
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

print("\n[6/8] Creating results dataframe...")

results_df = pd.DataFrame(results)

# Sort best matches first
results_df = results_df.sort_values(
    by="combined_score",
    ascending=False
)

results_df = results_df.reset_index(
    drop=True
)

print(
    f"Generated {len(results_df)} match records."
)


print("\n[7/8] Saving match results...")

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Saved results to:\n{OUTPUT_FILE}"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n[8/8] MATCHING SUMMARY")

print("=" * 75)

print(
    "\nConfidence distribution:"
)

print(
    results_df["decision"]
    .value_counts()
)


print(
    "\nTop 10 AI-generated matches:"
)

display_columns = [
    "cpse_1",
    "material_code_1",
    "description_1",
    "cpse_2",
    "material_code_2",
    "description_2",
    "combined_score",
    "decision"
]

print(
    results_df[
        display_columns
    ].head(10).to_string(
        index=False
    )
)


print("\n" + "=" * 75)
print("FULL MATCHING ENGINE COMPLETED")
print("=" * 75)