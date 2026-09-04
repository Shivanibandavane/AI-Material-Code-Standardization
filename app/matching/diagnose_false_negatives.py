import pandas as pd
from pathlib import Path


# ============================================================
# FILE PATH
# ============================================================

INPUT_FILE = Path(
    "data/processed/material_matches.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AI MATERIAL CODE STANDARDIZATION")
print("FALSE NEGATIVE DIAGNOSTIC")
print("=" * 70)


# ============================================================
# LOAD MATCH RESULTS
# ============================================================

print("\n[1/5] Loading match results...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} match records.")


# ============================================================
# REMOVE REVERSE DUPLICATES
# ============================================================

print("\n[2/5] Removing reverse duplicate pairs...")

df["pair_id"] = df.apply(
    lambda row: " | ".join(
        sorted([
            f"{row['cpse_1']}:{row['material_code_1']}",
            f"{row['cpse_2']}:{row['material_code_2']}"
        ])
    ),
    axis=1
)

df = df.drop_duplicates(
    subset="pair_id"
).copy()

print(
    f"Unique pairs: {len(df)}"
)


# ============================================================
# FIND ACTUAL MATCHES MISSED BY AI
# ============================================================

print("\n[3/5] Finding missed genuine matches...")

false_negatives = df[
    (
        df["same_benchmark_group"]
        == True
    )
    &
    (
        df["confidence"]
        == "LOW CONFIDENCE"
    )
].copy()

print(
    f"Actual equivalent pairs missed by AI: "
    f"{len(false_negatives)}"
)


# ============================================================
# SHOW MISSED MATCHES
# ============================================================

print("\n[4/5] Showing missed genuine matches...")

if len(false_negatives) == 0:

    print(
        "\nExcellent! No false negatives found."
    )

else:

    # Lowest scoring missed matches first
    false_negatives = false_negatives.sort_values(
        by="combined_score",
        ascending=True
    )

    print("\n" + "-" * 70)

    for index, (_, row) in enumerate(
        false_negatives.head(30).iterrows(),
        start=1
    ):

        print(f"\n#{index}")

        print(
            f"{row['cpse_1']} "
            f"{row['material_code_1']} "
            f"→ {row['description_1']}"
        )

        print(
            f"{row['cpse_2']} "
            f"{row['material_code_2']} "
            f"→ {row['description_2']}"
        )

        print(
            f"\nSemantic score : "
            f"{row['semantic_score']:.4f}"
        )

        print(
            f"Attribute score: "
            f"{row['attribute_score']:.4f}"
        )

        print(
            f"Lexical score  : "
            f"{row['lexical_score']:.4f}"
        )

        print(
            f"Conflict count : "
            f"{row['conflict_count']}"
        )

        print(
            f"Combined score : "
            f"{row['combined_score']:.4f}"
        )

        print(
            f"Confidence     : "
            f"{row['confidence']}"
        )

        print(
            f"Explanation    : "
            f"{row['explanation']}"
        )

        print("-" * 70)


# ============================================================
# SCORE DISTRIBUTION OF TRUE MATCHES
# ============================================================

print("\n[5/5] Analysing actual equivalent pairs...")

actual_matches = df[
    df["same_benchmark_group"]
    == True
].copy()

print(
    f"\nTotal actual equivalent pairs: "
    f"{len(actual_matches)}"
)

print(
    "\nConfidence distribution among actual equivalent pairs:"
)

print(
    actual_matches["confidence"]
    .value_counts()
)

print(
    "\nScore statistics for actual equivalent pairs:"
)

print(
    actual_matches["combined_score"]
    .describe()
)


# ============================================================
# SAVE FALSE NEGATIVES
# ============================================================

OUTPUT_FILE = Path(
    "data/processed/false_negative_diagnostics.csv"
)

false_negatives.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nDiagnostic results saved to:"
    f"\n{OUTPUT_FILE}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FALSE NEGATIVE DIAGNOSTIC COMPLETED")
print("=" * 70)