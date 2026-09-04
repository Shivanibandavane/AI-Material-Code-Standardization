import pandas as pd
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = Path(
    "data/processed/material_matches.csv"
)

OUTPUT_FILE = Path(
    "data/processed/matching_evaluation_v2.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AI MATERIAL CODE STANDARDIZATION")
print("MATCHING EVALUATION V2")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\n[1/6] Loading match results...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} match records.")


# ============================================================
# REMOVE REVERSE DUPLICATES
# ============================================================

print("\n[2/6] Removing reverse duplicate pairs...")

# Create a unique pair independent of direction
df["pair_id"] = df.apply(
    lambda row: " | ".join(
        sorted([
            f"{row['cpse_1']}:{row['material_code_1']}",
            f"{row['cpse_2']}:{row['material_code_2']}"
        ])
    ),
    axis=1
)

df_unique = df.drop_duplicates(
    subset="pair_id"
).copy()

print(
    f"Unique material pairs: {len(df_unique)}"
)


# ============================================================
# GROUND TRUTH
# ============================================================

print("\n[3/6] Preparing ground truth...")

df_unique["actual_match"] = (
    df_unique["same_benchmark_group"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
)

print(
    f"Actual equivalent pairs: "
    f"{df_unique['actual_match'].sum()}"
)

print(
    f"Actual non-equivalent pairs: "
    f"{(~df_unique['actual_match']).sum()}"
)


# ============================================================
# PREDICTION
# ============================================================

print("\n[4/6] Evaluating AI predictions...")


# HIGH CONFIDENCE = AI recommends automatic match
df_unique["high_confidence_prediction"] = (
    df_unique["confidence"]
    == "HIGH CONFIDENCE"
)


# HIGH + REVIEW = AI considers it a possible match
df_unique["review_prediction"] = (
    df_unique["confidence"]
    .isin([
        "HIGH CONFIDENCE",
        "REVIEW REQUIRED"
    ])
)


# ============================================================
# FUNCTION FOR METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    tp = (
        (actual == True)
        & (predicted == True)
    ).sum()

    tn = (
        (actual == False)
        & (predicted == False)
    ).sum()

    fp = (
        (actual == False)
        & (predicted == True)
    ).sum()

    fn = (
        (actual == True)
        & (predicted == False)
    ).sum()

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ============================================================
# METRICS — HIGH CONFIDENCE ONLY
# ============================================================

high_metrics = calculate_metrics(
    df_unique["actual_match"],
    df_unique["high_confidence_prediction"]
)


# ============================================================
# METRICS — HIGH + REVIEW
# ============================================================

review_metrics = calculate_metrics(
    df_unique["actual_match"],
    df_unique["review_prediction"]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULT 1 — HIGH CONFIDENCE ONLY")
print("=" * 70)

print(
    f"True Positives : {high_metrics['TP']}"
)

print(
    f"True Negatives : {high_metrics['TN']}"
)

print(
    f"False Positives: {high_metrics['FP']}"
)

print(
    f"False Negatives: {high_metrics['FN']}"
)

print(
    f"\nAccuracy : "
    f"{high_metrics['Accuracy']:.4f}"
    f" ({high_metrics['Accuracy'] * 100:.2f}%)"
)

print(
    f"Precision: "
    f"{high_metrics['Precision']:.4f}"
    f" ({high_metrics['Precision'] * 100:.2f}%)"
)

print(
    f"Recall   : "
    f"{high_metrics['Recall']:.4f}"
    f" ({high_metrics['Recall'] * 100:.2f}%)"
)

print(
    f"F1 Score : "
    f"{high_metrics['F1']:.4f}"
    f" ({high_metrics['F1'] * 100:.2f}%)"
)


# ============================================================
# PRINT REVIEW METRICS
# ============================================================

print("\n" + "=" * 70)
print("RESULT 2 — HIGH CONFIDENCE + REVIEW REQUIRED")
print("=" * 70)

print(
    f"True Positives : {review_metrics['TP']}"
)

print(
    f"True Negatives : {review_metrics['TN']}"
)

print(
    f"False Positives: {review_metrics['FP']}"
)

print(
    f"False Negatives: {review_metrics['FN']}"
)

print(
    f"\nAccuracy : "
    f"{review_metrics['Accuracy']:.4f}"
    f" ({review_metrics['Accuracy'] * 100:.2f}%)"
)

print(
    f"Precision: "
    f"{review_metrics['Precision']:.4f}"
    f" ({review_metrics['Precision'] * 100:.2f}%)"
)

print(
    f"Recall   : "
    f"{review_metrics['Recall']:.4f}"
    f" ({review_metrics['Recall'] * 100:.2f}%)"
)

print(
    f"F1 Score : "
    f"{review_metrics['F1']:.4f}"
    f" ({review_metrics['F1'] * 100:.2f}%)"
)


# ============================================================
# CONFIDENCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE DISTRIBUTION")
print("=" * 70)

print(
    df_unique["confidence"]
    .value_counts()
)


# ============================================================
# HIGH-CONFIDENCE PRECISION
# ============================================================

high_confidence_df = df_unique[
    df_unique["high_confidence_prediction"]
]

if len(high_confidence_df) > 0:

    high_confidence_precision = (
        high_confidence_df["actual_match"]
        .mean()
    )

else:

    high_confidence_precision = 0


print(
    f"\nHigh-confidence matches: "
    f"{len(high_confidence_df)}"
)

print(
    f"High-confidence precision: "
    f"{high_confidence_precision:.4f}"
    f" ({high_confidence_precision * 100:.2f}%)"
)


# ============================================================
# REVIEW GROUP
# ============================================================

review_df = df_unique[
    df_unique["confidence"]
    == "REVIEW REQUIRED"
]

if len(review_df) > 0:

    review_match_rate = (
        review_df["actual_match"]
        .mean()
    )

else:

    review_match_rate = 0


print(
    f"\nReview-required pairs: "
    f"{len(review_df)}"
)

print(
    f"Review group actual-match rate: "
    f"{review_match_rate:.4f}"
    f" ({review_match_rate * 100:.2f}%)"
)


# ============================================================
# SAVE EVALUATION
# ============================================================

print("\n[5/6] Saving evaluation results...")

evaluation_summary = pd.DataFrame([
    {
        "prediction_type":
            "HIGH CONFIDENCE ONLY",
        **high_metrics
    },
    {
        "prediction_type":
            "HIGH + REVIEW",
        **review_metrics
    }
])

evaluation_summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Saved evaluation to: "
    f"{OUTPUT_FILE}"
)


# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\n[6/6] Showing high-confidence examples...")

print("\nTop high-confidence matches:")

examples = df_unique[
    df_unique["confidence"]
    == "HIGH CONFIDENCE"
].head(10)

for _, row in examples.iterrows():

    actual = (
        "CORRECT"
        if row["actual_match"]
        else "INCORRECT"
    )

    print(
        f"\n{row['cpse_1']} "
        f"{row['material_code_1']} "
        f"{row['material_description_1']}"
    )

    print(
        f"  ↔ {row['cpse_2']} "
        f"{row['material_code_2']} "
        f"{row['material_description_2']}"
    )

    print(
        f"  Score: {row['hybrid_score']:.4f}"
    )

    print(
        f"  Evaluation: {actual}"
    )


print("\n" + "=" * 70)
print("MATCHING EVALUATION V2 COMPLETED")
print("=" * 70)