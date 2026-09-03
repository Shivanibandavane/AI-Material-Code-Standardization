import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

MATCH_FILE = Path(
    "data/processed/material_matches.csv"
)

OUTPUT_FILE = Path(
    "data/processed/matching_evaluation.csv"
)


print("=" * 75)
print("AI MATERIAL CODE STANDARDIZATION")
print("MATCHING EVALUATION")
print("=" * 75)


# ============================================================
# 1. LOAD MATCH RESULTS
# ============================================================

print("\n[1/6] Loading AI matching results...")

df = pd.read_csv(MATCH_FILE)

print(f"Loaded {len(df)} match records.")


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[2/6] Checking required columns...")

required_columns = [
    "cpse_1",
    "material_code_1",
    "cpse_2",
    "material_code_2",
    "combined_score",
    "decision",
    "same_benchmark_group"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")

    for column in missing_columns:
        print(f"- {column}")

    raise ValueError(
        "Required columns are missing from material_matches.csv"
    )

print("All required columns are available.")


# ============================================================
# 3. CONVERT GROUND TRUTH TO BOOLEAN
# ============================================================

print("\n[3/6] Preparing ground-truth labels...")

df["actual_match"] = (
    df["same_benchmark_group"]
    .astype(str)
    .str.lower()
    .map({
        "true": True,
        "false": False
    })
)

df["predicted_match"] = (
    df["decision"]
    .isin([
        "HIGH CONFIDENCE",
        "REVIEW REQUIRED"
    ])
)

print("Ground-truth and prediction labels created.")


# ============================================================
# 4. CALCULATE CONFUSION MATRIX
# ============================================================

print("\n[4/6] Calculating evaluation metrics...")

true_positive = (
    (df["predicted_match"] == True)
    & (df["actual_match"] == True)
).sum()

true_negative = (
    (df["predicted_match"] == False)
    & (df["actual_match"] == False)
).sum()

false_positive = (
    (df["predicted_match"] == True)
    & (df["actual_match"] == False)
).sum()

false_negative = (
    (df["predicted_match"] == False)
    & (df["actual_match"] == True)
).sum()


# ============================================================
# 5. METRICS
# ============================================================

total = len(df)

accuracy = (
    (true_positive + true_negative) / total
    if total > 0
    else 0
)

precision = (
    true_positive / (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)

recall = (
    true_positive / (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)

f1_score = (
    2 * precision * recall
    / (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ============================================================
# HIGH CONFIDENCE EVALUATION
# ============================================================

high_confidence = df[
    df["decision"] == "HIGH CONFIDENCE"
]

if len(high_confidence) > 0:

    high_confidence_correct = (
        high_confidence["actual_match"]
        == True
    ).sum()

    high_confidence_precision = (
        high_confidence_correct
        / len(high_confidence)
    )

else:

    high_confidence_precision = 0


# ============================================================
# REVIEW REQUIRED EVALUATION
# ============================================================

review_required = df[
    df["decision"] == "REVIEW REQUIRED"
]

if len(review_required) > 0:

    review_correct = (
        review_required["actual_match"]
        == True
    ).sum()

    review_match_rate = (
        review_correct
        / len(review_required)
    )

else:

    review_match_rate = 0


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 75)
print("MATCHING EVALUATION RESULTS")
print("=" * 75)


print("\nCONFUSION MATRIX")
print("-" * 40)

print(f"True Positives  : {true_positive}")
print(f"True Negatives  : {true_negative}")
print(f"False Positives : {false_positive}")
print(f"False Negatives : {false_negative}")


print("\nOVERALL METRICS")
print("-" * 40)

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score  : {f1_score:.4f} ({f1_score * 100:.2f}%)")


print("\nCONFIDENCE ANALYSIS")
print("-" * 40)

print(
    f"High-confidence matches : "
    f"{len(high_confidence)}"
)

print(
    f"High-confidence precision: "
    f"{high_confidence_precision:.4f} "
    f"({high_confidence_precision * 100:.2f}%)"
)

print(
    f"Review-required matches  : "
    f"{len(review_required)}"
)

print(
    f"Review group actual-match rate: "
    f"{review_match_rate:.4f} "
    f"({review_match_rate * 100:.2f}%)"
)


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

evaluation_summary = pd.DataFrame([
    {
        "metric": "Total comparisons",
        "value": total
    },
    {
        "metric": "True positives",
        "value": true_positive
    },
    {
        "metric": "True negatives",
        "value": true_negative
    },
    {
        "metric": "False positives",
        "value": false_positive
    },
    {
        "metric": "False negatives",
        "value": false_negative
    },
    {
        "metric": "Accuracy",
        "value": round(accuracy, 4)
    },
    {
        "metric": "Precision",
        "value": round(precision, 4)
    },
    {
        "metric": "Recall",
        "value": round(recall, 4)
    },
    {
        "metric": "F1 Score",
        "value": round(f1_score, 4)
    },
    {
        "metric": "High-confidence precision",
        "value": round(
            high_confidence_precision,
            4
        )
    }
])


evaluation_summary.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nEvaluation summary saved to:")

print(OUTPUT_FILE)


print("\n" + "=" * 75)
print("MATCHING EVALUATION COMPLETED")
print("=" * 75)