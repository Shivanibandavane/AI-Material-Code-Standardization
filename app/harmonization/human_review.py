import pandas as pd
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

MASTER_FILE = "data/processed/harmonized_material_master.csv"

OUTPUT_FILE = "data/processed/human_review_decisions.csv"


# ============================================================
# LOAD CANONICAL MASTER
# ============================================================

print("=" * 70)
print("AI MATERIAL CODE STANDARDIZATION")
print("HUMAN REVIEW QUEUE")
print("=" * 70)

print("\nLoading harmonized material master...")

master_df = pd.read_csv(MASTER_FILE)

print(f"Total canonical groups: {len(master_df)}")


# ============================================================
# SELECT GROUPS REQUIRING HUMAN REVIEW
# ============================================================

review_df = master_df[
    master_df["source_material_count"] > 1
].copy()

print(
    f"Groups requiring human review: {len(review_df)}"
)


# ============================================================
# CREATE REVIEW QUEUE
# ============================================================

review_records = []

for _, row in review_df.iterrows():

    review_records.append({

        "prototype_canonical_material_code":
            row["prototype_canonical_material_code"],

        "canonical_category":
            row["canonical_category"],

        "canonical_material":
            row["canonical_material"],

        "canonical_size":
            row["canonical_size"],

        "canonical_unit":
            row["canonical_unit"],

        "canonical_grade":
            row["canonical_grade"],

        "source_material_count":
            row["source_material_count"],

        "source_cpse_count":
            row["source_cpse_count"],

        "source_cpses":
            row["source_cpses"],

        "source_material_codes":
            row["source_material_codes"],

        "source_descriptions":
            row["source_descriptions"],

        "ai_recommendation":
            row["ai_recommendation"],

        # Human has not reviewed it yet
        "human_decision":
            "PENDING",

        "reviewer":
            "",

        "review_timestamp":
            "",

        "review_comment":
            ""
    })


# ============================================================
# SAVE REVIEW QUEUE
# ============================================================

decisions_df = pd.DataFrame(review_records)

Path(OUTPUT_FILE).parent.mkdir(
    parents=True,
    exist_ok=True
)

decisions_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("HUMAN REVIEW QUEUE CREATED")
print("=" * 70)

print(
    f"\nTotal canonical groups : {len(master_df)}"
)

print(
    f"Groups for human review: {len(decisions_df)}"
)

print(
    f"Pending decisions      : "
    f"{len(decisions_df)}"
)

print(
    f"\nReview file created:"
)

print(
    f"{OUTPUT_FILE}"
)


# ============================================================
# SHOW FIRST FEW GROUPS
# ============================================================

if len(decisions_df) > 0:

    print("\nFirst 5 groups waiting for review:\n")

    print(
        decisions_df[
            [
                "prototype_canonical_material_code",
                "canonical_category",
                "canonical_material",
                "canonical_size",
                "canonical_grade",
                "source_material_count",
                "source_cpses",
                "human_decision"
            ]
        ].head(5).to_string(index=False)
    )

else:

    print("\nNo groups currently require human review.")


print("\n" + "=" * 70)
print("READY FOR FRONTEND HUMAN REVIEW")
print("=" * 70)