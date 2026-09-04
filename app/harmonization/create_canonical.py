import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

MATCHES_FILE = "data/processed/material_matches.csv"
MATERIALS_FILE = "data/processed/cpse_materials_attributes.csv"

OUTPUT_MASTER = "data/processed/harmonized_material_master.csv"
OUTPUT_REVIEW = "data/processed/harmonization_review_queue.csv"


# Category abbreviations for prototype canonical codes
CATEGORY_CODES = {
    "fastener": "FAST",
    "bearing": "BEAR",
    "valve": "VALV",
    "cable": "CABL",
    "pipe": "PIPE",
    "gasket": "GASK",
    "motor": "MOTR",
    "pump": "PUMP",
    "lubricant": "LUBE",
    "welding": "WELD",
    "ppe": "PPE",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """Convert missing values to empty strings."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_mode(values):
    """Return the most common non-empty value."""
    cleaned = [clean_value(v) for v in values]
    cleaned = [v for v in cleaned if v]

    if not cleaned:
        return ""

    return Counter(cleaned).most_common(1)[0][0]


def get_category_code(category):
    """Return short category code."""
    category = clean_value(category).lower()

    if category in CATEGORY_CODES:
        return CATEGORY_CODES[category]

    # Fallback for unexpected categories
    return category[:4].upper() if category else "MAT"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AI MATERIAL CODE STANDARDIZATION")
print("CANONICAL MATERIAL GROUP CREATION")
print("=" * 70)

print("\nLoading material data...")

materials = pd.read_csv(MATERIALS_FILE)
matches = pd.read_csv(MATCHES_FILE)

print(f"Materials loaded : {len(materials)}")
print(f"Matches loaded   : {len(matches)}")


# ============================================================
# CREATE UNIQUE MATERIAL IDENTIFIERS
# ============================================================

materials["material_id"] = (
    materials["cpse"].astype(str)
    + "||"
    + materials["material_code"].astype(str)
)


# ============================================================
# BUILD GRAPH OF HIGH-CONFIDENCE MATCHES
# ============================================================

print("\nBuilding high-confidence material groups...")

# Each material initially belongs to its own group
parent = {
    material_id: material_id
    for material_id in materials["material_id"]
}


def find(x):
    """Find root of a group."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]

    return x


def union(a, b):
    """Merge two material groups."""
    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


high_confidence_count = 0
used_match_count = 0


for _, row in matches.iterrows():

    confidence = clean_value(row.get("confidence", "")).upper()

    if confidence != "HIGH CONFIDENCE":
       continue

    # We use only HIGH confidence matches
    # without detected conflicts for automatic grouping.
    conflicts = clean_value(row.get("conflicts", ""))

    if conflicts.lower() not in ["", "none", "nan"]:
        continue

    cpse_1 = clean_value(row.get("cpse_1", ""))
    code_1 = clean_value(row.get("material_code_1", ""))

    cpse_2 = clean_value(row.get("cpse_2", ""))
    code_2 = clean_value(row.get("material_code_2", ""))

    material_1 = f"{cpse_1}||{code_1}"
    material_2 = f"{cpse_2}||{code_2}"

    if material_1 in parent and material_2 in parent:
        union(material_1, material_2)
        used_match_count += 1

    high_confidence_count += 1


print(f"HIGH confidence matches : {high_confidence_count}")
print(f"Used for auto-grouping  : {used_match_count}")


# ============================================================
# CREATE GROUPS
# ============================================================

groups = defaultdict(list)

for material_id in materials["material_id"]:
    root = find(material_id)
    groups[root].append(material_id)


print(f"Canonical groups created : {len(groups)}")


# ============================================================
# MAP MATERIAL ID TO ORIGINAL ROW
# ============================================================

material_lookup = {}

for _, row in materials.iterrows():

    material_id = row["material_id"]

    material_lookup[material_id] = row


# ============================================================
# CREATE CANONICAL MASTER
# ============================================================

master_records = []
review_records = []


# Sort groups to make output deterministic
sorted_groups = sorted(
    groups.values(),
    key=lambda x: sorted(x)[0]
)


canonical_counter = 1


for group_members in sorted_groups:

    rows = [
        material_lookup[material_id]
        for material_id in group_members
    ]

    categories = [row.get("category", "") for row in rows]
    materials_list = [row.get("material", "") for row in rows]
    sizes = [row.get("size", "") for row in rows]
    units = [row.get("unit", "") for row in rows]
    grades = [row.get("grade", "") for row in rows]

    # Determine canonical attributes using the most common value
    canonical_category = get_mode(categories)
    canonical_material = get_mode(materials_list)
    canonical_size = get_mode(sizes)
    canonical_unit = get_mode(units)
    canonical_grade = get_mode(grades)

    category_code = get_category_code(canonical_category)

    prototype_code = (
        f"PCM-{category_code}-{canonical_counter:03d}"
    )

    cpse_list = sorted(
        set(
            clean_value(row.get("cpse", ""))
            for row in rows
        )
    )

    source_codes = [
        f"{clean_value(row.get('cpse', ''))}:{clean_value(row.get('material_code', ''))}"
        for row in rows
    ]

    source_descriptions = [
        clean_value(row.get("material_description", ""))
        for row in rows
    ]

    # --------------------------------------------------------
    # AI recommendation
    # --------------------------------------------------------

    if len(group_members) > 1:

        ai_recommendation = (
            "AI recommends harmonizing these materials "
            "into one canonical group."
        )

        review_status = "PENDING_HUMAN_APPROVAL"

    else:

        ai_recommendation = (
            "No high-confidence cross-CPSE match found. "
            "Material remains as an individual group."
        )

        review_status = "PENDING_HUMAN_REVIEW"

    # --------------------------------------------------------
    # MASTER RECORD
    # --------------------------------------------------------

    master_records.append({

        "prototype_canonical_material_code":
            prototype_code,

        "canonical_category":
            canonical_category,

        "canonical_material":
            canonical_material,

        "canonical_size":
            canonical_size,

        "canonical_unit":
            canonical_unit,

        "canonical_grade":
            canonical_grade,

        "source_material_count":
            len(group_members),

        "source_cpse_count":
            len(cpse_list),

        "source_cpses":
            ", ".join(cpse_list),

        "source_material_codes":
            " | ".join(source_codes),

        "source_descriptions":
            " | ".join(source_descriptions),

        "ai_recommendation":
            ai_recommendation,

        "human_review_status":
            review_status,
    })

    # --------------------------------------------------------
    # REVIEW QUEUE
    # --------------------------------------------------------

    if len(group_members) > 1:

        review_records.append({

            "prototype_canonical_material_code":
                prototype_code,

            "source_material_count":
                len(group_members),

            "source_cpses":
                ", ".join(cpse_list),

            "source_material_codes":
                " | ".join(source_codes),

            "source_descriptions":
                " | ".join(source_descriptions),

            "ai_recommendation":
                "REVIEW - AI suggests these materials "
                "represent the same material.",

            "human_decision":
                "PENDING",

        })


    canonical_counter += 1


# ============================================================
# SAVE OUTPUT
# ============================================================

master_df = pd.DataFrame(master_records)

review_df = pd.DataFrame(review_records)


Path(OUTPUT_MASTER).parent.mkdir(
    parents=True,
    exist_ok=True
)


master_df.to_csv(
    OUTPUT_MASTER,
    index=False
)


review_df.to_csv(
    OUTPUT_REVIEW,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("HARMONIZATION COMPLETE")
print("=" * 70)

print(f"\nTotal source materials : {len(materials)}")
print(f"Canonical groups       : {len(master_df)}")
print(f"Groups needing review  : {len(review_df)}")

print("\nOutput files created:")

print(f"1. {OUTPUT_MASTER}")
print(f"2. {OUTPUT_REVIEW}")

print("\nSample canonical records:")
print(
    master_df[
        [
            "prototype_canonical_material_code",
            "canonical_category",
            "canonical_material",
            "canonical_size",
            "canonical_grade",
            "source_material_count",
            "human_review_status",
        ]
    ].head(10).to_string(index=False)
)

print("\nDone!")