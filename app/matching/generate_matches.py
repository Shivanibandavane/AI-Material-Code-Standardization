import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

INPUT_FILE = "data/processed/cpse_materials_attributes.csv"
OUTPUT_FILE = "data/processed/material_matches.csv"

print("=" * 70)
print("AI MATERIAL MATCHING ENGINE - V4")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)
print(f"\nLoaded records: {len(df)}")


def norm(value):
    if pd.isna(value):
        return ""
    value = str(value).lower().strip()
    value = value.replace("×", "x").replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value)
    return value


def material(value):
    value = norm(value)

    mapping = {
        "ss": "stainless steel",
        "stainless": "stainless steel",
        "ms": "mild steel",
        "mild": "mild steel",
        "cs": "carbon steel",
        "carbon": "carbon steel",
        "gi": "galvanized steel",
        "galvanized iron": "galvanized steel",
        "galvanised iron": "galvanized steel",
        "galvanised": "galvanized steel",
        "galvanized": "galvanized steel",
        "cu": "copper",
        "al": "aluminium",
        "aluminum": "aluminium",
        "ci": "cast iron",
        "di": "ductile iron"
    }

    for old, new in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
        value = re.sub(rf"\b{re.escape(old)}\b", new, value)

    return re.sub(r"\s+", " ", value).strip()


def size(value):
    value = norm(value)

    value = re.sub(r"\bdn\s*(\d+)\b", r"\1 nb", value)
    value = re.sub(r"\b(\d+)\s*nb\b", r"\1 nb", value)

    value = re.sub(
        r"\bm\s*(\d+)\s*[-x]\s*(\d+)\b",
        r"\1 x \2",
        value
    )

    value = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*mm\s*x\s*(\d+(?:\.\d+)?)\s*mm\b",
        r"\1 x \2",
        value
    )

    value = re.sub(r"\bsquare\s*mm\b", "mm2", value)
    value = re.sub(r"\bsq\s*mm\b", "mm2", value)
    value = re.sub(r"\bsqmm\b", "mm2", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def grade(value):
    value = norm(value)

    value = re.sub(r"\bschedule\s*(\d+)\b", r"sch\1", value)
    value = re.sub(r"\bsch\s*(\d+)\b", r"sch\1", value)

    value = re.sub(r"\bclass\s*(\d+)\b", r"cl\1", value)
    value = re.sub(r"\bcl\s*(\d+)\b", r"cl\1", value)

    value = re.sub(r"\bep\s*-\s*(\d+)\b", r"ep\1", value)
    value = re.sub(r"\bep\s*(\d+)\b", r"ep\1", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def unit(value):
    value = norm(value)

    mapping = {
        "meter": "m",
        "metre": "m",
        "mtr": "m",
        "mtrs": "m",
        "litre": "l",
        "liter": "l",
        "ltr": "l",
        "lt": "l",
        "number": "nos",
        "no": "nos",
        "nos": "nos",
        "ea": "nos",
        "each": "nos"
    }

    return mapping.get(value, value)


df["norm_category"] = df["category"].apply(norm)
df["norm_material"] = df["material"].apply(material)
df["norm_size"] = df["size"].apply(size)
df["norm_grade"] = df["grade"].apply(grade)
df["norm_unit"] = df["unit"].apply(unit)


def description(row):
    return " ".join([
        row["norm_category"],
        row["norm_material"],
        row["norm_size"],
        row["norm_grade"]
    ])


df["normalized_description"] = df.apply(description, axis=1)


print("\nLoading semantic model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")
embeddings = model.encode(
    df["normalized_description"].tolist(),
    show_progress_bar=True
)

print(f"Embedding shape: {embeddings.shape}")


def lexical(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()


def attribute_score(a, b):

    scores = []

    for column in [
        "norm_material",
        "norm_size",
        "norm_grade"
    ]:

        x = a[column]
        y = b[column]

        if x and y:
            scores.append(1.0 if x == y else 0.0)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def specification_conflicts(a, b):

    conflicts = []

    # Category
    if a["norm_category"] != b["norm_category"]:
        conflicts.append("category")

    # Material
    if (
        a["norm_material"]
        and b["norm_material"]
        and a["norm_material"] != b["norm_material"]
    ):
        conflicts.append("material")

    # Size
    if (
        a["norm_size"]
        and b["norm_size"]
        and a["norm_size"] != b["norm_size"]
    ):
        conflicts.append("size")

    # Grade
    if (
        a["norm_grade"]
        and b["norm_grade"]
        and a["norm_grade"] != b["norm_grade"]
    ):
        conflicts.append("grade/specification")

    return conflicts


def confidence(score, conflicts, attribute):

    if "category" in conflicts:
        return "LOW CONFIDENCE"

    # Strong specifications must agree
    if len(conflicts) >= 2:
        return "LOW CONFIDENCE"

    # If material + size/specification information exists,
    # require reasonable attribute agreement.
    if len(conflicts) == 1 and attribute < 0.50:
        return "REVIEW REQUIRED"

    if score >= 0.82:
        return "HIGH CONFIDENCE"

    if score >= 0.65:
        return "REVIEW REQUIRED"

    return "LOW CONFIDENCE"


results = []

print("\nGenerating cross-CPSE matches...")

for i in range(len(df)):

    for j in range(i + 1, len(df)):

        a = df.iloc[i]
        b = df.iloc[j]

        # Different CPSEs only
        if a["cpse"] == b["cpse"]:
            continue

        # Same category only
        if a["norm_category"] != b["norm_category"]:
            continue

        semantic_score = cosine_similarity(
            [embeddings[i]],
            [embeddings[j]]
        )[0][0]

        attr_score = attribute_score(a, b)

        lexical_score = lexical(
            a["normalized_description"],
            b["normalized_description"]
        )

        conflicts = specification_conflicts(a, b)

        # Hybrid score
        hybrid = (
            0.50 * semantic_score
            + 0.35 * attr_score
            + 0.15 * lexical_score
        )

        # Only genuine specification conflicts receive penalties.
        if len(conflicts) == 1:
            hybrid -= 0.05

        elif len(conflicts) >= 2:
            hybrid -= 0.15

        hybrid = max(0.0, min(1.0, hybrid))

        conf = confidence(
            hybrid,
            conflicts,
            attr_score
        )

        if conflicts:
            explanation = (
                f"Semantic={semantic_score:.3f}, "
                f"Attribute={attr_score:.3f}, "
                f"Lexical={lexical_score:.3f}. "
                f"Potential difference: {', '.join(conflicts)}."
            )
        else:
            explanation = (
                f"Strong agreement across semantic, "
                f"material, size and specification signals. "
                f"Semantic={semantic_score:.3f}, "
                f"Attribute={attr_score:.3f}, "
                f"Lexical={lexical_score:.3f}."
            )

        results.append({

            "cpse_1": a["cpse"],
            "material_code_1": a["material_code"],
            "material_description_1": a["material_description"],

            "cpse_2": b["cpse"],
            "material_code_2": b["material_code"],
            "material_description_2": b["material_description"],

            "category": a["category"],

            "semantic_score": round(semantic_score, 4),
            "attribute_score": round(attr_score, 4),
            "lexical_score": round(lexical_score, 4),
            "hybrid_score": round(hybrid, 4),

            "confidence": conf,

            "conflicts": ", ".join(conflicts)
            if conflicts else "None",

            "explanation": explanation,

            # Ground truth ONLY for evaluation
            "same_benchmark_group":
                a["benchmark_group_id"] ==
                b["benchmark_group_id"]
        })


matches = pd.DataFrame(results)

matches = matches.sort_values(
    "hybrid_score",
    ascending=False
)

matches.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("MATCHING V4 COMPLETE")
print("=" * 70)

print(f"\nTotal comparisons: {len(matches)}")

print("\nConfidence distribution:")
print(matches["confidence"].value_counts())

print("\nTop 10 matches:")

print(
    matches[
        [
            "cpse_1",
            "material_description_1",
            "cpse_2",
            "material_description_2",
            "hybrid_score",
            "confidence"
        ]
    ].head(10).to_string(index=False)
)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nDone!")