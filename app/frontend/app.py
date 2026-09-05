import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CPSE Material Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ATTRIBUTES_FILE = ROOT / "data" / "processed" / "cpse_materials_attributes.csv"
MATCHES_FILE = ROOT / "data" / "processed" / "material_matches.csv"
MASTER_FILE = ROOT / "data" / "processed" / "harmonized_material_master.csv"
REVIEW_FILE = ROOT / "data" / "processed" / "human_review_decisions.csv"

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #172033;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #312e81;
}

.subtitle {
    color: #64748b;
    font-size: 15px;
    margin-bottom: 25px;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #312e81, #4f46e5, #7c3aed);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 10px;
}

.hero p {
    color: #e0e7ff;
    font-size: 15px;
    line-height: 1.6;
}

.kpi-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(0,0,0,0.04);
}

.kpi-number {
    font-size: 30px;
    font-weight: 800;
    color: #4f46e5;
}

.kpi-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
}

.section-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}

.status {
    padding: 10px;
    border-radius: 10px;
    background-color: #ecfdf5;
    color: #047857;
    margin-bottom: 8px;
    font-size: 13px;
    font-weight: 600;
}

.code {
    background-color: #eef2ff;
    color: #3730a3;
    padding: 5px 9px;
    border-radius: 7px;
    font-family: monospace;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    attributes = pd.read_csv(ATTRIBUTES_FILE)
    matches = pd.read_csv(MATCHES_FILE)
    master = pd.read_csv(MASTER_FILE)
    review = pd.read_csv(REVIEW_FILE)

    return attributes, matches, master, review


try:
    attributes_df, matches_df, master_df, review_df = load_data()

except Exception as e:

    st.error("Unable to load project data.")
    st.exception(e)
    st.stop()

# ============================================================
# STATISTICS
# ============================================================

total_materials = len(attributes_df)
total_matches = len(matches_df)
total_groups = len(master_df)

high_matches = len(
    matches_df[
        matches_df["confidence"] == "HIGH CONFIDENCE"
    ]
)

review_matches = len(
    matches_df[
        matches_df["confidence"] == "REVIEW REQUIRED"
    ]
)

low_matches = len(
    matches_df[
        matches_df["confidence"] == "LOW CONFIDENCE"
    ]
)

if "source_material_count" in master_df.columns:

    review_groups = len(
        master_df[
            master_df["source_material_count"] > 1
        ]
    )

else:

    review_groups = 0


pending_reviews = len(
    review_df[
        review_df["human_decision"] == "PENDING"
    ]
)

approved_reviews = len(
    review_df[
        review_df["human_decision"] == "APPROVED"
    ]
)

rejected_reviews = len(
    review_df[
        review_df["human_decision"] == "REJECTED"
    ]
)

cpse_count = attributes_df["cpse"].nunique()
category_count = attributes_df["category"].nunique()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏭 Material Intelligence")

    st.caption(
        "CPSE Standardization Platform"
    )

    st.divider()

    st.subheader("WORKSPACE")

    page = st.radio(
        "Select Module",
        [
            "Overview",
            "AI Matching",
            "Harmonization",
            "Human Review",
            "Export"
        ]
    )

    st.divider()

    st.subheader("SYSTEM STATUS")

    st.markdown(
        '<div class="status">✅ Matching Engine Online</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="status">✅ Harmonization Engine Online</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="status">👤 Human Review Enabled</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.caption(
        f"📦 {total_materials} source materials"
    )

    st.caption(
        f"🏢 {cpse_count} CPSE organizations"
    )

    st.caption(
        f"🏷️ {category_count} categories"
    )

    st.caption(
        "SIH 2026 Prototype"
    )

# ============================================================
# TOP HEADER
# ============================================================

top1, top2 = st.columns([4, 1])

with top1:

    st.caption(
        "CPSE MATERIAL INTELLIGENCE / STANDARDIZATION WORKSPACE"
    )

with top2:

    st.caption(
        datetime.now().strftime("%d %b %Y")
    )

st.divider()

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <h1>CPSE Material Intelligence Platform</h1>
            <p>
            AI-assisted standardization and harmonization of
            material codes across Central Public Sector Enterprises.
            </p>
            <p>
            Semantic Matching • Attribute Analysis • Conflict Detection
            • Harmonization • Human-in-the-Loop
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Platform Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Source Materials",
            total_materials
        )

    with c2:

        st.metric(
            "AI Comparisons",
            total_matches
        )

    with c3:

        st.metric(
            "Canonical Groups",
            total_groups
        )

    with c4:

        st.metric(
            "Review Queue",
            pending_reviews
        )

    st.write("")

    left, right = st.columns(2)

    with left:

        st.subheader(
            "AI Confidence Distribution"
        )

        confidence_df = pd.DataFrame(
            {
                "Confidence": [
                    "HIGH CONFIDENCE",
                    "REVIEW REQUIRED",
                    "LOW CONFIDENCE"
                ],
                "Matches": [
                    high_matches,
                    review_matches,
                    low_matches
                ]
            }
        )

        fig = px.bar(
            confidence_df,
            x="Confidence",
            y="Matches",
            text="Matches"
        )

        fig.update_layout(
            height=350,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    with right:

        st.subheader(
            "Material Categories"
        )

        category_df = (
            attributes_df[
                "category"
            ]
            .value_counts()
            .reset_index()
        )

        category_df.columns = [
            "Category",
            "Materials"
        ]

        fig2 = px.bar(
            category_df,
            x="Category",
            y="Materials",
            text="Materials"
        )

        fig2.update_layout(
            height=350,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10
            )
        )

        st.plotly_chart(
            fig2,
            width="stretch"
        )

    st.subheader(
        "End-to-End Processing Pipeline"
    )

    p1, p2, p3, p4, p5, p6 = st.columns(6)

    with p1:
        st.info("01\n\n📁 Upload")

    with p2:
        st.info("02\n\n🧹 Clean")

    with p3:
        st.info("03\n\n🤖 Match")

    with p4:
        st.info("04\n\n🧩 Harmonize")

    with p5:
        st.warning("05\n\n👤 Review")

    with p6:
        st.success("06\n\n📥 Export")

# ============================================================
# AI MATCHING
# ============================================================

elif page == "AI Matching":

    st.title("🔍 AI Matching Explorer")

    st.write(
        "Explore AI-generated material similarity "
        "and the evidence behind each recommendation."
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Total Comparisons",
            total_matches
        )

    with m2:
        st.metric(
            "High Confidence",
            high_matches
        )

    with m3:
        st.metric(
            "Review Required",
            review_matches
        )

    st.divider()

    f1, f2, f3 = st.columns(3)

    with f1:

        cpse_options = sorted(
            set(
                matches_df["cpse_1"]
                .dropna()
                .astype(str)
            )
            |
            set(
                matches_df["cpse_2"]
                .dropna()
                .astype(str)
            )
        )

        selected_cpse = st.multiselect(
            "CPSE",
            cpse_options
        )

    with f2:

        selected_confidence = st.multiselect(
            "Confidence",
            [
                "HIGH CONFIDENCE",
                "REVIEW REQUIRED",
                "LOW CONFIDENCE"
            ]
        )

    with f3:

        category_options = sorted(
            matches_df["category"]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_category = st.multiselect(
            "Category",
            category_options
        )

    filtered = matches_df.copy()

    if selected_cpse:

        filtered = filtered[
            filtered["cpse_1"]
            .astype(str)
            .isin(selected_cpse)
            |
            filtered["cpse_2"]
            .astype(str)
            .isin(selected_cpse)
        ]

    if selected_confidence:

        filtered = filtered[
            filtered["confidence"]
            .isin(selected_confidence)
        ]

    if selected_category:

        filtered = filtered[
            filtered["category"]
            .astype(str)
            .isin(selected_category)
        ]

    st.write(
        f"Showing {len(filtered)} comparisons"
    )

    columns = [
        "cpse_1",
        "material_code_1",
        "cpse_2",
        "material_code_2",
        "category",
        "semantic_score",
        "attribute_score",
        "lexical_score",
        "hybrid_score",
        "confidence"
    ]

    columns = [
        c for c in columns
        if c in filtered.columns
    ]

    st.dataframe(
        filtered[columns],
        width="stretch",
        hide_index=True
    )

    if len(filtered) > 0:

        st.divider()

        st.subheader(
            "Match Evidence"
        )

        selected_index = st.selectbox(
            "Select comparison",
            filtered.index,
            format_func=lambda x:
            f"{filtered.loc[x, 'material_code_1']} ↔ "
            f"{filtered.loc[x, 'material_code_2']}"
        )

        row = filtered.loc[selected_index]

        a, b = st.columns(2)

        with a:

            st.markdown(
                "### Source Material 1"
            )

            st.info(
                f"""
CPSE: {row["cpse_1"]}

Material Code: {row["material_code_1"]}

Description:
{row["material_description_1"]}
"""
            )

        with b:

            st.markdown(
                "### Source Material 2"
            )

            st.info(
                f"""
CPSE: {row["cpse_2"]}

Material Code: {row["material_code_2"]}

Description:
{row["material_description_2"]}
"""
            )

        st.subheader(
            "AI Scoring"
        )

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric(
                "Semantic",
                f"{row['semantic_score']:.3f}"
            )

        with s2:
            st.metric(
                "Attributes",
                f"{row['attribute_score']:.3f}"
            )

        with s3:
            st.metric(
                "Lexical",
                f"{row['lexical_score']:.3f}"
            )

        with s4:
            st.metric(
                "Hybrid",
                f"{row['hybrid_score']:.3f}"
            )

        st.subheader(
            "AI Decision"
        )

        confidence = row["confidence"]

        if confidence == "HIGH CONFIDENCE":

            st.success(
                "🟢 HIGH CONFIDENCE"
            )

        elif confidence == "REVIEW REQUIRED":

            st.warning(
                "🟡 REVIEW REQUIRED"
            )

        else:

            st.error(
                "🔴 LOW CONFIDENCE"
            )

        st.info(
            row.get(
                "explanation",
                "No explanation available."
            )
        )

# ============================================================
# HARMONIZATION
# ============================================================

elif page == "Harmonization":

    st.title(
        "🧩 Material Harmonization"
    )

    st.write(
        "Prototype canonical material groups generated "
        "from high-confidence AI matches."
    )

    h1, h2, h3 = st.columns(3)

    with h1:
        st.metric(
            "Canonical Groups",
            total_groups
        )

    with h2:
        st.metric(
            "Multi-Source Groups",
            review_groups
        )

    with h3:
        st.metric(
            "Pending Review",
            pending_reviews
        )

    st.divider()

    search = st.text_input(
        "🔎 Search canonical groups",
        placeholder="Search material, category, code or CPSE..."
    )

    groups = master_df.copy()

    if "source_material_count" in groups.columns:

        groups = groups[
            groups["source_material_count"] > 1
        ]

    if search:

        mask = (
            groups
            .astype(str)
            .apply(
                lambda x:
                x.str.contains(
                    search,
                    case=False,
                    na=False
                )
            )
            .any(axis=1)
        )

        groups = groups[mask]

    st.write(
        f"{len(groups)} harmonized groups found."
    )

    for _, row in groups.iterrows():

        canonical = row.get(
            "canonical_material_code",
            "N/A"
        )

        category = row.get(
            "category",
            "N/A"
        )

        material = row.get(
            "material",
            "N/A"
        )

        size = row.get(
            "size",
            "N/A"
        )

        grade = row.get(
            "grade",
            "N/A"
        )

        count = row.get(
            "source_material_count",
            0
        )

        with st.container(border=True):

            st.markdown(
                f"### 🔹 {canonical}"
            )

            st.write(
                f"**Category:** {category}"
            )

            st.write(
                f"**Material:** {material}"
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.write(
                    f"**Size:** {size}"
                )

            with c2:
                st.write(
                    f"**Grade:** {grade}"
                )

            with c3:
                st.write(
                    f"**Source Records:** {count}"
                )

            st.write(
                f"**CPSEs:** "
                f"{row.get('source_cpses', 'N/A')}"
            )

            with st.expander(
                "View source evidence"
            ):

                st.write(
                    "**Material Codes:**"
                )

                st.write(
                    row.get(
                        "source_material_codes",
                        "N/A"
                    )
                )

                st.write(
                    "**Original Descriptions:**"
                )

                st.write(
                    row.get(
                        "source_descriptions",
                        "N/A"
                    )
                )

                st.write(
                    "**AI Recommendation:**"
                )

                st.info(
                    row.get(
                        "ai_recommendation",
                        "N/A"
                    )
                )

# ============================================================
# HUMAN REVIEW
# ============================================================

elif page == "Human Review":

    st.title(
        "👤 Human-in-the-Loop Review"
    )

    st.write(
        "Review AI recommendations before final standardization."
    )

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Pending",
            pending_reviews
        )

    with r2:
        st.metric(
            "Approved",
            approved_reviews
        )

    with r3:
        st.metric(
            "Rejected",
            rejected_reviews
        )

    st.divider()

    pending_df = review_df[
        review_df["human_decision"]
        == "PENDING"
    ].copy()

    if pending_df.empty:

        st.success(
            "🎉 No pending reviews!"
        )

    else:

        selected_code = st.selectbox(
            "Select canonical material group",
            pending_df[
                "canonical_material_code"
            ].astype(str).tolist()
        )

        row = pending_df[
            pending_df[
                "canonical_material_code"
            ].astype(str)
            == selected_code
        ].iloc[0]

        st.subheader(
            f"Review: {selected_code}"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.write(
                f"**Category:** "
                f"{row.get('canonical_category', 'N/A')}"
            )

            st.write(
                f"**Material:** "
                f"{row.get('canonical_material', 'N/A')}"
            )

            st.write(
                f"**Size:** "
                f"{row.get('canonical_size', 'N/A')}"
            )

            st.write(
                f"**Grade:** "
                f"{row.get('canonical_grade', 'N/A')}"
            )

        with c2:

            st.write(
                f"**Source Count:** "
                f"{row.get('source_material_count', 'N/A')}"
            )

            st.write(
                f"**CPSEs:** "
                f"{row.get('source_cpses', 'N/A')}"
            )

        st.divider()

        st.subheader(
            "Source Evidence"
        )

        st.write(
            "**Material Codes**"
        )

        st.info(
            row.get(
                "source_material_codes",
                "N/A"
            )
        )

        st.write(
            "**Original Descriptions**"
        )

        st.info(
            row.get(
                "source_descriptions",
                "N/A"
            )
        )

        st.subheader(
            "🤖 AI Recommendation"
        )

        st.info(
            row.get(
                "ai_recommendation",
                "No recommendation available."
            )
        )

        reviewer = st.text_input(
            "Reviewer Name / ID"
        )

        comment = st.text_area(
            "Review Comment",
            placeholder="Explain your decision..."
        )

        b1, b2, b3 = st.columns(3)

        if "review_decision_message" not in st.session_state:

            st.session_state.review_decision_message = ""

        def save_review(decision):

            mask = (
                review_df[
                    "canonical_material_code"
                ].astype(str)
                == selected_code
            )

            review_df.loc[
                mask,
                "human_decision"
            ] = decision

            if "reviewer" in review_df.columns:

                review_df.loc[
                    mask,
                    "reviewer"
                ] = reviewer

            if "review_comment" in review_df.columns:

                review_df.loc[
                    mask,
                    "review_comment"
                ] = comment

            if "review_timestamp" in review_df.columns:

                review_df.loc[
                    mask,
                    "review_timestamp"
                ] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            review_df.to_csv(
                REVIEW_FILE,
                index=False
            )

            st.cache_data.clear()

        with b1:

            if st.button(
                "✅ APPROVE",
                width="stretch"
            ):

                save_review(
                    "APPROVED"
                )

                st.success(
                    "Material group approved."
                )

                st.rerun()

        with b2:

            if st.button(
                "❌ REJECT",
                width="stretch"
            ):

                save_review(
                    "REJECTED"
                )

                st.warning(
                    "Material group rejected."
                )

                st.rerun()

        with b3:

            if st.button(
                "⏭️ SKIP",
                width="stretch"
            ):

                st.info(
                    "Review skipped. Decision remains pending."
                )

# ============================================================
# EXPORT
# ============================================================

elif page == "Export":

    st.title(
        "📥 Export Standardized Master"
    )

    st.write(
        "Download prototype harmonization results "
        "and the human-review audit trail."
    )

    e1, e2, e3 = st.columns(3)

    with e1:

        st.metric(
            "Approved Groups",
            approved_reviews
        )

    with e2:

        st.metric(
            "Rejected Groups",
            rejected_reviews
        )

    with e3:

        st.metric(
            "Pending Groups",
            pending_reviews
        )

    st.divider()

    st.subheader(
        "📦 Prototype Canonical Material Master"
    )

    st.warning(
        "These canonical material codes are prototype "
        "identifiers for demonstration and evaluation. "
        "They are NOT official CPSE material codes."
    )

    st.dataframe(
        master_df,
        width="stretch",
        hide_index=True
    )

    master_csv = (
        master_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Prototype Canonical Master",
        data=master_csv,
        file_name="prototype_canonical_material_master.csv",
        mime="text/csv",
        width="stretch"
    )

    st.divider()

    st.subheader(
        "📋 Human Review Audit Trail"
    )

    st.dataframe(
        review_df,
        width="stretch",
        hide_index=True
    )

    review_csv = (
        review_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Human Review Audit Trail",
        data=review_csv,
        file_name="human_review_audit_trail.csv",
        mime="text/csv",
        width="stretch"
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏭 CPSE Material Intelligence Platform • "
    "AI Matching • Harmonization • Human-in-the-Loop • "
    "SIH 2026 Prototype"
)