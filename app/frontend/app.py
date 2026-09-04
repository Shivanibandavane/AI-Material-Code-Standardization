import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CPSE Material Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F7F9FC;
}

/* Main container */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E6EAF0;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Header */
.hero {
    background: linear-gradient(
        135deg,
        #FFFFFF 0%,
        #F3F7FF 100%
    );
    border: 1px solid #E3EAF5;
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 5px 20px rgba(31, 52, 85, 0.05);
}

.hero-title {
    font-size: 30px;
    font-weight: 700;
    color: #172033;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 15px;
    color: #667085;
    line-height: 1.6;
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    background: #EAF2FF;
    color: #315EA8;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 12px;
}

/* KPI cards */
.kpi {
    background: #FFFFFF;
    border: 1px solid #E6EAF0;
    border-radius: 16px;
    padding: 20px;
    min-height: 130px;
    box-shadow: 0 4px 16px rgba(31, 52, 85, 0.04);
}

.kpi-label {
    color: #667085;
    font-size: 13px;
    font-weight: 500;
}

.kpi-value {
    color: #172033;
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.kpi-description {
    color: #98A2B3;
    font-size: 12px;
    margin-top: 5px;
}

/* Section titles */
.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #172033;
    margin-top: 28px;
    margin-bottom: 5px;
}

.section-subtitle {
    font-size: 13px;
    color: #667085;
    margin-bottom: 18px;
}

/* Cards */
.card {
    background: #FFFFFF;
    border: 1px solid #E6EAF0;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 4px 16px rgba(31, 52, 85, 0.04);
    margin-bottom: 16px;
}

.card-title {
    font-size: 16px;
    font-weight: 600;
    color: #172033;
    margin-bottom: 8px;
}

.card-text {
    font-size: 13px;
    color: #667085;
    line-height: 1.6;
}

/* Status */
.status-high {
    background: #ECFDF3;
    color: #087443;
    border: 1px solid #ABEFC6;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

.status-review {
    background: #FFFAEB;
    color: #B54708;
    border: 1px solid #FEDF89;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

.status-low {
    background: #FEF3F2;
    color: #B42318;
    border: 1px solid #FECDCA;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

/* AI explanation */
.ai-box {
    background: #F5F8FF;
    border: 1px solid #DCE7FF;
    border-radius: 14px;
    padding: 18px;
    margin-top: 12px;
}

.ai-title {
    color: #315EA8;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 6px;
}

.ai-text {
    color: #526071;
    font-size: 13px;
    line-height: 1.6;
}

/* Review card */
.review-card {
    background: #FFFFFF;
    border: 1px solid #E3E8EF;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 6px 22px rgba(31, 52, 85, 0.06);
}

/* Small label */
.small-label {
    font-size: 11px;
    color: #98A2B3;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.small-value {
    font-size: 14px;
    color: #172033;
    font-weight: 600;
}

/* Footer */
.footer {
    text-align: center;
    color: #98A2B3;
    font-size: 12px;
    padding: 30px 0 10px 0;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

MATERIAL_FILE = "data/processed/cpse_materials_attributes.csv"
MATCH_FILE = "data/processed/material_matches.csv"
MASTER_FILE = "data/processed/harmonized_material_master.csv"
REVIEW_FILE = "data/processed/human_review_decisions.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    materials = pd.read_csv(MATERIAL_FILE)
    matches = pd.read_csv(MATCH_FILE)
    master = pd.read_csv(MASTER_FILE)
    review = pd.read_csv(REVIEW_FILE)

    return materials, matches, master, review


try:

    materials_df, matches_df, master_df, review_df = load_data()

except Exception as e:

    st.error("Unable to load project data.")

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div class="badge">
AI-POWERED PROCUREMENT INTELLIGENCE
</div>

<div class="hero-title">
CPSE Material Intelligence Platform
</div>

<div class="hero-subtitle">
AI-driven standardization and harmonization of material codes
across Central Public Sector Enterprises.
<br>
Transforming fragmented material masters into a unified,
explainable and reviewable procurement intelligence layer.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="font-size:22px;font-weight:700;color:#172033;">
        Material Intelligence
        </div>
        <div style="font-size:12px;color:#98A2B3;margin-top:4px;">
        CPSE Standardization Platform
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "",
        [
            "📊 Overview",
            "🔍 AI Matching",
            "🧩 Harmonization",
            "👤 Human Review",
            "📥 Export"
        ]
    )

    st.divider()

    st.markdown("### System Status")

    st.success("AI Matching Engine Online")
    st.success("Harmonization Engine Online")
    st.info("Human Review Enabled")

    st.divider()

    st.caption(
        "Prototype • SIH 2026\n\n"
        "AI recommendations require human validation."
    )


# ============================================================
# COMMON STATISTICS
# ============================================================

total_materials = len(materials_df)

total_matches = len(matches_df)

total_groups = len(master_df)

review_groups = len(
    master_df[
        master_df["source_material_count"] > 1
    ]
)

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


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "📊 Overview":

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A high-level view of the material standardization pipeline.'
        '</div>',
        unsafe_allow_html=True
    )

    # KPI cards

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Source Materials</div>
                <div class="kpi-value">{total_materials}</div>
                <div class="kpi-description">
                    Materials across CPSE datasets
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">AI Comparisons</div>
                <div class="kpi-value">{total_matches}</div>
                <div class="kpi-description">
                    Cross-CPSE comparisons evaluated
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Canonical Groups</div>
                <div class="kpi-value">{total_groups}</div>
                <div class="kpi-description">
                    Prototype standardized groups
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Human Review</div>
                <div class="kpi-value">{review_groups}</div>
                <div class="kpi-description">
                    Groups requiring validation
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">AI Matching Intelligence</div>',
        unsafe_allow_html=True
    )

    chart1, chart2 = st.columns(2)

    with chart1:

        confidence_counts = (
            matches_df["confidence"]
            .value_counts()
            .reset_index()
        )

        confidence_counts.columns = [
            "Confidence",
            "Count"
        ]

        fig = px.bar(
            confidence_counts,
            x="Confidence",
            y="Count",
            title="AI Match Confidence Distribution",
            text="Count"
        )

        fig.update_layout(
            template="simple_white",
            font=dict(
                family="Inter"
            ),
            height=350,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with chart2:

        category_counts = (
            materials_df["category"]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = [
            "Category",
            "Count"
        ]

        fig2 = px.bar(
            category_counts,
            x="Count",
            y="Category",
            orientation="h",
            title="Materials by Category"
        )

        fig2.update_layout(
            template="simple_white",
            font=dict(
                family="Inter"
            ),
            height=350,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


    # --------------------------------------------------------
    # AI architecture
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">How the AI Pipeline Works</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <div class="card-title">
    🔄 Hybrid Material Matching Pipeline
    </div>

    <div class="card-text">

    <b>1. Data Cleaning</b> → Standardizes descriptions and terminology.

    <br><br>

    <b>2. Attribute Extraction</b> → Identifies material, size, grade and other specifications.

    <br><br>

    <b>3. Semantic Matching</b> → Uses AI embeddings to understand similar descriptions.

    <br><br>

    <b>4. Hybrid Scoring</b> → Combines semantic, attribute and lexical similarity.

    <br><br>

    <b>5. Conflict Detection</b> → Detects specification differences before harmonization.

    <br><br>

    <b>6. Harmonization</b> → Creates prototype canonical material groups.

    <br><br>

    <b>7. Human Validation</b> → Procurement experts approve or reject AI recommendations.

    </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# AI MATCHING PAGE
# ============================================================

elif page == "🔍 AI Matching":

    st.markdown(
        '<div class="section-title">AI Matching Engine</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore how the AI evaluates material similarity across CPSEs.'
        '</div>',
        unsafe_allow_html=True
    )

    # Summary

    a, b, c = st.columns(3)

    with a:

        st.metric(
            "High Confidence",
            high_matches
        )

    with b:

        st.metric(
            "Review Required",
            review_matches
        )

    with c:

        st.metric(
            "Low Confidence",
            low_matches
        )


    st.divider()

    # Filters

    f1, f2, f3 = st.columns(3)

    with f1:

        cpse_filter = st.multiselect(
            "CPSE",
            sorted(
                set(
                    matches_df["cpse_1"]
                )
                |
                set(
                    matches_df["cpse_2"]
                )
            )
        )

    with f2:

        confidence_filter = st.multiselect(
            "Confidence",
            sorted(
                matches_df["confidence"]
                .dropna()
                .unique()
            )
        )

    with f3:

        category_filter = st.multiselect(
            "Category",
            sorted(
                matches_df["category"]
                .dropna()
                .unique()
            )
        )


    filtered = matches_df.copy()


    if cpse_filter:

        filtered = filtered[
            filtered["cpse_1"].isin(cpse_filter)
            |
            filtered["cpse_2"].isin(cpse_filter)
        ]


    if confidence_filter:

        filtered = filtered[
            filtered["confidence"].isin(
                confidence_filter
            )
        ]


    if category_filter:

        filtered = filtered[
            filtered["category"].isin(
                category_filter
            )
        ]


    st.markdown(
        f"**Showing {len(filtered)} AI comparisons**"
    )


    display_columns = [
        "cpse_1",
        "material_code_1",
        "material_description_1",
        "cpse_2",
        "material_code_2",
        "material_description_2",
        "category",
        "semantic_score",
        "attribute_score",
        "lexical_score",
        "hybrid_score",
        "confidence",
        "conflicts"
    ]

    st.dataframe(
        filtered[
            display_columns
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# HARMONIZATION PAGE
# ============================================================

elif page == "🧩 Harmonization":

    st.markdown(
        '<div class="section-title">Material Harmonization</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'AI-generated prototype canonical material groups.'
        '</div>',
        unsafe_allow_html=True
    )


    # Search

    search = st.text_input(
        "🔎 Search material, category, CPSE or canonical code"
    )


    harmonized = master_df[
        master_df["source_material_count"] > 1
    ].copy()


    if search:

        search_lower = search.lower()

        mask = (
            harmonized.astype(str)
            .apply(
                lambda col:
                col.str.lower().str.contains(
                    search_lower,
                    na=False
                )
            )
            .any(axis=1)
        )

        harmonized = harmonized[mask]


    st.markdown(
        f"**{len(harmonized)} harmonized groups found**"
    )


    for _, row in harmonized.iterrows():

        with st.container():

            st.markdown(
                f"""
                <div class="card">

                <div class="small-label">
                Prototype Canonical Material Code
                </div>

                <div style="
                    font-size:20px;
                    font-weight:700;
                    color:#315EA8;
                    margin:5px 0 15px 0;
                ">
                {row['prototype_canonical_material_code']}
                </div>

                <div class="card-text">

                <b>Category:</b>
                {row['canonical_category']}

                &nbsp;&nbsp; | &nbsp;&nbsp;

                <b>Material:</b>
                {row['canonical_material']}

                &nbsp;&nbsp; | &nbsp;&nbsp;

                <b>Size:</b>
                {row['canonical_size']}

                &nbsp;&nbsp; | &nbsp;&nbsp;

                <b>Grade:</b>
                {row['canonical_grade']}

                <br><br>

                <b>Source Materials:</b>
                {row['source_material_count']}

                &nbsp;&nbsp; | &nbsp;&nbsp;

                <b>CPSEs:</b>
                {row['source_cpses']}

                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander(
                "View source materials and AI recommendation"
            ):

                st.write(
                    "**Source Material Codes**"
                )

                st.info(
                    row["source_material_codes"]
                )

                st.write(
                    "**Original Descriptions**"
                )

                st.info(
                    row["source_descriptions"]
                )

                st.markdown(
                    f"""
                    <div class="ai-box">

                    <div class="ai-title">
                    🧠 AI Recommendation
                    </div>

                    <div class="ai-text">
                    {row['ai_recommendation']}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# HUMAN REVIEW PAGE
# ============================================================

elif page == "👤 Human Review":

    st.markdown(
        '<div class="section-title">Human-in-the-Loop Review</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Review AI recommendations before they become part of the standardized master.'
        '</div>',
        unsafe_allow_html=True
    )


    pending = review_df[
        review_df["human_decision"] == "PENDING"
    ].copy()


    approved = review_df[
        review_df["human_decision"] == "APPROVED"
    ]

    rejected = review_df[
        review_df["human_decision"] == "REJECTED"
    ]


    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Pending",
            len(pending)
        )

    with r2:

        st.metric(
            "Approved",
            len(approved)
        )

    with r3:

        st.metric(
            "Rejected",
            len(rejected)
        )


    st.divider()


    if len(pending) == 0:

        st.success(
            "🎉 No pending material groups require review."
        )

    else:

        selected_code = st.selectbox(
            "Select a material group to review",
            pending[
                "prototype_canonical_material_code"
            ].tolist()
        )


        row = pending[
            pending[
                "prototype_canonical_material_code"
            ] == selected_code
        ].iloc[0]


        st.markdown(
            '<div class="review-card">',
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="small-label">
            AI GENERATED CANONICAL GROUP
            </div>

            <div style="
                font-size:25px;
                font-weight:700;
                color:#315EA8;
                margin:7px 0 20px 0;
            ">
            {row['prototype_canonical_material_code']}
            </div>
            """,
            unsafe_allow_html=True
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.markdown(
                f"""
                <div class="small-label">Category</div>
                <div class="small-value">
                {row['canonical_category']}
                </div>
                """,
                unsafe_allow_html=True
            )


        with col2:

            st.markdown(
                f"""
                <div class="small-label">Material</div>
                <div class="small-value">
                {row['canonical_material']}
                </div>
                """,
                unsafe_allow_html=True
            )


        with col3:

            st.markdown(
                f"""
                <div class="small-label">Size</div>
                <div class="small-value">
                {row['canonical_size']}
                </div>
                """,
                unsafe_allow_html=True
            )


        with col4:

            st.markdown(
                f"""
                <div class="small-label">Grade</div>
                <div class="small-value">
                {row['canonical_grade']}
                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown("<br>", unsafe_allow_html=True)


        st.write("### Source Material Evidence")


        source_codes = str(
            row["source_material_codes"]
        ).split(" | ")


        descriptions = str(
            row["source_descriptions"]
        ).split(" | ")


        source_table = []


        for i in range(
            min(
                len(source_codes),
                len(descriptions)
            )
        ):

            code_parts = source_codes[i].split(":")

            if len(code_parts) >= 2:

                cpse = code_parts[0]

                code = ":".join(
                    code_parts[1:]
                )

            else:

                cpse = ""

                code = source_codes[i]


            source_table.append({

                "CPSE": cpse,

                "Material Code": code,

                "Original Description":
                    descriptions[i]
            })


        st.dataframe(
            pd.DataFrame(source_table),
            use_container_width=True,
            hide_index=True
        )


        st.markdown(
            f"""
            <div class="ai-box">

            <div class="ai-title">
            🧠 Why AI recommends harmonization
            </div>

            <div class="ai-text">
            {row['ai_recommendation']}
            <br><br>
            The reviewer should verify that the source materials
            represent the same procurement item and that their
            specifications are compatible.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown("<br>", unsafe_allow_html=True)


        comment = st.text_area(
            "Reviewer comment",
            placeholder=(
                "Example: Specifications are equivalent "
                "across the listed CPSEs."
            )
        )


        reviewer = st.text_input(
            "Reviewer name / ID",
            placeholder="Enter reviewer name"
        )


        b1, b2, b3 = st.columns(3)


        def save_decision(
            decision
        ):

            global review_df

            review_df.loc[
                review_df[
                    "prototype_canonical_material_code"
                ] == selected_code,
                "human_decision"
            ] = decision

            review_df.loc[
                review_df[
                    "prototype_canonical_material_code"
                ] == selected_code,
                "reviewer"
            ] = reviewer

            review_df.loc[
                review_df[
                    "prototype_canonical_material_code"
                ] == selected_code,
                "review_comment"
            ] = comment

            review_df.loc[
                review_df[
                    "prototype_canonical_material_code"
                ] == selected_code,
                "review_timestamp"
            ] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            review_df.to_csv(
                REVIEW_FILE,
                index=False
            )

            st.cache_data.clear()

            st.success(
                f"Decision recorded: {decision}"
            )

            st.rerun()


        with b1:

            if st.button(
                "✅ APPROVE",
                use_container_width=True
            ):

                save_decision(
                    "APPROVED"
                )


        with b2:

            if st.button(
                "❌ REJECT",
                use_container_width=True
            ):

                save_decision(
                    "REJECTED"
                )


        with b3:

            if st.button(
                "⏭️ SKIP",
                use_container_width=True
            ):

                save_decision(
                    "SKIPPED"
                )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================
# EXPORT PAGE
# ============================================================

elif page == "📥 Export":

    st.markdown(
        '<div class="section-title">Export Standardized Master</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Download AI-generated harmonization results and human review decisions.'
        '</div>',
        unsafe_allow_html=True
    )


    # Approval statistics

    approved_count = len(
        review_df[
            review_df["human_decision"] == "APPROVED"
        ]
    )

    rejected_count = len(
        review_df[
            review_df["human_decision"] == "REJECTED"
        ]
    )

    pending_count = len(
        review_df[
            review_df["human_decision"] == "PENDING"
        ]
    )


    e1, e2, e3 = st.columns(3)


    with e1:

        st.metric(
            "Approved Groups",
            approved_count
        )


    with e2:

        st.metric(
            "Rejected Groups",
            rejected_count
        )


    with e3:

        st.metric(
            "Pending Groups",
            pending_count
        )


    st.divider()


    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        📦 Prototype Standardized Material Master
        </div>

        <div class="card-text">
        This export contains the AI-generated prototype canonical
        material groups together with their source CPSE information.
        It is intended for demonstration and validation purposes,
        not as an official CPSE coding standard.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    master_csv = master_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="📥 Download Canonical Material Master",
        data=master_csv,
        file_name="prototype_canonical_material_master.csv",
        mime="text/csv",
        use_container_width=True
    )


    review_csv = review_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="📋 Download Human Review Audit Trail",
        data=review_csv,
        file_name="human_review_audit_trail.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    AI Material Code Standardization • SIH 2026 Prototype

    <br>

    <span style="font-size:11px;">
    AI-assisted recommendations • Explainable matching •
    Human-in-the-loop validation
    </span>

    </div>
    """,
    unsafe_allow_html=True
)