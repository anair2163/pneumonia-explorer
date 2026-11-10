import streamlit as st

from src.layout import (
    render_page_header,
    render_profile_links_card,
    render_sidebar,
    render_surface_card,
    setup_page,
)


def safe_page_link(page: str, *, label: str, icon: str) -> None:
    try:
        st.page_link(page, label=label, icon=icon)
    except Exception:
        st.markdown(f"`{icon} {label}`")


setup_page()
render_sidebar()

render_page_header(
    eyebrow="Why This Project",
    title="Chest X-Ray Classification and Augmentation Explorer",
    description=(
        "This project explores how chest X-ray classification, model evaluation, and augmentation experiments can be studied together in a practical medical imaging workflow."
    ),
    chips=[
        "Clinical relevance",
        "Computer vision storytelling",
        "Synthetic data strategy",
        "Portfolio-ready UX",
    ],
)

overview_left, overview_right = st.columns([1.1, 0.9], gap="large")

with overview_left:
    render_surface_card(
        eyebrow="Why Pneumonia Detection Matters",
        title="Chest X-rays can support faster, more consistent triage",
        body=(
            "Pneumonia remains a serious respiratory condition, and delayed recognition can affect outcomes. "
            "Chest X-rays are common, relatively accessible, and clinically meaningful, which makes them a strong "
            "setting for decision-support research and careful machine learning evaluation."
        ),
    )

with overview_right:
    render_surface_card(
        eyebrow="Why Image Prediction Is Interesting",
        title="Medical imaging combines pattern recognition with real-world stakes",
        body=(
            "Image-based prediction is compelling because it sits at the intersection of visual reasoning, "
            "data quality, interpretability, and deployment constraints. It is not just about classification "
            "accuracy. It is about whether a model can support a reliable and understandable workflow."
        ),
    )

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

render_surface_card(
    eyebrow="Why Compare Augmentation Settings",
    title="Better coverage can be as important as better architecture",
    body=(
        "Training-time augmentation is worth studying because medical datasets are often limited, imbalanced, or "
        "narrow in visual diversity. Exploring synthetic examples creates a structured way to ask whether broader "
        "training coverage can improve robustness without introducing misleading shortcuts or unrealistic artifacts."
    ),
)

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

st.subheader("What this site includes")
feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4, gap="large")

with feature_col1:
    render_surface_card(
        eyebrow="Context",
        title="Project framing",
        body="A concise narrative that explains the medical use case, the dataset, and the reasoning behind the project direction.",
    )

with feature_col2:
    render_surface_card(
        eyebrow="Analysis",
        title="Exploration views",
        body="Dedicated pages for dataset context, exploratory analytics, and early signals that inform later modeling decisions.",
    )

with feature_col3:
    render_surface_card(
        eyebrow="Interaction",
        title="Prediction concepts",
        body="A lightweight prediction lab scaffold that shows how inference and user-facing outputs can be presented clearly.",
    )

with feature_col4:
    render_surface_card(
        eyebrow="Evaluation",
        title="Model and augmentation story",
        body="Focused sections for performance comparison, image augmentation strategy, and final takeaways grounded in the saved model results.",
    )

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

st.subheader("Explore the site")
st.caption("Use these links as the main guided path through the project.")

nav_col1, nav_col2, nav_col3 = st.columns(3, gap="medium")
with nav_col1:
    safe_page_link("pages/2_Data_and_Context.py", label="Data & Context", icon="🗂️")
    safe_page_link("pages/3_Analytics_Exploration.py", label="Analytics Exploration", icon="📊")
with nav_col2:
    safe_page_link("pages/4_Prediction_Lab.py", label="Prediction Lab", icon="🧪")
    safe_page_link("pages/5_Model_Performance.py", label="Model Performance", icon="📈")
with nav_col3:
    safe_page_link("pages/6_Synthetic_Augmentation.py", label="Image Augmentation", icon="🧬")
    safe_page_link("pages/7_Key_Findings.py", label="Key Findings", icon="✅")

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

contact_left, contact_right = st.columns([1.2, 0.8], gap="large")

with contact_left:
    render_profile_links_card(eyebrow="Portfolio Links", title="Project Profiles")

with contact_right:
    render_surface_card(
        eyebrow="Audience",
        title="Built to be easy to review",
        body=(
            "The goal is to make the project easy to explore for anyone who wants to understand the modeling direction, the error tradeoffs, and the limits of the current results."
        ),
    )
