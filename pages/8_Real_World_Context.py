from __future__ import annotations

import streamlit as st

from src.layout import (
    render_page_header,
    render_profile_links_card,
    render_sidebar,
    render_surface_card,
    setup_page,
)


setup_page()
render_sidebar()

render_page_header(
    eyebrow="Real-World Context",
    title="Real-World Context",
    description=(
        "This page places the project in the broader context of medical imaging research while keeping the distinction clear between active research directions and clinical deployment."
    ),
    chips=["Medical imaging research", "Clinical support context", "Educational use only", "Responsible interpretation"],
)

top_left, top_right = st.columns([1.1, 0.9], gap="large")
with top_left:
    render_surface_card(
        eyebrow="Why This Direction Matters",
        title="Why medical imaging is a strong fit for computer vision",
        body=(
            "Computer vision in medical imaging is being studied because it has the potential to improve diagnostic accuracy, support faster image review, and make healthcare workflows more efficient. "
            "In imaging-heavy settings, even small improvements in prioritization, anomaly detection, or decision support can affect how quickly cases are reviewed and how consistently findings are identified."
        ),
    )
with top_right:
    render_surface_card(
        eyebrow="Important Disclaimer",
        title="This project is not a clinical tool",
        body=(
            "This website is an educational machine learning project. It is not a medical device, not a diagnostic system, and not a substitute for professional medical judgment. "
            "The predictions and visualizations shown here are demonstrations of model behavior on a public dataset and should not be used to make clinical decisions."
        ),
    )

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

render_surface_card(
    eyebrow="Where These Systems May Help",
    title="Common roles for vision models in imaging",
    body=(
        "In practice, computer vision methods are relevant to several medical imaging tasks, including image classification, object detection, segmentation, and workflow support. "
        "In chest imaging specifically, this makes pneumonia detection a useful case study: it is visually meaningful, clinically important, and closely tied to the broader question of how image-based models can assist with early review and prioritization."
    ),
)

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

card_1, card_2, card_3 = st.columns(3, gap="large")
with card_1:
    render_surface_card(
        eyebrow="Classification",
        title="Sorting images into clinically relevant groups",
        body=(
            "Models can be trained to distinguish between categories such as normal and abnormal imaging findings."
        ),
    )
with card_2:
    render_surface_card(
        eyebrow="Detection & Localization",
        title="Pointing attention toward suspicious regions",
        body=(
            "Some systems are designed to highlight suspicious regions or identify structures and abnormalities within an image."
        ),
    )
with card_3:
    render_surface_card(
        eyebrow="Workflow Support",
        title="Helping review queues move more efficiently",
        body=(
            "Image models may help with prioritization, triage, and faster review in imaging-intensive environments."
        ),
    )

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

middle_left, middle_right = st.columns([1.05, 0.95], gap="large")
with middle_left:
    render_surface_card(
        eyebrow="Clinical Boundary",
        title="Why research promise does not equal deployment readiness",
        body=(
            "Even though the broader direction is promising, real deployment requires much more than a good model score on a public dataset. "
            "Medical image models face challenges related to labeling cost, image variability, scanner differences, domain shift, generalization, interpretability, and trust. "
            "Research results do not automatically translate into safe clinical use, and meaningful deployment requires validation, oversight, and close collaboration between technical and medical experts."
        ),
    )
with middle_right:
    render_surface_card(
        eyebrow="Project Role",
        title="How this project fits into the larger picture",
        body=(
            "This project should be interpreted as a technical exploration of model behavior, not a claim of clinical readiness. "
            "It is designed to help users understand data patterns, prediction tradeoffs, threshold effects, and augmentation results in a more interactive way. "
            "The value of the site is in making those dynamics easier to see and interpret responsibly."
        ),
    )

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

render_surface_card(
    eyebrow="Key Realities",
    title="What experienced teams still need to account for",
    body=(
        "Even strong research prototypes sit inside a larger system of data collection, labeling, validation, workflow design, and human oversight. "
        "These practical constraints shape whether a model is informative, trustworthy, and usable in real settings."
    ),
)

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

reality_1, reality_2, reality_3, reality_4 = st.columns(4, gap="large")
with reality_1:
    render_surface_card(
        eyebrow="Labeling Cost",
        title="Expert review is expensive and limited",
        body="Medical image labeling often depends on specialist time, which makes high-quality datasets costly to create and maintain.",
    )
with reality_2:
    render_surface_card(
        eyebrow="Domain Shift",
        title="Performance can change across settings",
        body="Models may struggle when scanner types, image quality, workflow conditions, or patient populations differ from training data.",
    )
with reality_3:
    render_surface_card(
        eyebrow="Interpretability",
        title="Users still need to understand model behavior",
        body="Healthcare teams need more than a score. They need context for why a model behaves the way it does and where it may fail.",
    )
with reality_4:
    render_surface_card(
        eyebrow="Implementation Gap",
        title="Research success does not guarantee field readiness",
        body="There is still a meaningful gap between public-dataset prototypes and systems that are validated for frontline clinical use.",
    )

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

bottom_left, bottom_right = st.columns([1.1, 0.9], gap="large")
with bottom_left:
    render_surface_card(
        eyebrow="Takeaway",
        title="Why this context matters for this project",
        body=(
            "This page is meant to frame the rest of the app responsibly. The model results are useful for understanding behavior, "
            "but they should be interpreted within the broader realities of healthcare data, deployment risk, and clinical accountability."
        ),
    )
with bottom_right:
    render_profile_links_card(eyebrow="Profile Links", title="Connect")
