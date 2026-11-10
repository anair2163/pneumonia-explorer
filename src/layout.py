from __future__ import annotations

import streamlit as st


APP_TITLE = "Chest X-Ray Classification and Augmentation Explorer"
APP_ICON = "🫁"
GITHUB_URL = "https://github.com/anair2163"
LINKEDIN_URL = "https://www.linkedin.com/in/ahannair"


def setup_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-identity">
                <div class='sidebar-brand'>Pneumonia Detection</div>
                <div class='sidebar-subtitle'>Image Augmentation Explorer</div>
                <div class="sidebar-copy">
                    Built by Ahan Nair
                    <br><br>
                    Data science, analytics, and machine learning projects focused on building interpretable, interactive experiences from complex data.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="sidebar-links">
                <a href="{GITHUB_URL}" target="_blank">GitHub</a>
                <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sidebar-nav-label'>Explore the project</div>", unsafe_allow_html=True)
        st.page_link("app.py", label="Overview", icon="🏠")
        st.page_link("pages/1_Why_This_Project.py", label="Why This Project", icon="🧭")
        st.page_link("pages/2_Data_and_Context.py", label="Data & Context", icon="🗂️")
        st.page_link("pages/3_Analytics_Exploration.py", label="Analytics Exploration", icon="📊")
        st.page_link("pages/4_Prediction_Lab.py", label="Prediction Lab", icon="🧪")
        st.page_link("pages/5_Model_Performance.py", label="Model Performance", icon="📈")
        st.page_link("pages/6_Synthetic_Augmentation.py", label="Image Augmentation", icon="🧬")
        st.page_link("pages/7_Key_Findings.py", label="Key Findings", icon="✅")
        st.page_link("pages/8_Real_World_Context.py", label="Real-World Context", icon="🌍")
        st.markdown("<div class='sidebar-note-label'>Reading order</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-guidance">
                Best reading order:
                motivation, data, analytics, prediction, performance, augmentation, then final findings.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_page_header(
    *,
    eyebrow: str,
    title: str,
    description: str,
    chips: list[str] | None = None,
) -> None:
    chip_markup = ""
    if chips:
        chip_markup = "<div class='chip-row'>" + "".join(
            f"<span class='chip'>{chip}</span>" for chip in chips
        ) + "</div>"

    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="eyebrow">{eyebrow}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-description">{description}</p>
            {chip_markup}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_home_hero(*, title: str, body: str) -> None:
    st.markdown(
        f"""
        <section class="feature-panel">
            <div class="panel-label">Project Focus</div>
            <h2 class="panel-title">{title}</h2>
            <p class="panel-body">{body}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_surface_card(*, eyebrow: str, title: str, body: str) -> None:
    formatted_body = body.replace("\n", "<br>")
    st.markdown(
        f"""
        <section class="surface-card">
            <div class="panel-label">{eyebrow}</div>
            <h3 class="card-title">{title}</h3>
            <p class="panel-body">{formatted_body}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_info_strip(*, items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items), gap="small")
    for column, (label, value) in zip(columns, items):
        with column:
            st.markdown(
                f"""
                <div class="info-pill">
                    <div class="info-pill-label">{label}</div>
                    <div class="info-pill-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_profile_links_card(*, eyebrow: str = "Profile Links", title: str = "Connect") -> None:
    render_surface_card(
        eyebrow=eyebrow,
        title=title,
        body=(
            f"<a href='{GITHUB_URL}' target='_blank'>GitHub</a><br>"
            f"<a href='{LINKEDIN_URL}' target='_blank'>LinkedIn</a>"
        ),
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f4f7fb;
            --surface: rgba(255, 255, 255, 0.96);
            --surface-strong: #ffffff;
            --line: rgba(15, 23, 42, 0.07);
            --text: #0f172a;
            --muted: #475569;
            --teal: #0f766e;
            --blue: #1e40af;
            --amber: #b45309;
            --shadow: 0 10px 28px rgba(15, 23, 42, 0.055);
            --radius-xl: 20px;
            --radius-lg: 16px;
        }

        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(30, 64, 175, 0.05), transparent 24%),
                radial-gradient(circle at 100% 0%, rgba(15, 118, 110, 0.06), transparent 22%),
                linear-gradient(180deg, #f8fafc 0%, var(--page-bg) 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.7rem;
            padding-bottom: 2.75rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.03em;
            color: var(--text);
        }

        p, li, .stMarkdown, .stCaption, label {
            color: var(--muted);
        }

        h2 {
            font-size: 1.45rem;
            margin-top: 0.35rem;
            margin-bottom: 0.6rem;
        }

        h3 {
            margin-top: 0.2rem;
            margin-bottom: 0.55rem;
            font-size: 1.15rem;
        }

        .stMarkdown p, .stMarkdown li {
            line-height: 1.66;
            font-size: 0.98rem;
        }

        .hero-shell,
        .feature-panel,
        .surface-card {
            background: var(--surface);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
        }

        .hero-shell {
            border-radius: 24px;
            padding: 1.8rem 1.9rem;
            margin-bottom: 0.95rem;
        }

        .feature-panel {
            border-radius: var(--radius-xl);
            padding: 1.2rem 1.3rem;
        }

        .surface-card {
            border-radius: var(--radius-lg);
            padding: 1.05rem 1.15rem;
            min-height: 100%;
        }

        .info-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.9rem;
            margin-bottom: 0.2rem;
        }

        .info-pill {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(15, 23, 42, 0.06);
            border-radius: 15px;
            padding: 0.9rem 0.95rem;
        }

        .info-pill-label {
            color: var(--blue);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 0.32rem;
        }

        .info-pill-value {
            color: var(--text);
            font-size: 0.96rem;
            font-weight: 650;
            line-height: 1.45;
        }

        .eyebrow,
        .panel-label {
            margin-bottom: 0.65rem;
            color: var(--blue);
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero-title {
            margin: 0;
            font-size: 2.55rem;
            line-height: 1.08;
        }

        .hero-description,
        .panel-body {
            margin: 0;
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.65;
            max-width: 78ch;
        }

        .panel-title,
        .card-title {
            margin-top: 0;
            margin-bottom: 0.45rem;
            font-size: 1.14rem;
            line-height: 1.3;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.95rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.065);
            color: var(--teal);
            padding: 0.34rem 0.68rem;
            font-size: 0.81rem;
            font-weight: 650;
        }

        .sidebar-links {
            display: flex;
            gap: 0.7rem;
            flex-wrap: wrap;
            margin-top: 0.45rem;
            margin-bottom: 1rem;
        }

        .sidebar-links a {
            color: var(--blue);
            text-decoration: none;
            font-weight: 650;
            display: inline-flex;
            align-items: center;
            padding: 0.34rem 0.62rem;
            border-radius: 999px;
            background: rgba(30, 64, 175, 0.06);
            border: 1px solid rgba(30, 64, 175, 0.08);
        }

        .panel-body a,
        .stMarkdown a {
            color: var(--blue);
            font-weight: 650;
            text-decoration: none;
        }

        .sidebar-brand {
            font-size: 1.12rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--text);
            margin-bottom: 0.1rem;
        }

        .sidebar-subtitle {
            font-size: 0.9rem;
            color: var(--muted);
            margin-bottom: 0.55rem;
            font-weight: 600;
        }

        .sidebar-copy,
        .sidebar-guidance {
            color: var(--muted);
            font-size: 0.89rem;
            line-height: 1.58;
        }

        .sidebar-identity {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 1rem 0.95rem 0.9rem;
            box-shadow: var(--shadow);
            margin-bottom: 0.85rem;
        }

        .sidebar-nav-label,
        .sidebar-note-label {
            color: var(--text);
            font-size: 0.8rem;
            font-weight: 750;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-top: 0.15rem;
            margin-bottom: 0.45rem;
        }

        .section-spacer {
            height: 0.85rem;
        }

        @media (max-width: 900px) {
            .info-strip {
                grid-template-columns: 1fr;
            }
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,250,252,0.98) 100%);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
            padding-left: 1.05rem;
            padding-right: 1.05rem;
        }

        [data-testid="stSidebarNav"],
        [data-testid="stSidebarNavSeparator"] {
            display: none !important;
        }

        [data-testid="stSidebar"] [data-testid="stPageLink"] {
            margin-bottom: 0.2rem;
        }

        [data-testid="stSidebar"] [data-testid="stPageLink"] a {
            padding: 0.72rem 0.8rem;
            border-radius: 12px !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
            transform: translateY(-1px);
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--line);
            border-radius: 15px;
            padding: 0.78rem 0.88rem;
            box-shadow: var(--shadow);
            min-height: 100%;
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 600;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] > div {
            color: var(--text);
            letter-spacing: -0.02em;
            font-size: clamp(1.15rem, 1.8vw, 1.55rem);
            line-height: 1.18;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere !important;
            word-break: break-word;
        }

        .stPlotlyChart,
        [data-testid="stDataFrame"],
        [data-testid="stFileUploader"],
        .stAlert,
        [data-testid="stTable"] {
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: var(--shadow);
            padding: 0.45rem;
        }

        .stPlotlyChart > div {
            border-radius: 14px;
        }

        .stPlotlyChart {
            min-height: 420px;
        }

        [data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"],
        [data-testid="baseButton-secondary"],
        [data-testid="stPageLink"] a,
        [data-testid="toastViewButton"] {
            border-radius: 12px !important;
        }

        [data-testid="stPageLink"] a {
            display: block;
            background: #2447B2 !important;
            color: #ffffff !important;
            border: 1px solid #2447B2 !important;
            padding: 0.72rem 0.86rem;
            box-shadow: var(--shadow);
            text-decoration: none !important;
        }

        .stButton > button,
        .stButton > button[kind],
        [data-testid="baseButton-secondary"],
        [data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"],
        [data-testid="stPageLink"] a,
        [data-testid="toastViewButton"] {
            border: 1px solid #2447B2 !important;
            background: #2447B2 !important;
            color: #FFFFFF !important;
            font-weight: 650 !important;
        }

        [data-testid="stPageLink"] a *,
        [data-testid="stPageLink"] a p,
        [data-testid="stPageLink"] a span,
        .stButton > button *,
        .stButton > button p,
        .stButton > button span,
        .stButton > button div,
        [data-testid="baseButton-secondary"] *,
        [data-testid="baseButton-secondary"] p,
        [data-testid="baseButton-secondary"] span,
        [data-testid="baseButton-secondary"] div,
        [data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"] *,
        [data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"] p,
        [data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"] span,
        [data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"] div,
        [data-testid="toastViewButton"] a,
        [data-testid="toastViewButton"] *,
        [data-testid="toastViewButton"] p,
        [data-testid="toastViewButton"] span,
        [data-testid="toastViewButton"] div {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        .stButton > button,
        .stButton > button[kind],
        [data-testid="baseButton-secondary"],
        [data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"] {
            border-radius: 12px !important;
            box-shadow: var(--shadow) !important;
        }

        .stButton > button:hover,
        .stButton > button[kind]:hover,
        [data-testid="baseButton-secondary"]:hover,
        [data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"]:hover,
        [data-testid="stPageLink"] a:hover,
        [data-testid="toastViewButton"]:hover {
            background: #1D3990 !important;
            border-color: #1D3990 !important;
            color: #ffffff !important;
        }

        .stButton > button:focus,
        .stButton > button:focus-visible,
        .stButton > button[kind]:focus,
        .stButton > button[kind]:focus-visible,
        [data-testid="baseButton-secondary"]:focus,
        [data-testid="baseButton-secondary"]:focus-visible,
        [data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"]:focus,
        [data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"]:focus-visible,
        [data-testid="stPageLink"] a:focus,
        [data-testid="stPageLink"] a:focus-visible,
        [data-testid="toastViewButton"]:focus,
        [data-testid="toastViewButton"]:focus-visible {
            outline: 3px solid rgba(36, 71, 178, 0.28) !important;
            outline-offset: 2px !important;
            background: #1D3990 !important;
            border-color: #1D3990 !important;
            color: #ffffff !important;
        }

        .stButton > button:active,
        .stButton > button[kind]:active,
        [data-testid="baseButton-secondary"]:active,
        [data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"]:active,
        [data-testid="stPageLink"] a:active,
        [data-testid="toastViewButton"]:active {
            background: #172C74 !important;
            border-color: #172C74 !important;
            color: #ffffff !important;
        }

        [data-testid="toastViewButton"] {
            background: #2447B2 !important;
            color: #FFFFFF !important;
            border: 1px solid #2447B2 !important;
            border-radius: 999px !important;
            opacity: 1 !important;
            box-shadow: 0 8px 20px rgba(36, 71, 178, 0.18) !important;
        }

        [data-testid="toastViewButton"] a {
            color: #FFFFFF !important;
            text-decoration: none !important;
            opacity: 1 !important;
        }

        [data-testid="toastViewButton"] *,
        [data-testid="toastViewButton"] a,
        [data-testid="toastViewButton"] p,
        [data-testid="toastViewButton"] span,
        [data-testid="toastViewButton"] div {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            opacity: 1 !important;
        }

        [data-testid="toastViewButton"]:hover,
        [data-testid="toastViewButton"]:hover a,
        [data-testid="toastViewButton"]:hover *,
        [data-testid="toastViewButton"]:hover p,
        [data-testid="toastViewButton"]:hover span,
        [data-testid="toastViewButton"]:hover div {
            background: #1D3990 !important;
            border-color: #1D3990 !important;
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }

        [data-testid="toastViewButton"]:focus,
        [data-testid="toastViewButton"]:focus-visible,
        [data-testid="toastViewButton"]:focus a,
        [data-testid="toastViewButton"]:focus-visible a,
        [data-testid="toastViewButton"]:focus *,
        [data-testid="toastViewButton"]:focus-visible *,
        [data-testid="toastViewButton"]:focus p,
        [data-testid="toastViewButton"]:focus-visible p,
        [data-testid="toastViewButton"]:focus span,
        [data-testid="toastViewButton"]:focus-visible span,
        [data-testid="toastViewButton"]:focus div,
        [data-testid="toastViewButton"]:focus-visible div {
            background: #1D3990 !important;
            border-color: #1D3990 !important;
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            outline: 3px solid rgba(36, 71, 178, 0.3) !important;
            outline-offset: 2px !important;
            box-shadow: 0 0 0 4px rgba(36, 71, 178, 0.12) !important;
        }

        [data-testid="toastViewButton"]:active,
        [data-testid="toastViewButton"]:active a,
        [data-testid="toastViewButton"]:active *,
        [data-testid="toastViewButton"]:active p,
        [data-testid="toastViewButton"]:active span,
        [data-testid="toastViewButton"]:active div {
            background: #16306F !important;
            border-color: #16306F !important;
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }

        [data-testid="toastViewButton"] {
            border-radius: 999px !important;
            padding: 0.2rem 0.6rem !important;
            line-height: 1.3rem !important;
            box-shadow: none !important;
        }

        .stRadio > div,
        .stSelectbox > div,
        .stMultiSelect > div {
            border-radius: 14px;
        }

        .stRadio label,
        .stSelectbox label,
        .stMultiSelect label,
        .stSlider label,
        .stFileUploader label {
            font-weight: 650;
            color: var(--text);
        }

        .stCaption {
            font-size: 0.84rem;
        }

        .stSlider {
            padding-top: 0.45rem;
        }

        @media (max-width: 1100px) {
            .hero-title {
                font-size: 2.2rem;
            }

            .block-container {
                padding-top: 1.4rem;
                padding-left: 1.2rem;
                padding-right: 1.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
