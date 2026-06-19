"""
Shared theme system for Videx. All pages import this to get consistent styling.
"""
import streamlit as st


def get_theme():
    """Return the white + dark blue light-mode theme color dict."""
    return {
        "bg_main": "#f0f4ff",
        "bg_sidebar": "linear-gradient(180deg, #0f1e3d 0%, #0a1628 100%)",
        "bg_card": "#ffffff",
        "bg_card_hover": "#f5f8ff",
        "border_card": "rgba(29,78,216,0.12)",
        "text_primary": "#0f1e3d",
        "text_secondary": "#3d5a80",
        "text_muted": "#6b8aad",
        "text_heading": "#0a1628",
        "text_sidebar": "#b8d4f0",
        "text_explanation": "#4a6a8a",
        "accent": "#1d4ed8",
        "accent2": "#1e3a8a",
        "success": "#16a34a",
        "warning": "#d97706",
        "error": "#dc2626",
        "code_bg": "rgba(29,78,216,0.07)",
        "card_gradient": "linear-gradient(135deg, rgba(29,78,216,0.04), rgba(30,58,138,0.04))",
        "error_bg": "rgba(220,38,38,0.07)",
        "error_border": "rgba(220,38,38,0.2)",
        "success_bg": "rgba(22,163,74,0.08)",
        "success_border": "rgba(22,163,74,0.2)",
        "warning_bg": "rgba(217,119,6,0.08)",
        "warning_border": "rgba(217,119,6,0.2)",
    }


def get_common_css(t=None):
    """Return the common CSS block using theme colors."""
    if t is None:
        t = get_theme()

    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header [data-testid="stHeader"] {{ background: transparent !important; }}
    [data-testid="stMainMenu"] {{visibility: hidden;}}
    [data-testid="stToolbar"] [data-testid="stBaseButton-headerNoPadding"] {{display: none;}}

    .stApp {{ background: {t['bg_main']} !important; }}
    * {{ font-family: 'Inter', sans-serif; }}

    /* Sidebar — dark navy with light text */
    [data-testid="stSidebar"] {{
        background: {t['bg_sidebar']} !important;
        border-right: 1px solid rgba(29,78,216,0.15);
    }}

    /* Force ALL text inside sidebar to be light */
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {{
        color: {t['text_sidebar']} !important;
    }}

    /* Headings in sidebar */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {{
        color: #e2eeff !important;
    }}

    /* Nav links — every selector variant Streamlit uses */
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] a *,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a *,
    [data-testid="stSidebar"] [data-testid="stSidebarNavLink"],
    [data-testid="stSidebar"] [data-testid="stSidebarNavLink"] *,
    [data-testid="stSidebar"] [data-testid="stSidebarNavLink"] p,
    [data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span,
    [data-testid="stSidebar"] nav a,
    [data-testid="stSidebar"] nav a span,
    [data-testid="stSidebar"] nav li {{
        color: {t['text_sidebar']} !important;
    }}

    /* Nav link hover */
    [data-testid="stSidebarNavLink"]:hover,
    [data-testid="stSidebarNavLink"]:hover * {{
        color: #ffffff !important;
    }}

    /* Active/selected nav link */
    [data-testid="stSidebarNavLink"][aria-current="page"],
    [data-testid="stSidebarNavLink"][aria-current="page"] * {{
        color: #ffffff !important;
        background: rgba(59,130,246,0.2) !important;
        border-radius: 8px;
    }}

    /* Labels and form controls */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label span,
    [data-testid="stSidebar"] .stRadio label {{
        color: {t['text_sidebar']} !important;
    }}

    /* Buttons inside sidebar should keep their own styling */
    [data-testid="stSidebar"] .stButton button {{
        color: {t['text_heading']} !important;
    }}
    [data-testid="stSidebar"] .stDownloadButton button {{
        color: {t['text_heading']} !important;
    }}

    /* Main content text — dark on light */
    .page-header {{ text-align: center; padding: 2rem 0 1rem; }}
    .page-title {{
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, {t['accent']}, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .page-subtitle {{ color: {t['text_secondary']}; font-size: 1rem; }}

    .stApp p, .stApp li, .stApp div {{
        color: {t['text_primary']};
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: {t['text_heading']} !important;
    }}
    .stApp label, .stApp .stSelectbox label, .stApp .stRadio label,
    .stApp .stCheckbox label, .stApp .stSlider label,
    .stApp .stTextInput label, .stApp .stNumberInput label,
    .stApp .stFileUploader label {{
        color: {t['text_primary']} !important;
    }}
    .stApp .stRadio div[role="radiogroup"] label p,
    .stApp .stCheckbox label span {{
        color: {t['text_primary']} !important;
    }}
    .stApp [data-baseweb="select"] span,
    .stApp [data-baseweb="input"] input,
    .stApp .stSelectbox div[data-baseweb="select"] > div {{
        color: {t['text_primary']} !important;
    }}
    .stApp .stTabs [data-baseweb="tab"] {{
        color: {t['text_secondary']} !important;
    }}
    .stApp .stTabs [aria-selected="true"] {{
        color: {t['text_primary']} !important;
    }}
    .stApp .stExpander summary span {{
        color: {t['text_primary']} !important;
    }}
    .stApp .stChatMessage p {{
        color: {t['text_primary']} !important;
    }}
    .stApp .stMarkdown {{
        color: {t['text_primary']};
    }}
    .stApp .stMarkdown p, .stApp .stMarkdown li,
    .stApp .stMarkdown td, .stApp .stMarkdown th {{
        color: {t['text_primary']} !important;
    }}
    .stApp .stMarkdown code {{
        color: {t['accent']} !important;
        background: {t['code_bg']};
        padding: 1px 5px; border-radius: 4px;
    }}
    .stApp .stButton button {{
        color: {t['text_primary']};
    }}
    .stApp .stDownloadButton button {{
        color: {t['text_primary']};
    }}
    .stApp .stAlert p {{
        color: inherit !important;
    }}
    .stApp .stCaption, .stApp small {{
        color: {t['text_muted']} !important;
    }}
    .stApp [data-testid="stMetricValue"] {{
        color: {t['text_primary']} !important;
    }}
    .stApp [data-testid="stMetricLabel"] {{
        color: {t['text_secondary']} !important;
    }}

    /* Chat input box */
    [data-testid="stChatInput"] {{
        background: #ffffff !important;
        border: 1px solid rgba(29,78,216,0.2) !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: {t['text_primary']} !important;
    }}

    /* Expander background */
    .stApp .stExpander {{
        background: #ffffff;
        border: 1px solid {t['border_card']};
        border-radius: 12px;
    }}
</style>
"""
