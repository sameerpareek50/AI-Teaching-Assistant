"""
Shared theme system for Videx. All pages import this to get
consistent light/dark mode support.
"""
import streamlit as st


def get_theme():
    """Return the dark theme color dict."""
    return {
        "bg_main": "linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%)",
        "bg_sidebar": "linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%)",
        "bg_card": "rgba(255,255,255,0.03)",
        "bg_card_hover": "rgba(255,255,255,0.06)",
        "border_card": "rgba(255,255,255,0.08)",
        "text_primary": "#d0d0f0",
        "text_secondary": "#8888aa",
        "text_muted": "#6868a0",
        "text_heading": "#c0c8ff",
        "text_sidebar": "#b8b8cc",
        "text_explanation": "#9898b0",
        "accent": "#667eea",
        "accent2": "#764ba2",
        "success": "#68d391",
        "warning": "#f0c83c",
        "error": "#ea6666",
        "code_bg": "rgba(102,126,234,0.15)",
        "card_gradient": "linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15))",
        "error_bg": "rgba(234,102,102,0.08)",
        "error_border": "rgba(234,102,102,0.2)",
        "success_bg": "rgba(72,187,120,0.15)",
        "success_border": "rgba(72,187,120,0.3)",
        "warning_bg": "rgba(240,200,60,0.2)",
        "warning_border": "rgba(240,200,60,0.3)",
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
    .stApp {{ background: {t['bg_main']}; }}
    * {{ font-family: 'Inter', sans-serif; }}
    [data-testid="stSidebar"] {{
        background: {t['bg_sidebar']};
        border-right: 1px solid {t['border_card']};
    }}
    [data-testid="stSidebar"] .stMarkdown p {{ color: {t['text_sidebar']} !important; }}
    [data-testid="stSidebar"] h2 {{ color: {t['text_primary']} !important; }}

    /* Sidebar navigation links */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a span,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
        color: {t['text_primary']} !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] li {{
        color: {t['text_primary']} !important;
    }}
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] a span {{
        color: {t['text_primary']} !important;
    }}
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {{
        color: {t['text_sidebar']} !important;
    }}
    .page-header {{ text-align: center; padding: 2rem 0 1rem; }}
    .page-title {{
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, {t['accent']}, {t['accent2']});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .page-subtitle {{ color: {t['text_secondary']}; font-size: 1rem; }}

    /* Global text color overrides for all Streamlit elements */
    .stApp, .stApp * {{
        --text-color: {t['text_primary']};
    }}
    .stApp p, .stApp li, .stApp span, .stApp div {{
        color: {t['text_primary']};
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: {t['text_primary']} !important;
    }}
    .stApp label, .stApp .stSelectbox label, .stApp .stRadio label,
    .stApp .stCheckbox label, .stApp .stSlider label,
    .stApp .stTextInput label, .stApp .stNumberInput label,
    .stApp .stFileUploader label {{
        color: {t['text_primary']} !important;
    }}
    /* Radio button and checkbox text */
    .stApp .stRadio div[role="radiogroup"] label p,
    .stApp .stCheckbox label span {{
        color: {t['text_primary']} !important;
    }}
    /* Selectbox and input text */
    .stApp [data-baseweb="select"] span,
    .stApp [data-baseweb="input"] input,
    .stApp .stSelectbox div[data-baseweb="select"] > div {{
        color: {t['text_primary']} !important;
    }}
    /* Tabs */
    .stApp .stTabs [data-baseweb="tab"] {{
        color: {t['text_secondary']} !important;
    }}
    .stApp .stTabs [aria-selected="true"] {{
        color: {t['text_primary']} !important;
    }}
    /* Expander */
    .stApp .stExpander summary span {{
        color: {t['text_primary']} !important;
    }}
    /* Chat messages */
    .stApp .stChatMessage p {{
        color: {t['text_primary']} !important;
    }}
    /* Markdown rendered content */
    .stApp .stMarkdown {{
        color: {t['text_primary']};
    }}
    .stApp .stMarkdown p, .stApp .stMarkdown li,
    .stApp .stMarkdown td, .stApp .stMarkdown th {{
        color: {t['text_primary']} !important;
    }}
    .stApp .stMarkdown code {{
        color: {t['text_heading']} !important;
    }}
    /* Button text - keep readable */
    .stApp .stButton button {{
        color: {t['text_primary']};
    }}
    .stApp .stDownloadButton button {{
        color: {t['text_primary']};
    }}
    /* Info, warning, error, success boxes */
    .stApp .stAlert p {{
        color: inherit !important;
    }}
    /* Caption and small text */
    .stApp .stCaption, .stApp small {{
        color: {t['text_muted']} !important;
    }}
    /* Metrics */
    .stApp [data-testid="stMetricValue"] {{
        color: {t['text_primary']} !important;
    }}
    .stApp [data-testid="stMetricLabel"] {{
        color: {t['text_secondary']} !important;
    }}
</style>
"""
