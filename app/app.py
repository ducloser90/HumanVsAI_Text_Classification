import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from src.artifacts import load_artifacts, predict as run_inference
from src.config import DEFAULT_MODEL_DIR

st.set_page_config(
    page_title="Human vs Machine Text Classifier",
    page_icon="🤖",
    layout="centered"
)

MIN_CHARS = 1500

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #e0e0e0 !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        transition: all 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin-top: 0.5rem;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    .prediction-box {
        padding: 2rem;
        border-radius: 16px;
        border: none;
        margin: 1.5rem 0;
        font-size: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }

    .prediction-box:hover {
        transform: translateY(-4px);
    }

    .human-box {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.05) 0%, rgba(76, 175, 80, 0.15) 100%);
        border-left: 5px solid #4CAF50;
    }

    .machine-box {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.05) 0%, rgba(33, 150, 243, 0.15) 100%);
        border-left: 5px solid #2196F3;
    }

    .prediction-box h2 {
        background: none !important;
        -webkit-text-fill-color: inherit !important;
        margin-top: 0 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    .stAlert {
        border-radius: 12px;
        border-left: 5px solid #ff9800;
    }

    .stSpinner > div {
        border-top-color: #667eea !important;
    }

    hr {
        margin: 1.5rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e0e0e0, transparent);
    }

    #rt-char-counter {
        font-weight: 600 !important;
        font-size: 14px !important;
        margin-top: 8px !important;
        margin-bottom: 8px !important;
        padding: 8px 12px;
        border-radius: 8px;
        display: inline-block;
        transition: all 0.3s ease;
    }

    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load vectorizer and classifier via src.artifacts.load_artifacts."""
    try:
        vectorizer, clf = load_artifacts(model_dir=DEFAULT_MODEL_DIR)
        return vectorizer, clf, True
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None, False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, False


def predict_text(text, vectorizer, classifier):
    """Run inference and return (label, proba array)."""
    predictions = run_inference(text, vectorizer, classifier)
    raw_pred = predictions[0]
    X = vectorizer.transform([text])
    proba = classifier.predict_proba(X)[0]
    return str(raw_pred), proba


def map_prediction(pred):
    """
    Map raw classifier label → display string.
    Labels: 0 = Human, 1 = Machine.
    """
    pred_str = str(pred).strip().lower()
    if pred_str in ("0", "human"):
        return "Human"
    return "Machine"


def main():
    vectorizer, classifier, model_loaded = load_model()

    if not model_loaded:
        st.error("Failed to load model. Please check your setup.")
        st.stop()

    st.title("🤖 Human vs Machine Text Classifier")
    st.markdown("---")

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    user_text = st.text_area(
        "📝 Enter text to analyze:",
        height=300,
        placeholder="Paste your text here (minimum 1,500 characters)...",
        key="input_text",
        label_visibility="visible"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Text", key="clear_btn", help="Clear the text field"):
            if "input_text" in st.session_state:
                del st.session_state["input_text"]
            st.rerun()

    components.html(
        f"""
        <script>
        (function () {{
            const MIN = {MIN_CHARS};
            function fmt(n) {{ return n.toLocaleString(); }}

            function init() {{
                const parentDoc = window.parent.document;
                const textareas = parentDoc.querySelectorAll('.stTextArea textarea');
                if (!textareas.length) {{ setTimeout(init, 150); return; }}

                const textarea = textareas[0];

                const old = parentDoc.getElementById('rt-char-counter');
                if (old) old.remove();

                const counter = parentDoc.createElement('div');
                counter.id = 'rt-char-counter';

                const wrapper = textarea.closest('.stTextArea');
                if (wrapper) wrapper.insertAdjacentElement('afterend', counter);

                function update() {{
                    const n = textarea.value.length;
                    if (n === 0) {{
                        counter.textContent = '0 / ' + fmt(MIN) + ' characters minimum';
                        counter.style.color = '#F44336';
                        counter.style.background = 'rgba(244, 67, 54, 0.1)';
                    }} else if (n < MIN) {{
                        counter.textContent =
                            fmt(n) + ' / ' + fmt(MIN) + ' — ' + fmt(MIN - n) + ' more needed';
                        counter.style.color = '#FF9800';
                        counter.style.background = 'rgba(255, 152, 0, 0.1)';
                    }} else {{
                        counter.textContent = fmt(n) + ' characters ✓ Ready to analyze';
                        counter.style.color = '#4CAF50';
                        counter.style.background = 'rgba(76, 175, 80, 0.1)';
                    }}
                }}

                textarea.addEventListener('input', update);
                update();
            }}

            if (document.readyState === 'complete') {{ init(); }}
            else {{ window.addEventListener('load', init); }}
        }})();
        </script>
        """,
        height=0
    )

    if st.button("🔍 Analyze Text"):
        char_count = len(user_text)

        if char_count < MIN_CHARS:
            st.warning(
                f"⚠️ Text is {char_count:,} characters. "
                f"Please enter at least {MIN_CHARS:,} characters."
            )
        else:
            with st.spinner("🔄 Analyzing your text..."):
                try:
                    raw_pred, proba = predict_text(user_text, vectorizer, classifier)
                    prediction = map_prediction(raw_pred)
                    confidence = float(max(proba))

                    emoji     = "👤" if prediction == "Human" else "🤖"
                    box_class = "human-box" if prediction == "Human" else "machine-box"

                    st.markdown(f"""
                        <div class="prediction-box {box_class}">
                            <h2>{emoji} {prediction} Written Text</h2>
                            <p style="font-size: 1.2rem; margin: 12px 0 0 0; font-weight: 600;">
                                Confidence: <strong style="font-size: 1.3rem;">{confidence * 100:.1f}%</strong>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                    fig = go.Figure(data=[
                        go.Bar(
                            y=["Human", "Machine"],
                            x=[proba[0] * 100, proba[1] * 100],
                            orientation="h",
                            marker=dict(color=["#4CAF50", "#2196F3"], line=dict(width=0)),
                            text=[f"{proba[0] * 100:.1f}%", f"{proba[1] * 100:.1f}%"],
                            textposition="inside",
                            textfont=dict(size=16, color="white", family="Inter"),
                        )
                    ])
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.85rem; padding: 0.5rem 0;">
            Made by <strong style="color: #667eea;">Dergaoui Ayoub</strong>
            &nbsp;|&nbsp;
            <a href="https://github.com/ducloser90/HumanVsAI_Text_Classification"
               target="_blank"
               style="color: #667eea; text-decoration: none; font-weight: 600;">
                GitHub Repo
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()