"""
Streamlit Web App — Chest X-Ray Cancer Detection
With Claude API integration for AI-powered diagnostic reports.

Run: streamlit run app/streamlit_app.py
"""

import os
import sys
import io
from pathlib import Path

import streamlit as st
import numpy as np
import torch
import anthropic
from PIL import Image
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import predict, predict_with_gradcam, preprocess_image
from src.dataset import CLASS_NAMES

load_dotenv()
matplotlib.use("Agg")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray AI Analyzer",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1e3a5f;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #f0f9ff;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .warning-card {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .disclaimer {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        font-size: 0.85rem;
        color: #b91c1c;
        margin-top: 2rem;
    }
    .metric-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stProgress > div > div { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Chest_Xray_PA_3-8-2010.png/220px-Chest_Xray_PA_3-8-2010.png", width=180)
    st.markdown("## ⚙️ Settings")

    model_path = st.text_input("Model Path", value="models/best_model.pth")
    show_gradcam = st.toggle("Show Grad-CAM Heatmap", value=True)
    show_claude = st.toggle("Generate Claude AI Report", value=True)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses **EfficientNetB3** trained on the 
    [Chest X-Ray Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
    to classify chest X-rays as **Normal** or **Pneumonia**.

    **Claude API** is used to generate plain-language diagnostic reports.
    """)

    st.markdown("---")
    st.markdown("### Disclaimer")
    st.markdown("*For educational use only. Not a medical device.*")


# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header"> Chest X-Ray AI Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep learning-powered detection with AI-generated diagnostic reports</div>', unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader(
    "Upload a Chest X-Ray Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a frontal chest X-ray (PA or AP view)",
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    tmp_path = Path("tmp_upload.jpg")
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Layout: image + results
    col1, col2 = st.columns([1, 1.4], gap="large")

    with col1:
        st.markdown("#### Uploaded X-Ray")
        pil_img = Image.open(tmp_path).convert("RGB")
        st.image(pil_img, use_column_width=True, caption="Input Image")

    with col2:
        st.markdown("#### Analysis Results")

        # Check model exists
        if not Path(model_path).exists():
            st.error(
                f"Model not found at `{model_path}`\n\n"
                "Please train the model first:\n```bash\npython src/train.py\n```"
            )
            st.stop()

        # Run inference
        with st.spinner("Running inference..."):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                if show_gradcam:
                    result, overlay = predict_with_gradcam(
                        str(tmp_path), model_path, device=device
                    )
                else:
                    result = predict(str(tmp_path), model_path, device=device)
                    overlay = None

            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.stop()

        # Display prediction
        class_name  = result["class_name"]
        confidence  = result["confidence"]
        probs       = result["probabilities"]

        if class_name == "NORMAL":
            st.markdown(f"""
            <div class="result-card">
                <h3 style="color:#16a34a; margin:0">✅ {class_name}</h3>
                <p style="margin:0.3rem 0 0 0">Confidence: <strong>{confidence:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-card">
                <h3 style="color:#dc2626; margin:0">⚠️ {class_name} DETECTED</h3>
                <p style="margin:0.3rem 0 0 0">Confidence: <strong>{confidence:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        # Probability bars
        st.markdown("**Class Probabilities:**")
        for cls_name, prob in probs.items():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(prob, text=cls_name)
            with col_b:
                st.write(f"`{prob:.2%}`")

        # Device info
        st.caption(f"🖥️ Inference on: {str(device).upper()}")

    # Grad-CAM visualization
    if show_gradcam and overlay is not None:
        st.markdown("---")
        st.markdown("#### 🔥 Grad-CAM Heatmap")
        st.markdown("*Areas the model focused on when making its prediction.*")

        col_orig, col_heat = st.columns(2)
        with col_orig:
            st.image(pil_img, caption="Original X-Ray", use_column_width=True)
        with col_heat:
            overlay_img = (overlay * 255).astype(np.uint8)
            st.image(overlay_img, caption="Grad-CAM Overlay", use_column_width=True)

    # Claude AI Report
    if show_claude:
        st.markdown("---")
        st.markdown("#### 🤖 Claude AI Diagnostic Report")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            st.warning("ANTHROPIC_API_KEY not set. Add it to your .env file to enable AI reports.")
        else:
            if st.button("Generate AI Report", type="primary"):
                with st.spinner("Claude is writing your report..."):
                    try:
                        client = anthropic.Anthropic(api_key=api_key)

                        probs_text = "\n".join([
                            f"- {k}: {v:.2%}" for k, v in probs.items()
                        ])

                        prompt = f"""You are an expert radiologist AI assistant. A deep learning model has analyzed a chest X-ray with the following results:

**Prediction:** {class_name}
**Confidence:** {confidence:.2%}
**Class Probabilities:**
{probs_text}

Please provide a concise radiology-style report that includes:
1. **Findings** — What the model detected and what this might indicate
2. **Clinical Significance** — Why this finding matters in plain language
3. **Confidence Assessment** — What the confidence level means for reliability
4. **Recommended Next Steps** — What the patient or clinician should do next
5. **Limitations** — Important caveats about AI-based analysis

Keep the tone professional but accessible to a non-medical audience. Be clear this is an AI analysis for educational purposes only, not a clinical diagnosis. Format the response with clear section headers."""

                        message = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=1000,
                            messages=[{"role": "user", "content": prompt}],
                        )

                        report = message.content[0].text
                        st.markdown(report)

                    except anthropic.AuthenticationError:
                        st.error("Invalid Claude API key. Check your ANTHROPIC_API_KEY in .env")
                    except Exception as e:
                        st.error(f"Claude API error: {e}")

    # Cleanup
    if tmp_path.exists():
        tmp_path.unlink()

else:
    # Empty state
    st.markdown("---")
    st.info("👆Upload a chest X-ray image to get started.")

    # Example info
    with st.expander("at this model can detect"):
        st.markdown("""
        | Class | Description |
        |---|---|
        | **NORMAL** | Healthy chest X-ray with no signs of infection or abnormality |
        | **PNEUMONIA** | Chest X-ray showing signs of pneumonia (bacterial or viral) |
        """)

    with st.expander("How to train this model"):
        st.code("""
# 1. Download dataset
python data/download_dataset.py

# 2. Train the model (~20 epochs)
python src/train.py --epochs 20 --batch_size 32

# 3. Evaluate performance
python src/evaluate.py

# 4. Launch this app
streamlit run app/streamlit_app.py
        """, language="bash")

# Footer disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Medical Disclaimer:</strong> This application is for <strong>educational and research purposes only</strong>. 
    It is NOT a medical device and must NOT be used for clinical diagnosis, treatment decisions, or medical advice. 
    Always consult a qualified medical professional for health concerns.
</div>
""", unsafe_allow_html=True)
