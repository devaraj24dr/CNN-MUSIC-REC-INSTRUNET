# ============================================
# 🎵 InstruNet Dashboard v2.0
# CNN-Based Music Instrument Recognition System
# Production-Level Analytics Dashboard
# ============================================

import os
import io
import json
import pickle
import warnings
import datetime

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="InstruNet Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# CUSTOM CSS — Production Premium Dark Theme
# ============================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ---- Base ---- */
    .stApp {
        background: #0a0e1a !important;
        color: #e6edf3 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ---- Animated gradient blurs ---- */
    .bg-blur-1, .bg-blur-2 {
        position: fixed; border-radius: 50%; filter: blur(140px); opacity: 0.18;
        pointer-events: none; z-index: 0;
    }
    .bg-blur-1 { width: 600px; height: 600px; top: -150px; left: -100px;
        background: radial-gradient(circle, #6366f1 0%, transparent 70%);
        animation: float1 20s ease-in-out infinite; }
    .bg-blur-2 { width: 500px; height: 500px; bottom: -120px; right: -80px;
        background: radial-gradient(circle, #f97316 0%, transparent 70%);
        animation: float2 24s ease-in-out infinite; }
    @keyframes float1 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(60px,40px)} }
    @keyframes float2 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-50px,-30px)} }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: rgba(13, 18, 32, 0.95) !important;
        border-right: 1px solid rgba(99,102,241,0.12) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #e6edf3 !important; }

    h1, h2, h3, h4, h5, h6 {
        color: #e6edf3 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ---- Section Headers ---- */
    .section-header {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
        text-transform: uppercase; color: #6366f1; margin-top: 2rem;
        margin-bottom: 1rem; padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    }

    /* ---- Metric Cards ---- */
    .metric-card {
        background: rgba(22, 27, 45, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px; padding: 1.2rem 1.4rem;
        position: relative; overflow: hidden;
        transition: all 0.3s ease;
        min-height: 100px;
    }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #6366f1, #f97316);
        border-radius: 16px 16px 0 0;
    }
    .metric-card:hover {
        border-color: #6366f1; transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(99, 102, 241, 0.18);
    }
    .metric-label {
        font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: #8b949e; margin-bottom: 0.3rem;
        word-wrap: break-word;
    }
    .metric-value {
        font-size: 1.45rem; font-weight: 800; color: #e6edf3;
        line-height: 1.2; word-wrap: break-word;
    }
    .metric-sub {
        font-size: 0.68rem; color: #818cf8; margin-top: 0.25rem;
        font-weight: 500; word-wrap: break-word;
    }
    .metric-sub-green { font-size: 0.7rem; color: #22c55e; margin-top: 0.25rem; font-weight: 500; }
    .metric-sub-red { font-size: 0.7rem; color: #ef4444; margin-top: 0.25rem; font-weight: 500; }

    /* ---- Glass Card ---- */
    .glass-card {
        background: rgba(22, 27, 45, 0.65);
        backdrop-filter: blur(18px);
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 20px; padding: 1.8rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative; z-index: 1;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(22, 27, 45, 0.8); border-radius: 10px;
        color: #8b949e; border: 1px solid rgba(99, 102, 241, 0.15);
        font-weight: 600; font-size: 0.85rem; padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important; border-color: #6366f1 !important;
    }

    /* ---- File Uploader ---- */
    section[data-testid="stFileUploader"] {
        background: rgba(22, 27, 45, 0.6);
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 16px; padding: 1rem;
    }
    section[data-testid="stFileUploader"]:hover { border-color: #6366f1; }

    /* ---- Top Prediction Card ---- */
    .top-prediction-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(249,115,22,0.08) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 20px; padding: 2rem; text-align: center;
        box-shadow: 0 12px 40px rgba(99,102,241,0.12);
    }
    .top-pred-label {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
        text-transform: uppercase; color: #f97316; margin-bottom: 0.4rem;
    }
    .top-pred-name {
        font-size: 2.2rem; font-weight: 900;
        background: linear-gradient(135deg, #818cf8, #f97316);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; line-height: 1.2; margin-bottom: 0.3rem;
    }
    .top-pred-conf { font-size: 1.1rem; font-weight: 600; color: #a5b4fc; }

    /* ---- Instrument Cards ---- */
    .instr-present {
        background: rgba(34,197,94,0.05); border: 1px solid rgba(34,197,94,0.2);
        border-left: 3px solid #22c55e; border-radius: 12px;
        padding: 0.8rem 1rem; margin-bottom: 0.5rem;
    }
    .instr-absent {
        background: rgba(239,68,68,0.04); border: 1px solid rgba(239,68,68,0.15);
        border-left: 3px solid #ef4444; border-radius: 12px;
        padding: 0.8rem 1rem; margin-bottom: 0.5rem; opacity: 0.7;
    }

    /* ---- Progress bar ---- */
    .prog-bg {
        width: 100%; height: 5px; background: rgba(99,102,241,0.1);
        border-radius: 3px; overflow: hidden; margin-top: 0.3rem;
    }
    .prog-fill-green {
        height: 100%; border-radius: 3px;
        background: linear-gradient(90deg, #22c55e, #86efac);
    }
    .prog-fill-red {
        height: 100%; border-radius: 3px;
        background: linear-gradient(90deg, #ef4444, #f87171);
    }

    /* ---- Hide defaults ---- */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

    /* ---- Footer ---- */
    .app-footer {
        text-align: center; color: #475569; font-size: 0.72rem;
        padding: 2rem 0 1rem; border-top: 1px solid rgba(99,102,241,0.1);
        margin-top: 3rem;
    }
</style>
"""

# ============================================
# CONSTANTS
# ============================================
INSTRUMENT_COLORS = [
    "#818cf8", "#22c55e", "#3b82f6", "#d8b4fe",
    "#f97316", "#67e8f9", "#86efac", "#fb923c",
    "#f87171", "#7dd3fc", "#f472b6",
]

INSTRUMENT_NAMES = {
    "cel": "Cello", "cla": "Clarinet", "flu": "Flute",
    "gac": "Acoustic Guitar", "gel": "Electric Guitar",
    "org": "Organ", "pia": "Piano", "sax": "Saxophone",
    "tru": "Trumpet", "vio": "Violin", "voi": "Voice",
}

INSTRUMENT_EMOJIS = {
    "Cello": "🎻", "Clarinet": "🎶", "Flute": "🪈",
    "Acoustic Guitar": "🎸", "Electric Guitar": "🎸",
    "Organ": "🎹", "Piano": "🎹", "Saxophone": "🎷",
    "Trumpet": "🎺", "Violin": "🎻", "Voice": "🎤",
}

# ============================================
# MODEL DEFINITION
# ============================================
class CustomCNN(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================
# MODEL LOADING
# ============================================
@st.cache_resource
def load_model_bundle():
    pkl_path = os.path.join(os.path.dirname(__file__), "models", "instrument_classifier_full.pkl")
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    num_classes = bundle["architecture_config"]["num_classes"]
    model = CustomCNN(num_classes=num_classes)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, bundle


# ============================================
# AUDIO PREPROCESSING
# ============================================
def preprocess_audio(audio_bytes, config):
    sr = config.get("sample_rate", 22050)
    duration = config.get("duration_seconds", 3)
    n_mels = config.get("n_mels", 128)
    hop = config.get("hop_length", 512)

    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
    segment_len = sr * duration
    segments = [audio[i:i+segment_len] for i in range(0, len(audio), segment_len) if len(audio[i:i+segment_len]) >= sr]
    if not segments:
        segments = [audio]

    all_tensors = []
    for seg in segments:
        target_len = sr * duration
        if len(seg) < target_len:
            seg = np.pad(seg, (0, target_len - len(seg)))
        else:
            seg = seg[:target_len]
        mel_spec = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=n_mels, hop_length=hop)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
        target_w = config.get("target_shape", (128, 128))[1]
        if mel_db.shape[1] < target_w:
            mel_db = np.pad(mel_db, ((0, 0), (0, target_w - mel_db.shape[1])), mode="constant")
        else:
            mel_db = mel_db[:, :target_w]
        channels = config.get("channels", 1)
        if channels == 3:
            spec = np.stack([mel_db]*3, axis=0)
        else:
            spec = np.expand_dims(mel_db, axis=0)
        all_tensors.append(torch.tensor(spec, dtype=torch.float32).unsqueeze(0))
    return all_tensors, audio, sr


def run_prediction(model, audio_bytes, config, class_names, instr_names):
    """Run prediction on a single audio file. Returns dict with results."""
    tensors, audio_signal, audio_sr = preprocess_audio(audio_bytes, config)
    all_probs = []
    with torch.no_grad():
        for t in tensors:
            output = model(t)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

    timeline_data = np.array(all_probs)
    avg_probs = np.mean(timeline_data, axis=0)
    predictions = []
    for i, cls in enumerate(class_names):
        predictions.append({
            "Instrument": instr_names.get(cls, cls),
            "Code": cls,
            "Confidence": round(float(avg_probs[i]) * 100, 2),
        })
    pred_df = pd.DataFrame(predictions).sort_values("Confidence", ascending=False)
    return {
        "predictions": pred_df,
        "timeline": timeline_data,
        "audio_signal": audio_signal,
        "audio_sr": audio_sr,
        "avg_probs": avg_probs,
        "top_instrument": pred_df.iloc[0]["Instrument"],
        "top_confidence": pred_df.iloc[0]["Confidence"],
    }


# ============================================
# PLOTLY THEME HELPER
# ============================================
def apply_theme(fig, height=400, title="", x_title="", y_title="", show_legend=True, x_range=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1220",
        font=dict(family="Inter", color="#e6edf3", size=12),
        margin=dict(l=55, r=25, t=55, b=50),
        hoverlabel=dict(bgcolor="#161b2e", font_color="#e6edf3", bordercolor="#30363d"),
        height=height, showlegend=show_legend,
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#e6edf3")))
    xkw = dict(gridcolor="#1e2438", zerolinecolor="#1e2438", automargin=True)
    ykw = dict(gridcolor="#1e2438", zerolinecolor="#1e2438", automargin=True)
    if x_title: xkw["title"] = dict(text=x_title, font=dict(size=11))
    if y_title: ykw["title"] = dict(text=y_title, font=dict(size=11))
    if x_range: xkw["range"] = x_range
    fig.update_xaxes(**xkw)
    fig.update_yaxes(**ykw)
    return fig


def metric_card(label, value, sub, sub_class="metric-sub"):
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="{sub_class}">{sub}</div>
    </div>"""


# ============================================
# TAB 1: MODEL OVERVIEW
# ============================================
def render_model_overview(bundle, class_names, instr_names):
    eval_results = bundle.get("evaluation_results", {})
    report = eval_results.get("classification_report", {})
    arch = bundle.get("architecture_config", {})
    meta = bundle.get("metadata", {})

    # --- KPI Row ---
    accuracy = eval_results.get("accuracy", 0) * 100
    precision_val = eval_results.get("precision", 0) * 100
    recall_val = eval_results.get("recall", 0) * 100
    f1_val = eval_results.get("f1_score", 0) * 100

    st.markdown('<div class="section-header">📊 Model Performance Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(metric_card("Accuracy", f"{accuracy:.1f}%", "Weighted avg"), unsafe_allow_html=True)
    k2.markdown(metric_card("Precision", f"{precision_val:.1f}%", "Weighted avg"), unsafe_allow_html=True)
    k3.markdown(metric_card("Recall", f"{recall_val:.1f}%", "Weighted avg"), unsafe_allow_html=True)
    k4.markdown(metric_card("F1 Score", f"{f1_val:.1f}%", "Harmonic mean"), unsafe_allow_html=True)

    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(metric_card("Backbone", "CustomCNN", "4-layer conv"), unsafe_allow_html=True)
    m2.markdown(metric_card("Classes", str(arch.get("num_classes", 11)), "Instruments"), unsafe_allow_html=True)
    m3.markdown(metric_card("Input", "1×128×128", "Mel spectro"), unsafe_allow_html=True)
    m4.markdown(metric_card("Version", meta.get("version", "2.0"), "PyTorch"), unsafe_allow_html=True)

    # --- Charts Row ---
    st.markdown('<div class="section-header">📈 Per-Class Performance</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        if report:
            f1_data = [{"Instrument": instr_names.get(c, c), "F1 Score": report.get(c, {}).get("f1-score", 0) * 100}
                       for c in class_names if c in report]
            f1_df = pd.DataFrame(f1_data).sort_values("F1 Score", ascending=True)
            fig = go.Figure(go.Bar(
                y=f1_df["Instrument"], x=f1_df["F1 Score"], orientation="h",
                marker=dict(color=f1_df["F1 Score"],
                            colorscale=[[0, "#ef4444"], [0.5, "#f97316"], [1, "#22c55e"]],
                            line=dict(width=0)),
                text=f1_df["F1 Score"].round(1).astype(str) + "%",
                textposition="outside", textfont=dict(size=10, color="#e6edf3"),
                hovertemplate="<b>%{y}</b><br>F1: %{x:.1f}%<extra></extra>",
            ))
            fig = apply_theme(fig, height=420, title="Per-Class F1 Score", x_title="F1 Score (%)", x_range=[0, 105])
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)

    with ch2:
        if report:
            prec_rec = []
            for c in class_names:
                if c in report:
                    prec_rec.append({
                        "Instrument": instr_names.get(c, c),
                        "Precision": report[c].get("precision", 0) * 100,
                        "Recall": report[c].get("recall", 0) * 100,
                    })
            pr_df = pd.DataFrame(prec_rec)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Precision", x=pr_df["Instrument"], y=pr_df["Precision"],
                                 marker_color="#818cf8", text=pr_df["Precision"].round(1).astype(str)+"%",
                                 textposition="outside", textfont=dict(size=9, color="#e6edf3")))
            fig.add_trace(go.Bar(name="Recall", x=pr_df["Instrument"], y=pr_df["Recall"],
                                 marker_color="#f97316", text=pr_df["Recall"].round(1).astype(str)+"%",
                                 textposition="outside", textfont=dict(size=9, color="#e6edf3")))
            fig.update_layout(barmode="group")
            fig = apply_theme(fig, height=420, title="Precision vs Recall", y_title="Score (%)")
            fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
            fig.update_yaxes(range=[0, 105])
            st.plotly_chart(fig, use_container_width=True)

    # --- Confusion Matrix from static image ---
    st.markdown('<div class="section-header">🔥 Training Artifacts</div>', unsafe_allow_html=True)
    img1, img2 = st.columns(2)
    cm_path = os.path.join(os.path.dirname(__file__), "models", "confusion_matrix.png")
    th_path = os.path.join(os.path.dirname(__file__), "models", "training_history.png")
    with img1:
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix", use_column_width=True)
        else:
            st.info("Confusion matrix image not found.")
    with img2:
        if os.path.exists(th_path):
            st.image(th_path, caption="Training History", use_column_width=True)
        else:
            st.info("Training history image not found.")


# ============================================
# TAB 2: LIVE PREDICTION
# ============================================
def render_live_prediction(model, bundle, model_loaded):
    class_names = bundle["class_names"]
    instr_names = bundle.get("instrument_names", INSTRUMENT_NAMES)
    config = bundle.get("preprocessing_config", {})

    st.markdown('<div class="section-header">🎤 Upload Audio for Prediction</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload an audio file (WAV, MP3, FLAC)", type=["wav", "mp3", "flac"],
                                key="live_upload", help="Max 25MB. Supported: WAV, MP3, FLAC.")

    if uploaded is not None:
        audio_bytes = uploaded.read()
        uploaded.seek(0)
        st.audio(uploaded, format="audio/wav")
        st.markdown("")

        if not model_loaded:
            st.error("⚠️ Model not loaded. Cannot make predictions.")
            return

        with st.spinner("🔍 Analyzing audio..."):
            try:
                result = run_prediction(model, audio_bytes, config, class_names, instr_names)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

        pred_df = result["predictions"]
        top = pred_df.iloc[0]
        threshold = st.session_state.get("threshold", 30)

        # Top prediction card
        emoji = INSTRUMENT_EMOJIS.get(top["Instrument"], "🎵")
        st.markdown(f"""
            <div class="top-prediction-card">
                <div class="top-pred-label">🏆 Detected Instrument</div>
                <div class="top-pred-name">{emoji} {top["Instrument"]}</div>
                <div class="top-pred-conf">{top["Confidence"]}% Confidence</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Audio info + visualizations
        audio_signal = result["audio_signal"]
        audio_sr = result["audio_sr"]
        duration = round(len(audio_signal) / audio_sr, 2)

        info1, info2, info3 = st.columns(3)
        info1.markdown(metric_card("📁 File", uploaded.name[:20], "Uploaded"), unsafe_allow_html=True)
        info2.markdown(metric_card("⏱ Duration", f"{duration}s", f"{audio_sr} Hz"), unsafe_allow_html=True)
        info3.markdown(metric_card("🔢 Segments", str(result["timeline"].shape[0]),
                                   f'{config.get("duration_seconds",3)}s each'), unsafe_allow_html=True)

        st.markdown('<div class="section-header">📊 Audio Analysis</div>', unsafe_allow_html=True)
        v1, v2 = st.columns(2)

        with v1:
            fig_w, ax_w = plt.subplots(figsize=(6, 2.5), facecolor="#0a0e1a")
            ax_w.set_facecolor("#0d1220")
            librosa.display.waveshow(y=audio_signal, sr=audio_sr, ax=ax_w, color="#818cf8")
            ax_w.set_title("Waveform", color="#e6edf3", fontsize=11, fontweight="bold")
            ax_w.set_xlabel("Time (s)", color="#8b949e", fontsize=8)
            ax_w.set_ylabel("Amplitude", color="#8b949e", fontsize=8)
            ax_w.tick_params(colors="#8b949e", labelsize=7)
            for sp in ax_w.spines.values(): sp.set_color("#1e2438")
            fig_w.tight_layout()
            buf = io.BytesIO()
            fig_w.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig_w)
            buf.seek(0)
            st.image(buf, use_column_width=True)

        with v2:
            mel = librosa.feature.melspectrogram(y=audio_signal, sr=audio_sr, n_mels=128, hop_length=512)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            fig_s, ax_s = plt.subplots(figsize=(6, 2.5), facecolor="#0a0e1a")
            ax_s.set_facecolor("#0d1220")
            img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=audio_sr, hop_length=512, cmap='magma', ax=ax_s)
            ax_s.set_title("Mel Spectrogram", color="#e6edf3", fontsize=11, fontweight="bold")
            ax_s.set_xlabel("Time (s)", color="#8b949e", fontsize=8)
            ax_s.set_ylabel("Hz", color="#8b949e", fontsize=8)
            ax_s.tick_params(colors="#8b949e", labelsize=7)
            for sp in ax_s.spines.values(): sp.set_color("#1e2438")
            cbar = fig_s.colorbar(img, ax=ax_s, format='%+2.0f dB')
            cbar.ax.tick_params(colors="#8b949e", labelsize=7)
            cbar.outline.set_edgecolor("#1e2438")
            fig_s.tight_layout()
            buf2 = io.BytesIO()
            fig_s.savefig(buf2, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig_s)
            buf2.seek(0)
            st.image(buf2, use_column_width=True)

        # Prediction bar chart + instrument cards
        st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
        pc1, pc2 = st.columns([1.2, 0.8])

        with pc1:
            fig = go.Figure(go.Bar(
                x=pred_df["Confidence"], y=pred_df["Instrument"], orientation="h",
                marker=dict(color=["#22c55e" if c >= threshold else "#ef4444" for c in pred_df["Confidence"]]),
                text=pred_df["Confidence"].astype(str) + "%", textposition="outside",
                textfont=dict(size=10, color="#e6edf3"),
                hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>",
            ))
            fig.add_vline(x=threshold, line=dict(color="#fbbf24", width=2, dash="dash"),
                          annotation_text=f"Threshold ({threshold}%)", annotation_font=dict(color="#fbbf24", size=10))
            fig = apply_theme(fig, height=400, title="All Predictions", x_title="Confidence (%)", x_range=[0, 105], show_legend=False)
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)

        with pc2:
            present = pred_df[pred_df["Confidence"] >= threshold]
            absent = pred_df[pred_df["Confidence"] < threshold]
            st.markdown(f"**✅ Present ({len(present)})**")
            for _, p in present.iterrows():
                emoji_i = INSTRUMENT_EMOJIS.get(p["Instrument"], "🎵")
                st.markdown(f"""<div class="instr-present">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-weight:700;color:#e6edf3;font-size:0.9rem;">{emoji_i} {p["Instrument"]}</span>
                        <span style="background:rgba(34,197,94,0.15);color:#22c55e;padding:2px 8px;border-radius:20px;font-size:0.6rem;font-weight:700;">PRESENT</span>
                    </div>
                    <div style="color:#8b949e;font-size:0.75rem;margin-top:0.2rem;">Confidence: <strong style="color:#e6edf3;">{p["Confidence"]}%</strong></div>
                    <div class="prog-bg"><div class="prog-fill-green" style="width:{p['Confidence']}%;"></div></div>
                </div>""", unsafe_allow_html=True)

            if len(absent) > 0:
                st.markdown(f"**❌ Not Present ({len(absent)})**")
                for _, p in absent.iterrows():
                    st.markdown(f"""<div class="instr-absent">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-weight:700;color:#e6edf3;font-size:0.9rem;">{p["Instrument"]}</span>
                            <span style="background:rgba(239,68,68,0.15);color:#ef4444;padding:2px 8px;border-radius:20px;font-size:0.6rem;font-weight:700;">ABSENT</span>
                        </div>
                        <div style="color:#8b949e;font-size:0.75rem;margin-top:0.2rem;">Confidence: <strong>{p["Confidence"]}%</strong></div>
                        <div class="prog-bg"><div class="prog-fill-red" style="width:{p['Confidence']}%;"></div></div>
                    </div>""", unsafe_allow_html=True)

        # Add to session prediction history
        _add_to_history(uploaded.name, pred_df, result["top_instrument"], result["top_confidence"])

        # Timeline
        if result["timeline"].shape[0] > 1:
            st.markdown('<div class="section-header">⏱ Instrument Activity Timeline</div>', unsafe_allow_html=True)
            seg_dur = config.get("duration_seconds", 3)
            x_time = np.arange(result["timeline"].shape[0]) * seg_dur
            fig_tl = go.Figure()
            readable = [instr_names.get(c, c) for c in class_names]
            for i, (name, color) in enumerate(zip(readable, INSTRUMENT_COLORS[:len(readable)])):
                fig_tl.add_trace(go.Scatter(
                    x=x_time, y=result["timeline"][:, i] * 100, mode="lines+markers",
                    name=name, line=dict(width=2.5, color=color), marker=dict(size=4),
                    hovertemplate=f"<b>{name}</b><br>Time: %{{x}}s<br>Confidence: %{{y:.1f}}%<extra></extra>",
                ))
            fig_tl.add_hline(y=threshold, line=dict(color="#fbbf24", width=2, dash="dash"),
                             annotation_text=f"Threshold ({threshold}%)", annotation_font=dict(color="#fbbf24", size=10))
            fig_tl = apply_theme(fig_tl, height=420, title="Instrument Activity Over Time",
                                 x_title="Time (s)", y_title="Confidence (%)")
            fig_tl.update_layout(legend=dict(font=dict(size=9, color="#e6edf3"), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_tl, use_container_width=True)


# ============================================
# TAB 3: BATCH PREDICTION
# ============================================
def _render_batch_file_analysis(fname, audio_signal, audio_sr, pred_df, threshold, is_dark=True):
    """Render waveform, mel spectrogram, and prediction chart for one batch file."""
    duration = round(len(audio_signal) / audio_sr, 2)
    top = pred_df.iloc[0]
    emoji = INSTRUMENT_EMOJIS.get(top["Instrument"], "🎵")

    # Info row
    i1, i2, i3 = st.columns(3)
    i1.markdown(metric_card("Detected", f'{emoji} {top["Instrument"][:14]}', f'{top["Confidence"]}%'), unsafe_allow_html=True)
    i2.markdown(metric_card("Duration", f"{duration}s", f"{audio_sr} Hz"), unsafe_allow_html=True)
    present_count = len(pred_df[pred_df["Confidence"] >= threshold])
    i3.markdown(metric_card("Present", str(present_count), f"≥{threshold}% threshold"), unsafe_allow_html=True)

    st.markdown("")

    # Waveform + Spectrogram
    w_col, s_col = st.columns(2)
    with w_col:
        fig_w, ax_w = plt.subplots(figsize=(5, 2), facecolor="#0a0e1a")
        ax_w.set_facecolor("#0d1220")
        librosa.display.waveshow(y=audio_signal, sr=audio_sr, ax=ax_w, color="#818cf8")
        ax_w.set_title("Waveform", color="#e6edf3", fontsize=9, fontweight="bold")
        ax_w.set_xlabel("Time (s)", color="#8b949e", fontsize=7)
        ax_w.set_ylabel("Amp", color="#8b949e", fontsize=7)
        ax_w.tick_params(colors="#8b949e", labelsize=6)
        for sp in ax_w.spines.values(): sp.set_color("#1e2438")
        fig_w.tight_layout()
        buf = io.BytesIO()
        fig_w.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig_w)
        buf.seek(0)
        st.image(buf, use_column_width=True)

    with s_col:
        mel = librosa.feature.melspectrogram(y=audio_signal, sr=audio_sr, n_mels=128, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        fig_s, ax_s = plt.subplots(figsize=(5, 2), facecolor="#0a0e1a")
        ax_s.set_facecolor("#0d1220")
        img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=audio_sr, hop_length=512, cmap='magma', ax=ax_s)
        ax_s.set_title("Mel Spectrogram", color="#e6edf3", fontsize=9, fontweight="bold")
        ax_s.set_xlabel("Time (s)", color="#8b949e", fontsize=7)
        ax_s.set_ylabel("Hz", color="#8b949e", fontsize=7)
        ax_s.tick_params(colors="#8b949e", labelsize=6)
        for sp in ax_s.spines.values(): sp.set_color("#1e2438")
        cbar = fig_s.colorbar(img, ax=ax_s, format='%+2.0f dB')
        cbar.ax.tick_params(colors="#8b949e", labelsize=6)
        cbar.outline.set_edgecolor("#1e2438")
        fig_s.tight_layout()
        buf2 = io.BytesIO()
        fig_s.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig_s)
        buf2.seek(0)
        st.image(buf2, use_column_width=True)

    # Prediction bar chart
    fig = go.Figure(go.Bar(
        x=pred_df["Confidence"], y=pred_df["Instrument"], orientation="h",
        marker=dict(color=["#22c55e" if c >= threshold else "#ef4444" for c in pred_df["Confidence"]]),
        text=pred_df["Confidence"].astype(str) + "%", textposition="outside",
        textfont=dict(size=9, color="#e6edf3"),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
    ))
    fig = apply_theme(fig, height=300, title="Predictions", x_title="Confidence (%)", x_range=[0, 105], show_legend=False)
    fig.update_yaxes(title="", tickfont=dict(size=9))
    st.plotly_chart(fig, use_container_width=True)


def render_batch_prediction(model, bundle, model_loaded):
    class_names = bundle["class_names"]
    instr_names = bundle.get("instrument_names", INSTRUMENT_NAMES)
    config = bundle.get("preprocessing_config", {})

    st.markdown('<div class="section-header">📁 Batch Upload & Predict</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;font-size:0.85rem;">Upload multiple audio files to get predictions for all of them. Each file gets waveform, spectrogram, and predictions. Download results as CSV or JSON.</p>', unsafe_allow_html=True)

    files = st.file_uploader("Upload audio files", type=["wav", "mp3", "flac"],
                             accept_multiple_files=True, key="batch_upload")

    if not files:
        st.info("Upload one or more audio files to start batch prediction.")
        return

    if not model_loaded:
        st.error("⚠️ Model not loaded. Cannot make predictions.")
        return

    # Create a hashable key from filenames to detect new uploads
    file_key = "|".join(sorted([f.name for f in files]))
    prev_key = st.session_state.get("batch_file_key", "")

    if file_key != prev_key:
        st.session_state["batch_file_key"] = file_key
        st.session_state["batch_results"] = None
        st.session_state["batch_audio_data"] = None

    run_clicked = st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True, key="batch_btn")

    if run_clicked or st.session_state.get("batch_results") is None:
        with st.spinner("🔍 Processing all files..."):
            all_results = []
            audio_data = {}
            batch_status = st.empty()
            for idx, f in enumerate(files):
                batch_status.markdown(f'<p style="color:#818cf8;font-size:0.85rem;">Processing file {idx+1}/{len(files)}: <strong>{f.name}</strong></p>', unsafe_allow_html=True)
                try:
                    audio_bytes = f.read()
                    f.seek(0)
                    res = run_prediction(model, audio_bytes, config, class_names, instr_names)
                    row = {"Filename": f.name, "Top Instrument": res["top_instrument"],
                           "Top Confidence (%)": res["top_confidence"],
                           "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    for _, p in res["predictions"].iterrows():
                        row[p["Instrument"]] = p["Confidence"]
                    all_results.append(row)
                    audio_data[f.name] = {
                        "signal": res["audio_signal"],
                        "sr": res["audio_sr"],
                        "predictions": res["predictions"],
                    }
                    _add_to_history(f.name, res["predictions"], res["top_instrument"], res["top_confidence"])
                except Exception as e:
                    all_results.append({"Filename": f.name, "Top Instrument": f"ERROR: {e}",
                                        "Top Confidence (%)": 0})
            batch_status.empty()
            results_df = pd.DataFrame(all_results)
            st.session_state["batch_results"] = results_df
            st.session_state["batch_audio_data"] = audio_data

    # Display results
    if st.session_state.get("batch_results") is not None:
        results_df = st.session_state["batch_results"]
        audio_data = st.session_state.get("batch_audio_data", {})
        threshold = st.session_state.get("threshold", 30)

        st.markdown('<div class="section-header">📊 Batch Results</div>', unsafe_allow_html=True)

        # Summary cards
        s1, s2, s3, s4 = st.columns(4)
        s1.markdown(metric_card("Files", str(len(results_df)), "Processed"), unsafe_allow_html=True)
        avg_conf = results_df["Top Confidence (%)"].mean()
        s2.markdown(metric_card("Avg Conf", f"{avg_conf:.1f}%", "Top preds"), unsafe_allow_html=True)
        top_instr = results_df["Top Instrument"].mode().iloc[0] if len(results_df) > 0 else "N/A"
        s3.markdown(metric_card("Most Common", top_instr[:12], "Instrument"), unsafe_allow_html=True)
        success_count = len(results_df[~results_df["Top Instrument"].str.startswith("ERROR")])
        s4.markdown(metric_card("Success", f"{success_count}/{len(results_df)}", "Files"), unsafe_allow_html=True)

        st.markdown("")

        # Distribution charts
        viz1, viz2 = st.columns(2)
        with viz1:
            dist = results_df["Top Instrument"].value_counts()
            fig = go.Figure(go.Bar(
                x=dist.index, y=dist.values,
                marker=dict(color=INSTRUMENT_COLORS[:len(dist)], line=dict(width=0)),
                text=dist.values, textposition="outside", textfont=dict(size=11, color="#e6edf3"),
                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
            ))
            fig = apply_theme(fig, height=350, title="Instrument Distribution", y_title="Count", show_legend=False)
            fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
            st.plotly_chart(fig, use_container_width=True)

        with viz2:
            fig2 = go.Figure(go.Histogram(
                x=results_df["Top Confidence (%)"], nbinsx=15,
                marker=dict(color="#818cf8", line=dict(color="#6366f1", width=1)),
                hovertemplate="Confidence: %{x:.1f}%<br>Count: %{y}<extra></extra>",
            ))
            fig2 = apply_theme(fig2, height=350, title="Confidence Spread", x_title="Confidence (%)", y_title="Count", show_legend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Per-file analysis with expanders
        st.markdown('<div class="section-header">🔬 Per-File Audio Analysis</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:0.82rem;">Click on any file below to see its waveform, mel spectrogram, and full prediction breakdown.</p>', unsafe_allow_html=True)

        for idx, row in results_df.iterrows():
            fname = row["Filename"]
            top_inst = row["Top Instrument"]
            top_conf = row["Top Confidence (%)"]

            if fname in audio_data:
                emoji = INSTRUMENT_EMOJIS.get(top_inst, "🎵")
                # Auto-expand the first file's results
                is_first = (idx == 0)
                with st.expander(f"{emoji} {fname}  →  **{top_inst}** ({top_conf}%)", expanded=is_first):
                    ad = audio_data[fname]
                    _render_batch_file_analysis(fname, ad["signal"], ad["sr"], ad["predictions"], threshold)
            else:
                with st.expander(f"❌ {fname} — Error"):
                    st.error(f"Failed to process: {top_inst}")

        # Results table
        st.markdown('<div class="section-header">📋 Detailed Results</div>', unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True, hide_index=True, height=min(400, 40 + len(results_df) * 35))

        # Download buttons
        st.markdown('<div class="section-header">💾 Download Results</div>', unsafe_allow_html=True)
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            csv_data = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", data=csv_data, file_name="instrunet_predictions.csv",
                               mime="text/csv", use_container_width=True)
        with dl2:
            json_data = results_df.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button("📥 Download JSON", data=json_data, file_name="instrunet_predictions.json",
                               mime="application/json", use_container_width=True)
        with dl3:
            st.download_button("📥 Download TXT", data=results_df.to_string(index=False).encode("utf-8"),
                               file_name="instrunet_predictions.txt", mime="text/plain", use_container_width=True)


# ============================================
# TAB 4: PREDICTION HISTORY
# ============================================
def _add_to_history(filename, pred_df, top_instrument, top_confidence):
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []
    st.session_state["prediction_history"].append({
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Top Instrument": top_instrument,
        "Top Confidence (%)": top_confidence,
    })


def render_prediction_history():
    st.markdown('<div class="section-header">📈 Session Prediction History</div>', unsafe_allow_html=True)

    history = st.session_state.get("prediction_history", [])

    if not history:
        st.info("No predictions made yet in this session. Use the Live Prediction or Batch Prediction tabs to start.")
        return

    hist_df = pd.DataFrame(history)

    # Summary
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(metric_card("Total Predictions", str(len(hist_df)), "This session"), unsafe_allow_html=True)
    s2.markdown(metric_card("Avg Confidence", f'{hist_df["Top Confidence (%)"].mean():.1f}%', "All predictions"), unsafe_allow_html=True)
    unique_files = hist_df["Filename"].nunique()
    s3.markdown(metric_card("Unique Files", str(unique_files), "Analyzed"), unsafe_allow_html=True)
    most_common = hist_df["Top Instrument"].mode().iloc[0] if len(hist_df) > 0 else "N/A"
    s4.markdown(metric_card("Most Detected", most_common, "Instrument"), unsafe_allow_html=True)

    st.markdown("")

    # Distribution chart
    ch1, ch2 = st.columns(2)
    with ch1:
        dist = hist_df["Top Instrument"].value_counts()
        fig = go.Figure(go.Pie(
            labels=dist.index, values=dist.values, hole=0.55,
            marker=dict(colors=INSTRUMENT_COLORS[:len(dist)]),
            textinfo="percent", textfont=dict(size=11, color="#e6edf3"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig = apply_theme(fig, height=380, title="Prediction Distribution")
        fig.update_layout(legend=dict(font=dict(size=10, color="#e6edf3"), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        conf_fig = go.Figure(go.Histogram(
            x=hist_df["Top Confidence (%)"], nbinsx=20,
            marker=dict(color="#818cf8", line=dict(color="#6366f1", width=1)),
            hovertemplate="Confidence: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        ))
        conf_fig = apply_theme(conf_fig, height=380, title="Confidence Distribution",
                               x_title="Confidence (%)", y_title="Count", show_legend=False)
        st.plotly_chart(conf_fig, use_container_width=True)

    # Full table
    st.markdown("")
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # Download
    st.markdown('<div class="section-header">💾 Export History</div>', unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    with dl1:
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="prediction_history.csv",
                           mime="text/csv", use_container_width=True)
    with dl2:
        js = hist_df.to_json(orient="records", indent=2).encode("utf-8")
        st.download_button("📥 Download JSON", data=js, file_name="prediction_history.json",
                           mime="application/json", use_container_width=True)

    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state["prediction_history"] = []
        st.rerun()


# ============================================
# MAIN APP
# ============================================
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="bg-blur-1"></div><div class="bg-blur-2"></div>', unsafe_allow_html=True)

    # Load model
    try:
        model, bundle = load_model_bundle()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        model = None
        bundle = {
            "class_names": list(INSTRUMENT_NAMES.keys()),
            "instrument_names": INSTRUMENT_NAMES,
            "preprocessing_config": {}, "evaluation_results": {},
            "architecture_config": {"backbone": "CustomCNN", "num_classes": 11},
            "metadata": {"framework": "PyTorch", "version": "2.0"},
        }

    class_names = bundle["class_names"]
    instr_names = bundle.get("instrument_names", INSTRUMENT_NAMES)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🎵 InstruNet")
        st.markdown('<p style="color:#94a3b8;font-size:0.78rem;">CNN Music Instrument Recognition</p>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("#### ⚙️ Settings")
        threshold = st.slider("Confidence Threshold (%)", 10, 95, 30, 5)
        st.session_state["threshold"] = threshold

        st.markdown("---")
        status_color = "#22c55e" if model_loaded else "#ef4444"
        status_text = "Loaded" if model_loaded else "Not loaded"
        st.markdown(f"""
            <div style="background:rgba(22,27,45,0.6);border:1px solid rgba(99,102,241,0.15);border-radius:12px;padding:1rem;">
                <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#8b949e;margin-bottom:0.5rem;">System Status</div>
                <div style="font-size:0.8rem;color:#e6edf3;line-height:1.8;">
                    Model: <span style="color:{status_color};font-weight:700;">{status_text}</span><br>
                    Arch: CustomCNN<br>
                    Framework: PyTorch<br>
                    Classes: {len(class_names)}
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["password_correct"] = False
            st.rerun()

    # Header
    st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 0.5rem;">
            <div style="font-size:2.5rem;font-weight:900;
                background:linear-gradient(135deg,#818cf8 0%,#6366f1 30%,#f97316 70%,#fb923c 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;letter-spacing:-0.02em;line-height:1.2;">
                🎵 InstruNet Dashboard
            </div>
            <div style="font-size:0.95rem;color:#94a3b8;font-weight:400;margin-top:0.3rem;">
                Production-Level CNN Music Instrument Recognition System
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Overview", "🎤 Live Prediction", "📁 Batch Prediction", "📈 Prediction History"
    ])

    with tab1:
        render_model_overview(bundle, class_names, instr_names)
    with tab2:
        render_live_prediction(model, bundle, model_loaded)
    with tab3:
        render_batch_prediction(model, bundle, model_loaded)
    with tab4:
        render_prediction_history()

    # Footer
    st.markdown("""
        <div class="app-footer">
            🎵 InstruNet Dashboard v2.0 • CNN-Based Music Instrument Recognition • CustomCNN • PyTorch
        </div>
    """, unsafe_allow_html=True)


# ============================================
# LOGIN SYSTEM — Premium Design
# ============================================
def check_password():
    if "users" not in st.session_state:
        st.session_state["users"] = {"admin": "admin123"}
    if "login_mode" not in st.session_state:
        st.session_state["login_mode"] = "login"

    def form_submitted():
        mode = st.session_state["login_mode"]
        if mode == "login":
            u = st.session_state.get("username", "")
            p = st.session_state.get("password", "")
            if u in st.session_state["users"] and st.session_state["users"][u] == p:
                st.session_state["password_correct"] = True
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
        else:
            u = st.session_state.get("reg_username", "")
            p = st.session_state.get("reg_password", "")
            c = st.session_state.get("reg_confirm", "")
            if not u or not p:
                st.session_state["signup_error"] = "Please fill in all fields."
            elif p != c:
                st.session_state["signup_error"] = "Passwords do not match."
            elif u in st.session_state["users"]:
                st.session_state["signup_error"] = "Account already exists."
            else:
                st.session_state["users"][u] = p
                st.session_state["login_mode"] = "login"
                st.session_state["signup_success"] = True
                st.session_state["password_correct"] = None

    def toggle_mode():
        st.session_state["login_mode"] = "signup" if st.session_state["login_mode"] == "login" else "login"
        st.session_state["password_correct"] = None

    # Premium Login CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        .stApp {
            background: #050816 !important;
            color: #ffffff !important;
            font-family: 'Inter', sans-serif !important;
            overflow: hidden;
        }
        /* Animated orbs */
        .login-orb-1, .login-orb-2, .login-orb-3 {
            position: fixed; border-radius: 50%; filter: blur(100px);
            pointer-events: none; z-index: 0;
        }
        .login-orb-1 {
            width: 500px; height: 500px; top: -100px; left: -50px;
            background: radial-gradient(circle, rgba(99,102,241,0.4) 0%, transparent 70%);
            animation: orb1 15s ease-in-out infinite;
        }
        .login-orb-2 {
            width: 400px; height: 400px; bottom: -80px; right: -30px;
            background: radial-gradient(circle, rgba(249,115,22,0.35) 0%, transparent 70%);
            animation: orb2 18s ease-in-out infinite;
        }
        .login-orb-3 {
            width: 300px; height: 300px; top: 40%; left: 50%;
            background: radial-gradient(circle, rgba(139,92,246,0.25) 0%, transparent 70%);
            animation: orb3 12s ease-in-out infinite;
        }
        @keyframes orb1 { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(80px,60px) scale(1.1)} }
        @keyframes orb2 { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(-60px,-40px) scale(1.15)} }
        @keyframes orb3 { 0%,100%{transform:translate(-50%,-50%) scale(1)} 50%{transform:translate(-50%,-50%) scale(0.85)} }

        /* Glass login card */
        .login-glass {
            background: rgba(15, 20, 40, 0.6);
            backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
            border: 1px solid rgba(99,102,241,0.2);
            border-radius: 24px; padding: 2.5rem 2rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
            max-width: 400px; width: 100%; margin: 0 auto;
            position: relative; z-index: 1;
        }
        .login-glass::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #6366f1, #a855f7, #f97316);
            border-radius: 24px 24px 0 0;
        }
        .login-brand {
            font-size: 2.8rem; font-weight: 900; text-align: center;
            background: linear-gradient(135deg, #818cf8 0%, #6366f1 40%, #f97316 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; margin-bottom: 0.3rem; letter-spacing: -0.02em;
        }
        .login-subtitle {
            text-align: center; color: #64748b; font-size: 0.85rem;
            margin-bottom: 2rem; font-weight: 400;
        }
        .login-heading {
            text-align: center; color: #e2e8f0; font-size: 1.4rem;
            font-weight: 700; margin-bottom: 1.5rem;
        }

        /* Form styling */
        [data-testid="stForm"] {
            border: none !important; padding: 0 !important;
            background-color: transparent !important; box-shadow: none !important;
        }
        [data-testid="stTextInput"] input {
            background: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid rgba(99,102,241,0.25) !important;
            color: #ffffff !important; border-radius: 12px !important;
            padding: 14px 18px !important; font-size: 15px !important;
            font-family: 'Inter', sans-serif !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: #818cf8 !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
        }
        [data-testid="stTextInput"] input::placeholder { color: #475569 !important; }

        /* Submit button */
        [data-testid="stForm"] .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, #6366f1 0%, #7c3aed 50%, #a855f7 100%) !important;
            color: white !important; border: none !important;
            border-radius: 12px !important; padding: 14px !important;
            font-weight: 700 !important; font-size: 15px !important;
            margin-top: 1rem !important; letter-spacing: 0.02em;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
        }
        [data-testid="stForm"] .stButton>button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 25px rgba(99,102,241,0.45) !important;
        }

        /* Toggle button */
        .login-toggle .stButton>button {
            width: 100%; background: transparent !important;
            color: #818cf8 !important; border: 1px solid rgba(99,102,241,0.2) !important;
            border-radius: 12px !important; padding: 10px !important;
            font-size: 14px !important; font-weight: 600 !important;
            box-shadow: none !important; transition: all 0.3s ease !important;
        }
        .login-toggle .stButton>button:hover {
            background: rgba(99,102,241,0.08) !important;
            border-color: #818cf8 !important;
        }

        .error-msg {
            background: rgba(239,68,68,0.08); color: #f87171;
            border: 1px solid rgba(239,68,68,0.2);
            padding: 12px 16px; border-radius: 12px; font-size: 13px;
            margin-bottom: 1rem; backdrop-filter: blur(8px);
        }
        .success-msg {
            background: rgba(34,197,94,0.08); color: #4ade80;
            border: 1px solid rgba(34,197,94,0.2);
            padding: 12px 16px; border-radius: 12px; font-size: 13px;
            margin-bottom: 1rem; backdrop-filter: blur(8px);
        }
        .toggle-text {
            color: #64748b; font-size: 13px; text-align: center;
            margin-top: 1.5rem; margin-bottom: 0.3rem;
        }

        /* Feature pills */
        .feature-row {
            display: flex; gap: 8px; justify-content: center;
            flex-wrap: wrap; margin-top: 1.5rem;
        }
        .feature-pill {
            background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.15);
            border-radius: 20px; padding: 5px 14px;
            font-size: 0.7rem; color: #94a3b8; font-weight: 500;
        }

        #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.get("password_correct", False):
        return True

    mode = st.session_state["login_mode"]

    # Animated orbs
    st.markdown('<div class="login-orb-1"></div><div class="login-orb-2"></div><div class="login-orb-3"></div>', unsafe_allow_html=True)

    # Spacer
    st.markdown('<div style="height:8vh;"></div>', unsafe_allow_html=True)

    # Glass card start
    st.markdown('<div class="login-glass">', unsafe_allow_html=True)
    st.markdown('<div class="login-brand">🎵 InstruNet</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">AI-Powered Music Instrument Recognition</div>', unsafe_allow_html=True)

    if mode == "login":
        st.markdown('<div class="login-heading">Welcome back</div>', unsafe_allow_html=True)
        if st.session_state.get("signup_success"):
            st.markdown('<div class="success-msg">✅ Account created successfully! Please log in.</div>', unsafe_allow_html=True)
            st.session_state["signup_success"] = False
        if st.session_state.get("password_correct") is False:
            st.markdown('<div class="error-msg">❌ Incorrect username or password. Please try again.</div>', unsafe_allow_html=True)
        with st.form("login_form"):
            st.text_input("Username", key="username", placeholder="👤  Enter username", label_visibility="collapsed")
            st.text_input("Password", type="password", key="password", placeholder="🔒  Enter password", label_visibility="collapsed")
            st.form_submit_button("Sign In →", on_click=form_submitted)
        st.markdown('<div class="toggle-text">Don\'t have an account?</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-toggle">', unsafe_allow_html=True)
        st.button("Create Account", on_click=toggle_mode, key="toggle_signup")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="login-heading">Create your account</div>', unsafe_allow_html=True)
        if st.session_state.get("signup_error"):
            st.markdown(f'<div class="error-msg">❌ {st.session_state["signup_error"]}</div>', unsafe_allow_html=True)
            st.session_state["signup_error"] = None
        with st.form("signup_form"):
            st.text_input("Username", key="reg_username", placeholder="👤  Choose username", label_visibility="collapsed")
            st.text_input("Password", type="password", key="reg_password", placeholder="🔒  Create password", label_visibility="collapsed")
            st.text_input("Confirm", type="password", key="reg_confirm", placeholder="🔒  Confirm password", label_visibility="collapsed")
            st.form_submit_button("Create Account →", on_click=form_submitted)
        st.markdown('<div class="toggle-text">Already have an account?</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-toggle">', unsafe_allow_html=True)
        st.button("Sign In", on_click=toggle_mode, key="toggle_login")
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature pills
    st.markdown("""
        <div class="feature-row">
            <span class="feature-pill">🎵 11 Instruments</span>
            <span class="feature-pill">🧠 CustomCNN</span>
            <span class="feature-pill">📊 84% Accuracy</span>
            <span class="feature-pill">⚡ Real-time</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close glass card

    # Footer
    st.markdown('<p style="text-align:center;color:#334155;font-size:0.7rem;margin-top:2rem;position:relative;z-index:1;">Built with PyTorch & Streamlit • © 2024 InstruNet</p>', unsafe_allow_html=True)

    return False


if __name__ == "__main__":
    if check_password():
        main()