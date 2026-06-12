# app.py
import os
import logging

# Mute system, library logging, and oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# 1. Page Config
st.set_page_config(
    page_title="Sentilytics - Sequence Lab",
    page_icon="🔮",
    layout="wide"
)

# Initialize Session State History Logger
if 'history' not in st.session_state:
    st.session_state.history = []

# 2. Custom CSS (Light Theme: Cool Slates, Soft Teals, and Pure White)
st.markdown("""
    <style>
        /* Light Page Background */
        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
        }
        
        /* Styled Header Card */
        .light-header {
            background: linear-gradient(135deg, #f0fdfa 0%, #ecf5ff 100%);
            border: 1px solid #cbd5e1;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        
        .gradient-title {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }
        
        .title-subtitle {
            font-size: 1.05rem;
            color: #475569;
            margin-top: 5px;
        }

        /* Bordered Container Overrides for Light Cards */
        div[data-testid="stVerticalBlockBorderDiv"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            padding: 24px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
        }

        /* Light Theme Prediction Card with Soft Teal Accent */
        .teal-accent-card {
            background-color: #f0fdfa;
            border-radius: 14px;
            padding: 25px 20px;
            box-shadow: 0 4px 6px -1px rgba(13, 148, 136, 0.05);
            border: 1px solid #99f6e4;
            border-left: 6px solid #0d9488;
            text-align: center;
            margin-bottom: 25px;
        }
        .prediction-label {
            font-size: 0.85rem;
            color: #0f766e;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.08em;
        }
        .prediction-class {
            font-size: 1.8rem;
            font-weight: 850;
            color: #0f172a;
            margin: 12px 0;
        }
        .prediction-confidence {
            font-size: 1.1rem;
            color: #0d9488;
            font-weight: 700;
        }

        /* Left Side Status Indicators */
        .status-badge {
            background-color: #f1f5f9;
            color: #334155;
            padding: 10px 14px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 10px;
            display: inline-block;
            width: 100%;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #0d9488;
        }
        
        /* Table Styling */
        .history-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.9rem;
        }
        .history-table th {
            background-color: #f1f5f9;
            color: #0f766e;
            text-align: left;
            padding: 8px 12px;
            border-bottom: 2px solid #0d9488;
        }
        .history-table td {
            padding: 8px 12px;
            border-bottom: 1px solid #e2e8f0;
            color: #334155;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Class Mapping
CLASS_MAPPING = {
    'up_trend': '📈 Up-Trend Pattern',
    'down_trend': '📉 Down-Trend Pattern',
    'cyclic': '🔄 Cyclic / Range Pattern'
}
CLASS_NAMES = list(CLASS_MAPPING.keys())

PRESETS = {
    "Real-time Custom Waveform": None,
    "Standard Upward Drift": [0.4, 0.05],
    "Standard Downward Drift": [-0.4, 0.05],
    "Standard Cyclic Range": [0.0, 0.1]
}

# 4. Model Loader
@st.cache_resource
def load_transformer_model():
    model_path = 'transformer_model.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_transformer_model()

# 5. Sidebar - Diagnostics Control Panel
with st.sidebar:
    st.markdown("<h2 style='color: #0d9488; font-weight:800; margin-top:0;'>Diagnostic Center</h2>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<p style='color: #0f766e; font-weight: 600;'>Identifiable Wave signatures:</p>", unsafe_allow_html=True)
    for category in CLASS_MAPPING.values():
        st.markdown(f'<div class="status-badge">{category}</div>', unsafe_allow_html=True)
    st.divider()
    st.caption("Processor: Multi-Head Self-Attention Pipeline")

# 6. Light Header Card
st.markdown("""
    <div class="light-header">
        <h1 class="gradient-title">Vanilla Transformer Wave Classifier</h1>
        <div class="title-subtitle">Deep temporal sequence pattern recognition powered by multi-head attention blocks</div>
    </div>
""", unsafe_allow_html=True)

if model is None:
    st.error("🚨 **Transformer model artifact missing.** Could not locate `transformer_model.keras` in the root workspace directory. Please execute `model_training.py` first.")
else:
    # 7. Main Interface Split (Using columns)
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown("<h3 style='color: #0f766e; font-weight: 700; margin-top:0;'>📥 Input Sequence Generator</h3>", unsafe_allow_html=True)
            
            preset_choice = st.selectbox(
                "Choose a preloaded template baseline:",
                options=list(PRESETS.keys())
            )
            
            # Load default settings based on preset selection
            defaults = PRESETS[preset_choice] if PRESETS[preset_choice] is not None else [0.0, 0.15]
            
            st.write("###")
            st.markdown("<h5 style='color: #475569; font-weight: 600;'>Signal Tweaker</h5>", unsafe_allow_html=True)
            slope_angle = st.slider("Signal Drift Slope", min_value=-1.0, max_value=1.0, value=defaults[0], step=0.05)
            sensor_noise = st.slider("Signal Distortion Noise", min_value=0.01, max_value=0.5, value=defaults[1], step=0.01)
            
            # Build 50-step wave sequence
            t = np.linspace(0, 4, 50)
            np.random.seed(42)
            noise = np.random.normal(0, sensor_noise, 50)
            
            if preset_choice == "Standard Cyclic Range" or (slope_angle == 0.0 and preset_choice == "Real-time Custom Waveform"):
                signal = 0.8 * np.sin(3.14 * t) + noise
            else:
                signal = slope_angle * t + noise

    with col_right:
        with st.container(border=True):
            st.markdown("<h3 style='color: #0f172a; font-weight: 700; margin-top:0;'>📊 Transformer Prediction Head</h3>", unsafe_allow_html=True)
            
            # Format input array dimensions to (1, 50, 1)
            sequence_input = np.expand_dims(np.expand_dims(signal, axis=-1), axis=0)
            
            # Run prediction
            preds = model.predict(sequence_input, verbose=0)[0]
            best_match_idx = np.argmax(preds)
            label_name = CLASS_NAMES[best_match_idx]
            display_label = CLASS_MAPPING[label_name]
            confidence_score = preds[best_match_idx] * 100
            
            # Save query run state to session logs
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S") if 'datetime' in globals() else "Just now",
                "slope": f"{slope_angle:+.2f}",
                "noise": f"{sensor_noise:.2f}",
                "result": display_label
            })
            
            # Highlighted result card
            st.markdown(f"""
                <div class="teal-accent-card">
                    <div class="prediction-label">Evaluated Waveform Signature</div>
                    <div class="prediction-class">{display_label.upper()}</div>
                    <div class="prediction-confidence">Attention Certainty Index: {confidence_score:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Probability distribution meters
            st.markdown("<h4 style='color: #475569; font-weight: 700;'>Softmax Probability Vector</h4>", unsafe_allow_html=True)
            for i, score in enumerate(preds):
                class_tag = CLASS_MAPPING[CLASS_NAMES[i]]
                col_txt, col_pb = st.columns([2, 5])
                with col_txt:
                    if i == best_match_idx:
                        st.markdown(f"<span style='color: #0d9488; font-weight: 800;'>{class_tag}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color: #475569; font-weight: 500;'>{class_tag}</span>", unsafe_allow_html=True)
                with col_pb:
                    if i == best_match_idx:
                        st.progress(float(score), text=f"Match — {score*100:.1f}%")
                    else:
                        st.progress(float(score), text=f"{score*100:.1f}%")

    # 8. TABS WORKSPACE FOR TRACE VISUALIZATION AND ARCHIVED RUNS
    st.write("###")
    tab_plot, tab_logs = st.tabs(["⚡ Waveform Trace Plot", "📜 Interactive Session Logs"])
    
    with tab_plot:
        with st.container(border=True):
            # Plotly Trace Visualization styled for Light Theme
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t, y=signal,
                mode='lines+markers',
                name='Active Sequence',
                line=dict(color='#0d9488', width=3),
                marker=dict(size=6, color='#0f766e')
            ))
            fig.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                xaxis=dict(gridcolor='#e2e8f0', title="Sequence Timestep", color="#0f172a"),
                yaxis=dict(gridcolor='#e2e8f0', title="Amplitude Value", color="#0f172a"),
                margin=dict(l=20, r=20, t=10, b=20),
                height=350,
                legend=dict(font=dict(color="#0f172a"))
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

    with tab_logs:
        with st.container(border=True):
            st.write("Review recent runs inside this active browser session:")
            if len(st.session_state.history) == 0:
                st.info("No runs logged yet.")
            else:
                # Render a styled HTML Table to display log matrices cleanly
                table_rows = ""
                for idx, item in enumerate(reversed(st.session_state.history[-5:])):
                    table_rows += f"""
                    <tr>
                        <td>Run #{len(st.session_state.history) - idx}</td>
                        <td>{item['slope']}</td>
                        <td>{item['noise']}</td>
                        <td><b>{item['result']}</b></td>
                    </tr>
                    """
                st.markdown(f"""
                    <table class="history-table">
                        <thead>
                            <tr>
                                <th>Test ID</th>
                                <th>Drift Slope</th>
                                <th>Distortion Noise</th>
                                <th>Model Classification Result</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                """, unsafe_allow_html=True)