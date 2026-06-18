import streamlit as st
from ultralytics import YOLO, SAM
from PIL import Image

st.set_page_config(page_title="Sentinel-AI OS", layout="wide", initial_sidebar_state="expanded")

# Inject Custom Tactical CSS
st.markdown("""
<style>
    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tactical Fonts & Headers */
    h1, h2, h3 {
        font-family: 'Courier New', Courier, monospace;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #FF9F1C !important;
    }
    
    /* HUD Metrics Styling */
    div[data-testid="metric-container"] {
        background-color: #1A1D24;
        border: 1px solid #FF9F1C;
        padding: 10px 20px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(255, 159, 28, 0.2);
    }
    
    /* Image Glow Effects */
    img {
        border: 2px solid #2C3E50;
        border-radius: 5px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        border-right: 1px solid #FF9F1C;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ SENTINEL-AI ")

# 1. Lock both massive models into the GPU VRAM
@st.cache_resource
def load_models():
    # Load your custom weights!
    yolo = YOLO('best.pt')
    sam = SAM('mobile_sam.pt') 
    return yolo, sam

yolo_model, sam_model = load_models()

# Main Dashboard Container
with st.container():
    st.info("⚠️ **CALIBRATION NOTICE:** This neural network is strictly calibrated for aerial reconnaissance. It is trained exclusively to detect 10 specific targets from a top-down drone perspective: **Pedestrian, People, Bicycle, Car, Van, Truck, Tricycle, Awning-Tricycle, Bus, and Motor**. It will ignore unrelated objects.")
    
    st.sidebar.header("Radar Settings")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    iou_thresh = st.sidebar.slider("NMS Overlap (IoU)", 0.0, 1.0, 0.50, help="1.00 = Boxes can overlap heavily. 0.00 = Boxes are deleted if they even slightly touch.")

    uploaded_files = st.file_uploader("Upload Drone Images...", type=["jpg", "png"], accept_multiple_files=True)

    import pandas as pd

    if uploaded_files:
        for file in uploaded_files:
            image = Image.open(file)
            st.markdown("---")
            st.subheader(f"🎯 Reconnaissance Target: `{file.name}`")
            
            # Run Neural Scan with UI Feedback
            with st.spinner("📡 SATELLITE UPLINK ACTIVE // EXECUTING NEURAL SCAN..."):
                # Phase 1: YOLO Detection
                yolo_results = yolo_model.predict(image, conf=conf_thresh, iou=iou_thresh)
                boxes = yolo_results[0].boxes
                
                if len(boxes) > 0:
                    # Extract Deep Analytics
                    class_names = [yolo_model.names[int(c)] for c in boxes.cls]
                    raw_confs = boxes.conf.tolist()
                    confs = [f"{round(c * 100, 1)}%" for c in raw_confs]
                    coords = [f"X:{int(b[0])}, Y:{int(b[1])}" for b in boxes.xyxy.tolist()]
                    
                    df = pd.DataFrame({
                        "Target Class": class_names,
                        "Confidence": confs,
                        "Location": coords
                    })
                    
                    # ---------------- HUD TOP BAR ---------------- #
                    st.markdown("### 📊 TACTICAL TELEMETRY")
                    hud1, hud2, hud3 = st.columns(3)
                    with hud1:
                        st.metric(label="TARGETS LOCKED", value=len(boxes))
                    with hud2:
                        highest_conf = round(max(raw_confs) * 100, 1)
                        st.metric(label="MAX CONFIDENCE", value=f"{highest_conf}%")
                    with hud3:
                        st.metric(label="SYSTEM STATUS", value="ONLINE", delta="SECURE", delta_color="normal")
                    st.markdown("<br>", unsafe_allow_html=True)
                    # --------------------------------------------- #
                
                import plotly.express as px
                
                def render_zoomable_image(img_data, title_text):
                    fig = px.imshow(img_data)
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=40, b=0),
                        title=dict(text=title_text, font=dict(color="#FF9F1C", size=16)),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        dragmode="pan",
                        coloraxis_showscale=False
                    )
                    fig.update_xaxes(visible=False)
                    fig.update_yaxes(visible=False)
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

                # Visual Layout
                col1, col2 = st.columns(2)
                with col1:
                    render_zoomable_image(image, "1. RAW SATELLITE FEED")
                with col2:
                    annotated_bgr = yolo_results[0].plot()
                    annotated_rgb = annotated_bgr[..., ::-1]
                    render_zoomable_image(annotated_rgb, "2. YOLO TACTICAL RADAR")
                
                from streamlit_image_coordinates import streamlit_image_coordinates
                
                st.markdown("---")
                st.markdown("### 🧬 ADVANCED NEURAL EXTRACTION")
                extraction_mode = st.radio(
                    "SELECT EXTRACTION PROTOCOL:",
                    ["🌐 OMNI-SEGMENTATION PROTOCOL (Extract All Targets)", 
                     "🎯 SURGICAL TARGETING MODE (Manual Pixel Lock)"],
                    index=0
                )

                if "OMNI-SEGMENTATION" in extraction_mode:
                    with st.spinner("🌐 RUNNING OMNI-SEGMENTATION PROTOCOL..."):
                        sam_results = sam_model.predict(image, bboxes=boxes.xyxy)
                        sam_bgr = sam_results[0].plot()
                        sam_rgb = sam_bgr[..., ::-1]
                        render_zoomable_image(sam_rgb, "3. OMNI-SEGMENTATION RADAR")
                        
                elif "SURGICAL TARGETING" in extraction_mode:
                    st.info("🎯 AWAITING TARGET LOCK: Click anywhere on the raw drone image below to surgically extract the object.")
                    value = streamlit_image_coordinates(image, key="surgical_target")
                    if value is not None:
                        x, y = value["x"], value["y"]
                        with st.spinner(f"🎯 TARGET LOCKED AT [X:{x}, Y:{y}]. EXTRACTING..."):
                            sam_results = sam_model.predict(image, points=[[x, y]], labels=[1])
                            sam_bgr = sam_results[0].plot()
                            sam_rgb = sam_bgr[..., ::-1]
                            render_zoomable_image(sam_rgb, "3. ISOLATED TARGET PAYLOAD")
                
                # Deep Research Expander
                with st.expander("🔬 Deep Research: Intelligence Data Matrix", expanded=False):
                    st.dataframe(df, use_container_width=True)
                    st.write("Target Distribution:")
                    st.bar_chart(df["Target Class"].value_counts())
    else:
        st.warning("❌ No targets detected. Try lowering Confidence.")

# End of Sentinel-AI Dashboard
