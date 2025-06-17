# app/main.py
import streamlit as st
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Your custom modules
from src_3d.model1 import MobileNetV3UNet3D
from utils import segment_video_demo_2, generate_pdf_report, convert_avi_to_mp4

# ------------------------------------------------------------------------------
# 1) Determine the folder of this script (app folder)
# ------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------
# 2) Configure your data and model paths
# ------------------------------------------------------------------------------
DATA_PATH = Path(r"C:\Projects\python\echoframe\data\EchoNet-Dynamic\EchoNet-Dynamic")
MODEL_PATH = (APP_DIR / ".." / "models" / "pretrained_mobilenet_3d.pt").resolve()
TEST_CSV = DATA_PATH / "FileList.csv"
VIDEO_DIR = DATA_PATH / "Videos"

# Folders for saving segmented videos, converted videos, and reports
STATIC_DIR = (APP_DIR / "static").resolve()
STATIC_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = (APP_DIR / "reports").resolve()
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# 3) Streamlit UI: Title
# ------------------------------------------------------------------------------
st.title("EchoFrame: Left Ventricle Segmentation & EF Estimation")

# ------------------------------------------------------------------------------
# 4) Load test CSV and populate the select box
# ------------------------------------------------------------------------------
if not TEST_CSV.is_file():
    st.error(f"TEST CSV not found at: {TEST_CSV}")
    st.stop()

file_df = pd.read_csv(TEST_CSV)
file_df = file_df[file_df["Split"] == "TEST"]
file_df["FileName"] = file_df["FileName"].apply(lambda x: x if x.endswith(".avi") else x + ".avi")

selected_file = st.selectbox("Select Test Video", file_df["FileName"].tolist())

# ------------------------------------------------------------------------------
# 5) When a file is selected, display the original video
# ------------------------------------------------------------------------------
if selected_file:
    video_path = VIDEO_DIR / selected_file
    mp4_display_path = STATIC_DIR / selected_file.replace(".avi", "_display.mp4")

    # Convert to MP4 if not already converted
    if not mp4_display_path.exists():
        convert_avi_to_mp4(video_path, mp4_display_path)

    # Display the MP4 video
    if mp4_display_path.is_file():
        with open(mp4_display_path, 'rb') as f:
            st.video(f.read())
    else:
        st.error(f"Converted MP4 video not found: {mp4_display_path}")

    # --------------------------------------------------------------------------
    # 6) Segment Video
    # --------------------------------------------------------------------------
    if st.button("Segment Video"):
        model = MobileNetV3UNet3D()
        out_video_name = selected_file.replace(".avi", "_segmented.avi")
        out_video_path = STATIC_DIR / out_video_name

        with st.spinner("Segmenting..."):
            try:
                ef, edv, esv, volumes, overlay_frames = segment_video_demo_2(
                    video_path=str(video_path),
                    model=model,
                    model_path=str(MODEL_PATH),
                    save_path=str(out_video_path),
                    device="cuda",
                    threshold=0.5,
                    fps=50,
                    return_overlays=True
                )
                st.success("Segmentation Complete!")
            except Exception as e:
                st.error(f"Error during segmentation: {e}")
                st.stop()

        # Convert segmented video for display
        mp4_seg_path = STATIC_DIR / out_video_name.replace(".avi", "_display.mp4")
        convert_avi_to_mp4(out_video_path, mp4_seg_path)

        if mp4_seg_path.is_file():
            with open(mp4_seg_path, 'rb') as f:
                st.video(f.read())
        else:
            st.error(f"Segmented video not found: {mp4_seg_path}")

        st.markdown(
            f"**EDV**: {edv:.2f} mL &nbsp; | &nbsp;"
            f"**ESV**: {esv:.2f} mL &nbsp; | &nbsp;"
            f"**EF**: {ef:.2f}%"
        )

        fig, ax = plt.subplots()
        ax.plot(volumes, marker='o')
        ax.set_title("LV Volume per Frame")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Volume (mL)")
        st.pyplot(fig)

        if st.button("Generate PDF Report"):
            pdf_name = selected_file.replace(".avi", "_report.pdf")
            pdf_path = REPORTS_DIR / pdf_name

            try:
                generate_pdf_report(
                    pdf_path=str(pdf_path),
                    filename=selected_file,
                    volumes=volumes,
                    ef=ef,
                    edv=edv,
                    esv=esv,
                    overlay_frames=overlay_frames,
                    fps=50
                )
                st.success(f"Report saved to: {pdf_path}")
            except Exception as e:
                st.error(f"Error generating PDF report: {e}")
