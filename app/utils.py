# app/utils.py
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import subprocess
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

PX_TO_CM = 0.14

# ---- VIDEO CONVERSION FOR DISPLAY ----
def convert_avi_to_mp4(avi_path, mp4_path):
    """
    Converts an AVI video to MP4 using ffmpeg for HTML5 video playback compatibility.
    """
    cmd = [
        'ffmpeg',
        '-y',  # overwrite without asking
        '-i', str(avi_path),
        '-vcodec', 'libx264',
        '-crf', '23',
        str(mp4_path)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# ---- MAIN SEGMENTATION FUNCTION ----
def segment_video_demo_2(video_path, model, model_path, save_path, device, threshold=0.5, fps=50, return_overlays=False):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    video_np = np.array(frames)
    T, H, W, C = video_np.shape
    video_tensor = torch.from_numpy(video_np.transpose(3, 0, 1, 2)).unsqueeze(0).float().to(device) / 255.0

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(video_tensor)
    output_mask = (torch.sigmoid(output[0, 0]) > threshold).cpu().numpy().astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
    volumes, overlay_frames = [], []

    for t in range(T):
        frame = video_np[t]
        mask = output_mask[t]
        vol = compute_volume_from_mask(mask)
        volumes.append(vol)

        overlay = frame.copy()
        overlay[mask == 1] = [0, 255, 0]  # green
        blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        overlay_frames.append(blended)
        out.write(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    out.release()
    edv, esv = max(volumes), min(volumes)
    ef = ((edv - esv) / edv * 100.0) if edv > 1e-6 else 0.0
    if return_overlays:
        return ef, edv, esv, volumes, overlay_frames
    return ef, edv, esv, volumes


# ---- VOLUME COMPUTATION ----
def compute_volume_from_mask(mask_np):
    mask_255 = (mask_np * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    max_cnt = max(cnts, key=cv2.contourArea)
    dummy = np.zeros_like(mask_255)
    return simpsons_biplane(max_cnt, dummy)


def simpsons_biplane(cnt, frame):
    area_dummy, tri = cv2.minEnclosingTriangle(cnt)
    tri = tri.reshape(-1, 2)
    bp = [cnt[closest_point(tri[i], cnt[:, 0, :])[0], 0, :] for i in range(3)]
    apex, mv1, mv2 = label_points(*bp)
    mid_mv = (mv1 + mv2) / 2.0

    apex_slope = apex - mid_mv
    if np.linalg.norm(apex_slope) < 1e-6:
        return 0.0
    apex_slope /= np.linalg.norm(apex_slope)

    h, w = frame.shape
    diag = int(np.sqrt(h**2 + w**2))
    start = (apex - diag * apex_slope).astype(int)
    end = (mid_mv + diag * apex_slope).astype(int)

    blank = np.zeros_like(frame, dtype=np.uint8)
    mask = cv2.drawContours(blank.copy(), [cnt], 0, 1, 1)
    width = 2
    ann_diam = np.linalg.norm(mv1 - mv2)

    while width <= diag * 2:
        line = cv2.line(blank.copy(), tuple(start), tuple(end), 1, width)
        inter = np.logical_and(mask, line)
        pts = np.column_stack(np.where(inter == 1))
        if pts.size == 0:
            width += 1
            continue
        idx_far, arr = farthest_point(np.flip(apex), pts)
        if len(pts) >= 2 and arr[idx_far] >= 0.9 * ann_diam:
            break
        width += 1
    else:
        return 0.0

    intercept = np.flip(pts[idx_far])
    length_cm = np.linalg.norm(apex - intercept) * PX_TO_CM
    area_cm2 = cv2.contourArea(cnt) * (PX_TO_CM ** 2)
    return (8.0 * area_cm2 ** 2) / (3.0 * np.pi * length_cm) if length_cm >= 1e-6 else 0.0


def closest_point(point, array):
    diff = array - point
    dist_sq = np.einsum('ij,ij->i', diff, diff)
    return np.argmin(dist_sq), dist_sq


def farthest_point(point, array):
    diff = array - point
    dist_sq = np.einsum('ij,ij->i', diff, diff)
    return np.argmax(dist_sq), dist_sq


def label_points(p1, p2, p3):
    d12, d23, d13 = np.linalg.norm(p1 - p2), np.linalg.norm(p2 - p3), np.linalg.norm(p1 - p3)
    if d12 < d23 and d12 < d13:
        return p3, p1, p2
    elif d23 < d12 and d23 < d13:
        return p1, p3, p2
    return p2, p1, p3


# ---- PDF REPORT GENERATION USING REPORTLAB ----
def generate_pdf_report(pdf_path, filename, volumes, ef, edv, esv, overlay_frames, fps):
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 50, f"Report for {filename}")

    # Add overlay images (temp save then embed)
    y_position = height - 100
    for i, frame in enumerate(overlay_frames[:5]):
        img_path = f"temp_frame_{i}.png"
        cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        c.drawImage(ImageReader(img_path), 50 + (i % 3) * 180, y_position - 120, width=160, height=120)
        os.remove(img_path)

    # Add volume plot
    plot_path = "temp_plot.png"
    plt.figure()
    plt.plot(np.arange(len(volumes)) / fps, volumes, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Volume (mL)")
    plt.title("LV Volume Over Time")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    c.drawImage(ImageReader(plot_path), 100, y_position - 300, width=400, height=180)
    os.remove(plot_path)

    # Add metrics
    c.setFont("Helvetica", 12)
    c.drawString(50, y_position - 340, f"EF: {ef:.2f}%")
    c.drawString(50, y_position - 360, f"EDV: {edv:.2f} mL")
    c.drawString(50, y_position - 380, f"ESV: {esv:.2f} mL")
    c.save()
