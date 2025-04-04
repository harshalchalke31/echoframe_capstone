EchoFrame: Lightweight and Accurate LVEF Estimation with Heartbeat Analysis

Author: Harshal Chalke
Course: Capstone Project
Dataset: EchoNet-Dynamic by Stanford AIMI

⸻

Overview

EchoFrame is a lightweight, interpretable pipeline for real-time estimation of Left Ventricular Ejection Fraction (LVEF) using echocardiogram videos. It leverages a MobileNetV3-based U-Net architecture to segment the left ventricle and applies Simpson’s method to calculate LVEF.

⸻

Project Framework

A schematic showcasing the MobileNetV3-UNet architecture and the LVEF estimation process.

⸻

Directory Structure

echoframe_capstone/
├── assets/                     # Project figures and visualizations
├── data/                      # Sample data or placeholder path for dataset
├── src/                       # Core source code
│   ├── dataloader.py
│   ├── train.py
│   ├── model1.py              # Baseline model
│   ├── model3.py              # Alternative encoder
│   ├── model6.py              # Final model (MobileNetV3-Unet)
│   └── utils.py
├── results/                   # Evaluation outputs and visualizations
├── README.md
└── requirements.txt



⸻

Installation Instructions

1. Clone the Repository

git clone https://github.com/harshalchalke31/echoframe_capstone.git
cd echoframe_capstone

2. Set up the Environment

pip install -r requirements.txt

Or use conda:

conda create -n echoframe python=3.8
conda activate echoframe
pip install -r requirements.txt



⸻

Data Preparation
	1.	Download the EchoNet-Dynamic dataset.
	2.	Extract into a folder named data/ within the root directory.
	3.	Ensure structure resembles:

data/
├── EchoNet_Dynamic/
│   ├── Videos/
│   └── FileList.csv



⸻

Run Instructions

Training

python src/train.py --model model6 --epochs 30 --batch_size 8 --lr 0.001

Inference/Evaluation

python src/evaluate.py --model_path checkpoints/best_model.pth --mode test

Note: Inference script is named evaluate.py (placeholder — update as necessary).

⸻

Results

Model	Dice Score	MAE (EF)	RMSE (EF)
Baseline UNet	0.785	6.4%	9.2%
MobileNetV3-UNet	0.842	4.8%	6.7%

Visual results of segmentation and ejection fraction estimation:

LV segmentation over cardiac cycle

⸻

Acknowledgments
	•	Dataset: EchoNet-Dynamic by Stanford AIMI.
	•	Base U-Net and MobileNetV3 encoders adapted from open-source repositories cited in the code.
	•	Special thanks to echonet repo for benchmarks and protocols.

⸻

License

MIT License. See LICENSE file for details.
