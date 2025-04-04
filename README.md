# EchoFrame: Lightweight and Accurate LVEF Estimation with Heartbeat Analysis
**Author:** Harshal Chalke  
**Course:** Capstone Project  
**Dataset:** EchoNet-Dynamic by Stanford AIMI

---

## Overview
EchoFrame is a lightweight, interpretable pipeline for real-time estimation of Left Ventricular Ejection Fraction (LVEF) using echocardiogram videos. It leverages a MobileNetV3-based U-Net architecture to segment the left ventricle and applies Simpson’s method to calculate LVEF.

---

## Project Framework
A schematic showcasing the MobileNetV3-UNet architecture and the LVEF estimation process.

---

## Directory Structure

echoframe_capstone/
├── assets/      # Project figures and visualizations
├── data/        # Sample data or placeholder path for dataset
├── src/         # Core source code
│   ├── dataloader.py
│   ├── train.py
│   ├── model1.py        # Baseline model
│   ├── model3.py        # Alternative encoder
│   ├── model6.py        # Final model (MobileNetV3-Unet)
│   └── utils.py
├── results/     # Evaluation outputs and visualizations
├── README.md
└── requirements.txt


---

## Installation Instructions

1. **Clone the Repository**
    ```bash
    git clone https://github.com/harshalchalke31/echoframe_capstone.git
    cd echoframe_capstone
    ```

2. **Set up the Environment**
    ```bash
    pip install -r requirements.txt
    ```
    Or use conda:
    ```bash
    conda create -n echoframe python=3.8
    conda activate echoframe
    pip install -r requirements.txt
    ```

---

## Data Preparation
1. Download the **EchoNet-Dynamic** dataset.
2. Extract into a folder named `data/` within the root directory.
3. Ensure structure resembles:
    ```
    data/
    ├── EchoNet_Dynamic/
    │   ├── Videos/
    │   └── FileList.csv
    ```

---

## Run Instructions

### Training
```bash
python src/train.py --model model6 --epochs 30 --batch_size 8 --lr 0.001


---

## Inference / Evaluation

To evaluate the trained model on the test set, run:

```bash
python src/evaluate.py --model_path checkpoints/best_model.pth --mode test


## Results

| Model              | Dice Score | MAE (EF) | RMSE (EF) |
|--------------------|------------|----------|-----------|
| Baseline UNet      | 0.785      | 6.4%     | 9.2%      |
| MobileNetV3-UNet   | 0.842      | 4.8%     | 6.7%      |

## Acknowledgments

- **Dataset:** [EchoNet-Dynamic](https://echonet.github.io/dynamic/) by Stanford Center for Artificial Intelligence in Medicine & Imaging (AIMI).
- **Model Architectures:** Base U-Net and MobileNetV3 encoders adapted from publicly available open-source repositories cited within the source code.
- **Benchmarking:** Special thanks to the [EchoNet GitHub repository](https://github.com/echonet/dynamic) for providing evaluation protocols and baseline benchmarks.

## License

This project is licensed under the **MIT License**.  
For more details, refer to the [LICENSE](LICENSE) file in the repository.

