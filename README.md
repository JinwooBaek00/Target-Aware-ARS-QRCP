# SORG - SoftNorm gated Orthogonal Residual Greedy

![CI](https://github.com/JinwooBaek00/Target-Aware-ARS-QRCP/actions/workflows/ci.yml/badge.svg)

**SORG** (SoftNorm gated Orthogonal Residual Greedy) is a modular, zero-retraining data selection algorithm designed for machine learning researchers and data scientists who need to extract highly informative, small subsets from massive datasets.

## 📖 About the Project

Modern machine learning datasets are massive, noisy, and incredibly expensive to process. When researchers try to extract a smaller "starter pack" of data to speed up their workflow, traditional selection algorithms often fail. They get distracted by massive outliers, or they pick data that doesn't align with the final goal. This forces researchers to either waste expensive GPU hours training on redundant data or spend weeks retraining models just to figure out which samples are actually useful.

**The Solution:** SORG proves that you do not need to train a model to understand your data. By combining a "projector-greedy" mathematical approach with a unique stabilizing module (SoftNorm), SORG instantly and intelligently compresses a dataset using only its frozen features—completely ignoring noisy outliers and saving massive amounts of computational overhead.

## Key Features

* **Zero-Retraining Compression:** Skip the expensive warm-up phases. SORG operates entirely on frozen, pre-extracted features, allowing you to instantly reduce your dataset size by up to 90% before you ever spin up a heavy deep learning training loop.
* **Immunity to "Loud" Outliers (SoftNorm):** Messy datasets derail standard algorithms, which mistake extreme, noisy values for important data. SORG mathematically down-weights these outliers, guaranteeing your subset is filled with genuinely diverse, high-quality examples rather than just mathematical anomalies.
* **Target-Driven "Cherry Picking":** You aren't stuck with a generic summary of your data. If you are dealing with a mismatched dataset (e.g., trying to find vehicles inside a wildlife database), SORG accepts a target hint to actively ignore irrelevant data and pull exactly the samples that align with your specific end goal.
* **Multi-Dimensional Versatility:** SORG isn't just a tool for deleting rows of data (Coreset Selection); it is mathematically flexible enough to delete useless columns (Feature/Pixel Selection). Whether you are trying to find the best 50 images in a folder or the 50 most important pixels in a single image, the same core algorithm handles both efficiently.
---

## How to Access or Try it

### Run Experiments Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/JinwooBaek00/Target-Aware-ARS-QRCP.git
   cd Target-Aware-ARS-QRCP
2. Install Dependencies:
   ```bash
    pip install -r requirements.txt
3. Run Experiment script:
   ```bash
   python SORG-Experiments/<experiment_folder>/<script_name>.py

⚙️ Specs Recommendation:
- GPU (NVIDIA with CUDA support)
- 16–32 GB RAM minimum
- Multi-core CPU
- Linux or macOS environment preferred

## 📄 Paper

📥 **Read the full paper:**  
[Download PDF](./SORG_for_CS462.pdf)

  
## Team Members
- Jinwoo Baek (baekji@oregonstate.edu)
- Joy Lim (limjoy@oregonstate.edu)
- Kevin Nguyen (nguykev2@oregonstate.edu)
- Kevin Tran (trank8@oregonstate.edu)

Feel free to contact any of us at the emails above if you have any questions!
