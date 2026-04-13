# SORG Experiments Documentation

## 1. End-to-End Coreset Selection Experiment

### Overview

The end-to-end experiment evaluates the effectiveness of the SORG
Algorithm in reducing training data while maintaining model performance.
The goal is to determine whether a small, intelligently selected subset
of data can approximate the performance of training on the full dataset.

### Methodology

**1. Feature Extraction (Selection Phase)**\
Images from the dataset are passed through a pretrained model such as
ResNet-18 to obtain feature embeddings. These embeddings provide a
compact numerical representation of each image.

**2. Coreset Selection**\
SORG operates on the extracted features to select a subset of training
samples. The selection is performed in a group-wise manner, where each
class contributes proportionally to the final subset. Baseline methods
include: 
- Random selection
- Herding selection

**3. Model Training (End-to-End Phase)**
A new model is initialized and trained from scratch using: 
- the full
dataset
- random subset
- herding subset
- SORG subset

### Evaluation Metrics

-   **Test Accuracy**
-   **Training Time**

### Expected Outcome

SORG is expected to preserve accuracy within a small margin while
significantly reducing training time.

------------------------------------------------------------------------

## 2. Pixel Selection Experiment

### Overview

The pixel selection experiment evaluates SORG at a finer granularity by
selecting informative pixels or features within images.

### Methodology

**1. Feature Representation**\
MNIST images are flattened from $28 \times 28$ grids into 1D vectors of 
784 pixels, normalized to $[0, 1]$.

**2. Pixel Selection**\
Algorithms are tasked with selecting a subset of $k$ pixels 
(e.g., 10, 20, 50, 100) using only the training data. Four methods:
- Random: A naive lower-bound baseline making uniform random selections
- Variance: An unsupervised baseline that selects the $k$ pixels with the
  highest variance.
- L1-Logistic (Oracle): A supervised upper-bound baseline that uses labels
  and a Lasso penalty (SAGA solver) to rank the most discriminative pixels.
- SORG: Configured in an unsupervised "NoGate" mode ($p=0$, $r=\text{None}$),
- utilizing pure orthogonal matching pursuit across the feature space

**3. Reconstruction / Evaluation**\
Selected pixels are used for reconstruction or classification tasks.

### Evaluation Metrics

-   **Test Accuracy**
-   **Training Time**

### Expected Outcome

SORG is expected to significantly outperform the naive Variance method 
in classification accuracy and closely track the supervised L1-Logistic 
Oracle, especially at lower budgets

------------------------------------------------------------------------
