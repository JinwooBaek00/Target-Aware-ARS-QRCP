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

## 3. Outlier Robustness Experiment

### Overview

The outlier robustness experiment evaluates the stability of the SORG algorithm under corrupted training data. The goal is to determine whether SORG can still select informative subsets when the feature space contains a significant fraction of outliers.

We compare SORG against standard coreset and clustering-based baselines under varying levels of dataset contamination.

### Methodology

**1. Feature Extraction (Selection Phase)**  
Images from CIFAR-100 are passed through a pretrained ResNet-18 model to obtain feature embeddings. These embeddings provide a compact numerical representation of each image.

**2. Outlier Injection**  
To simulate real-world data corruption, synthetic outliers are injected into the training pool:

- A fraction of training samples is replaced with outlier samples
- Outliers are generated using heavy-tailed t-distributions
- Outlier feature norms are scaled significantly larger than clean samples
- Contamination levels evaluated: 0%, 10%, 30%

**3. Coreset Selection**  
SORG selects subsets from the contaminated dataset in a group-wise manner. Baseline methods include:

- Random selection
- Herding selection
- K-Center greedy selection
- K-Means minibatch coreset
- SORG (robust selection with SoftNorm gating)
- SORG-NoGate (ablation without gating mechanism)

**4. Model Training (Evaluation Phase)**  
A linear classifier (Ridge regression) is trained using the selected subset. Performance is evaluated on the clean CIFAR-100 test set.

### Evaluation Metrics

- **Test Accuracy**
- **Outlier Fraction in Selected Subset**
- **Training Time**

### Expected Outcome

SORG is expected to reduce the selection of outliers even under high contamination rates, leading to improved robustness and higher accuracy compared to baseline methods.

------------------------------------------------------------------------

## 4. Backdoor Robustness Experiment

### Overview

The backdoor robustness experiment evaluates the resilience of SORG under a security threat model involving both label noise and backdoor (Trojan) attacks. The goal is to determine whether SORG can still select safe and representative subsets when the training data is intentionally poisoned.

We compare Guided SORG against standard coreset and clustering-based baselines under adversarial data corruption.

### Methodology

**1. Feature Extraction (Selection Phase)**  
Images from CIFAR-100 are passed through a pretrained ResNet-18 model to obtain feature embeddings. All embeddings are L2-normalized for consistency.

**2. Backdoor and Noise Injection**  
Two types of corruption are introduced into the training data:

- Label noise (20%): random corruption of training labels
- Backdoor attack (5%): a fixed visual patch is inserted into images and assigned to a target class

The patch acts as a trigger that causes misclassification during evaluation.

**3. Coreset Selection**  
SORG performs guided subset selection under the poisoned training distribution. Baseline methods include:

- Random selection
- Herding selection
- K-Center greedy selection
- K-Means minibatch coreset
- SORG (guided selection with SoftNorm gating enabled)
- SORG-NoGate (ablation without gating mechanism)

**4. Model Training (Evaluation Phase)**  
A linear classifier is trained on the selected subset. Performance is evaluated on:

- Clean CIFAR-100 test set
- Backdoored test set

### Evaluation Metrics

- **Test Accuracy (Clean Data)**
- **Attack Success Rate (ASR)**
- **Backdoor Fraction in Selected Subset**
- **Label Noise Fraction in Selected Subset**
- **Training Time**

### Expected Outcome

SORG is expected to reduce the selection of poisoned samples, leading to lower attack success rates and improved robustness compared to baseline methods, while maintaining competitive clean accuracy.

------------------------------------------------------------------------