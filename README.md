# Source Code for the Experiments conducted as part of the Master Thesis: Attack Strategies and Robustness Certification in Decentralized FMTL Systems
**Zhi Wang, 2025, University of Zurich, Department of Informatics (IFI), Communication Systems Group (CSG)**

## Overview
I evaluate the robustness of a DFMTL framework (FedPer intra-task + HCA cross-task) under seven poisoning attacks, designed and integrated to work within decentralized, task-heterogeneous settings. The implementation spans two canonical scenarios:
- **Class-label heterogeneity** on CIFAR-10 (two task groups: Animals vs. Objects).
- **Task heterogeneity** on CelebA (Multi-label classification vs. Facial-landmark regression).

A subset of clients is attacked in each experiment. I measure (i) the direct degradation on compromised clients and (ii) the collateral impact on unselected clients within the same task group and across task groups via aggregation pathways.

### Attack Catalog (7 total)
**Data poisoning**
1. Random Label Flip  
2. Targeted Label Flipping  
3. Trigger Injection (image-space backdoor, e.g., mosaic patch)

**Model poisoning**
4. Sign Flip  
5. Scaled Boost  
6. AT2FL-style gradient-driven perturbation

**Aggregation/backdoor**
7. Malicious Aggregation Filter (tampering during cross-task aggregation)

### Results & Artifacts
**All trained clients, plots, tables, and configs will be available here:**
 [https://drive.google.com/drive/folders/1Au6neZziuD0q4_pd3T8qO6-kMceEvVpK?usp=sharing](https://drive.google.com/drive/folders/1Au6neZziuD0q4_pd3T8qO6-kMceEvVpK?usp=sharing)

## Installation
### Required Steps
1) Make sure Conda is installed: [https://anaconda.org/anaconda/conda](https://anaconda.org/anaconda/conda)
2) Create the conda environment. Specifications are located in the environment.yml file.
    ```
    conda env create -f environment.yml
    ```
3) Activate Environment
    ```
    conda activate asfdfmtl
    ```
4) If you want to train on a **GPU** install CUDA support. For this to work you need the appropriate hardware and CUDA toolkit. More information can be found here: [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit). Note that training on a GPU is highly recommended - Code was **not** tested on CPU!
    ```
    conda install pytorch-cuda=12.1 -c nvidia
    ```
5) If installed, CUDA can be tested with the following command.
    ```
    nvidia-smi
    ```

## Reproducing the Experiments
Below are the exact commands to reproduce clean (no attack) baselines and seven poisoning attacks on both CelebA (task heterogeneity: multilabel vs. landmarks) and CIFAR-10 (class-label heterogeneity: animals vs. objects).

Run each line from the project root.
### 0) Clean (No Attack)
- Run the code:
    ```
    # CelebA
    python src/run.py --configs_folder configs/celeba --configs celeba_fedper_hca.yml
    # CIFAR-10
    python src/run.py --configs_folder configs/cifar-10 --configs cifar-10_fedper_hca.yml
    ```
### 1) Random Label Flip (Data Poisoning)
- Run the code:
    ```
    # CelebA
    python src/run.py --configs_folder configs/celeba --configs celeba_fedper_hca_poison_random_label_flip.yml
    # CIFAR-10
    python src/run.py --configs_folder configs/cifar-10 --configs cifar-10_fedper_hca_poison_random_label_flip.yml
    ```
### 2) Targeted Label Flip (Data Poisoning)
- Run the code:
    ```
    # CelebA
    python src/run.py --configs_folder configs/celeba --configs celeba_fedper_hca_poison_target_label_flip.yml
    # CIFAR-10
    python src/run.py --configs_folder configs/cifar-10 --configs cifar-10_fedper_hca_poison_target_label_flip.yml
    ```

### 3) Trigger Injection (Data Poisoning)
- Run the code:
    ```
    # CelebA
    python src/run.py --configs_folder configs/celeba --configs celeba_fedper_hca_poison_trigger_injection.yml
    # CIFAR-10
    python src/run.py --configs_folder configs/cifar-10 --configs cifar-10_fedper_hca_poison_trigger_injection.yml
    ```

### 4) Sign Flip (Model Poisoning)
- Run the code:
    ```
    # CelebA
    python src/run.py --configs_folder configs/celeba --configs celeba_fedper_hca_poison_sign_flip.yml
    # CIFAR-10
    python src/run.py --configs_folder configs/cifar-10 --configs cifar-10_fedper_hca_poison_sign_flip.yml
    ```

### 5) Scaled Boost (Model Poisoning)
- Run the code:
    ```
    # CelebA
    python src/run.py --configs_folder configs/celeba --configs celeba_fedper_hca_poison_scale_boost.yml
    # CIFAR-10
    python src/run.py --configs_folder configs/cifar-10 --configs cifar-10_fedper_hca_poison_scale_boost.yml
    ```

### 6) AT2FL (Model Poisoning)
- Run the code:
    ```
    # CelebA
    python src/run.py --configs_folder configs/celeba --configs celeba_fedper_hca_poison_at2fl.yml
    # CIFAR-10
    python src/run.py --configs_folder configs/cifar-10 --configs cifar-10_fedper_hca_poison_at2fl.yml
    ```

### 7) Malicious Aggregation Filter (Backdoor-style Attack)
- Run the code:
    ```
    # CelebA
    python src/run.py --configs_folder configs/celeba --configs celeba_fedper_hca_poison_malicious_aggregation.yml
    # CIFAR-10
    python src/run.py --configs_folder configs/cifar-10 --configs cifar-10_fedper_hca_poison_malicious_aggregation.yml
    ```
---

### Plotting and Evaluation
After running all clean and poisoned experiments, generate evaluation plots and tables:
- Run the code:
    ```
    python src/plot.py
    ```
All results will be saved in the `results/` directory:
- **`Bar_All_Metrics.png`**: Bar charts comparing Clean vs. Poisoned clients across all metrics.  
- **`Cross_Client_Table.png`**: Tabular comparison of Clean vs. Poisoned metrics.  

Each dataset (`cifar-10` and `celeba`) contains **8 subfolders** (1 clean baseline + 7 poisoned attacks). Each poisoned folder includes the bar chart and table for direct comparison with the clean baseline.

## Figures

In addition to the raw results stored under `results/`, I also provide summary figures in the `figures/` directory.  
These plots and tables aggregate the impact of seven poisoning attacks on different clients, using the CIFAR-10 experiments as an example.

**Direct impact**: Shows how each attack affects the poisoned client itself (e.g., AN_C0).  
- The bar chart and summary table compare Loss Δ%, Precision Δ%, Recall Δ%, and F1 Δ% across all attacks.  
- Color coding:  
  - Red → Data poisoning (Random Label Flip, Targeted Label Flip, Trigger Injection)  
  - Green → Model poisoning (Sign Flip, Scaled Boost, AT2FL)  
  - Purple → Backdoor/aggregation attack (Malicious Aggregation Filter)

**Indirect impact (same task)**: Summarizes how an attack on one client propagates to *other clients in the same task group*.
**Indirect impact (different tasks)**: Summarizes cross-task effects.