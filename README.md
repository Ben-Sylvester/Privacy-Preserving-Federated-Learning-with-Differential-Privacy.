
# DP-FedAvg: Differentially Private Federated Learning Simulation

Welcome to the official simulation repository for the paper:  
**"Differentially Private Federated Learning for Secure Collaborative Analytics Across 100 Simulated Institutions"**

This codebase implements a modular, privacy-preserving Federated Learning pipeline using PyTorch and [Opacus](https://opacus.ai/). It is designed to evaluate the trade-offs between privacy (Differential Privacy), utility (model accuracy), and data heterogeneity (Non-IID partitions) across simulated institutional nodes.

Whether you are looking to reproduce the paper's results, test new privacy accountants, or experiment with gradient inversion attacks, this repository provides a clean, extensible foundation.


## 📂 Repository Structure

```files
├── main.py                 # The orchestrator: this handles training loops, sigma grids, and evaluation
├── model.py                # The Neural network architectures (MNIST MLP & CIFAR-10 CNN)
├── data.py                 # The Data loading and partitioning (IID & Dirichlet Non-IID)
├── client.py               # The DPClientTrainer: wraps models in Opacus for DP-SGD
├── federated_training.py   # The federated loop: manages rounds, budgets, and client selection
├── server.py               # FedAvg aggregation logic (Weighted averaging of client deltas)
├── attack.py               # Gradient Inversion Attack (Zhu et al.) & PSNR evaluation
└── data/                   # Auto-created directory for downloaded datasets
```

---

## Environment Setup

Differential privacy libraries can be particular about PyTorch versions. We recommend using a dedicated virtual environment.

```bash
# 1. Create and activate a virtual environment
python -m venv dp_fed_env
source dp_fed_env/bin/activate  # On Windows: dp_fed_env\Scripts\activate

# 2. Install dependencies 
# (Ensure you install the CUDA-compatible version of PyTorch if you have an NVIDIA GPU)
pip install torch torchvision opacus numpy
```

---

## Running Locally (VS Code / Terminal)

*A quick heads-up: Simulating 100 clients for 100 rounds with DP-SGD is computationally heavy. While you can run this on a CPU, it will take a very long time. We highly recommend using a machine with an NVIDIA GPU.*

1. **Open the Project**: Open the root folder in VS Code.
2. **Select your Interpreter**: Ensure VS Code is using the `dp_fed_env` Python interpreter you just created.
3. **Configure your Run**: Open `main.py`. At the top of the `main()` function, you'll find the configuration block. Adjust it based on what you want to test:
   ```python
   dataset_name = "mnist"      # Switch to "cifar10" for the harder dataset
   partition = "iid"           # Switch to "non-iid" for Dirichlet label skew (α=0.5)
   num_clients = 100           # Number of simulated institutions
   rounds = 100                # Communication rounds
   sigmas = [0.0, 0.8, 1.1, 2.5] # The noise multipliers (privacy levels) to test
   ```
4. **Execute**: Run the script via the VS Code terminal:
   ```bash
   python main.py
   ```

---

## Running in a Cloud Simulation Lab (Free GPUs)

If you don't have a local GPU, cloud notebooks are the best way to run these experiments. 

### Option A: Google Colab (Recommended)
1. Head to [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Go to **Runtime > Change runtime type** and select **T4 GPU**.
3. In the first cell, install the dependencies:
   ```python
   !pip install -q opacus torchvision
   ```
4. Click the **Folder icon** on the left sidebar and upload all the `.py` files from this repository.
5. Run your scripts in subsequent cells:
   ```python
   !python main.py
   ```
   *(Pro-tip: Mount your Google Drive at the top of the notebook using `from google.colab import drive; drive.mount('/content/drive')` so you can save your trained model weights and logs permanently!)*

### Option B: Kaggle Notebooks
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) and click **New Notebook**.
2. In the right-hand settings panel, turn on **GPU T4 x2**.
3. Click **Add Input > Upload** and drop in your `.py` files.
4. Run `!pip install opacus` and then `!python main.py` in the notebook cells.

---

## Reproducing the Paper's Results

To generate the exact data for the tables and figures in the paper, follow this roadmap:

### 1. Accuracy & Heterogeneity Gaps (Table II & Figure 1)
You need to run `main.py` across different configurations. Keep a log of the final test accuracies.
* **Centralised Baseline**: Train a standard model on the full dataset without FL or DP.
* **FedAvg (No DP)**: Set `sigmas = [0.0]`. Run for both `partition = "iid"` and `"non-iid"`.
* **DP-FedAvg**: Set `sigmas = [1.1]`. Run for both `partition = "iid"` and `"non-iid"`.
* *Repeat this entire process with `dataset_name = "cifar10"` for the CIFAR columns.*

### 2. The Privacy-Utility Curve (Table III)
Set `partition = "iid"` and run the full grid: `sigmas = [0.0, 0.8, 1.1, 2.5]`. 
The console will print the final accuracy and the exact $\epsilon$ logged by the Opacus Rényi accountant at $\delta = 10^{-5}$.

### 3. Gradient Inversion Attack (Table IV)
To prove that DP defeats gradient leakage, run the attack script:
```bash
python attack.py
```
* **No DP**: The script will reconstruct the images and print a high PSNR (Target: ~24.8 dB).
* **With DP**: To simulate the DP defense, modify `attack.py` to add Gaussian noise to the `target_grads` before passing them to the optimizer. The reconstructed images will turn to static, dropping the PSNR to < 5 dB.

---

## Expert Troubleshooting & Gotchas

* **`ModuleNotFoundError: No module named 'server'`**: 
  If you cloned an older version of the code, you might be missing `server.py`. Create it and paste this standard FedAvg logic:
  ```python
  import torch
  def fedavg(global_model, client_updates):
      global_state = global_model.state_dict()
      num_clients = len(client_updates)
      for key in global_state.keys():
          # Sum the deltas and average them
          delta_sum = torch.stack([update[key] for update in client_updates]).sum(dim=0)
          global_state[key] += delta_sum / num_clients
  ```
* **Opacus `Unsupported Module` Errors**: 
  Standard `BatchNorm` layers break DP-SGD because they require tracking individual example statistics. Our `client.py` uses `ModuleValidator.fix()` to automatically swap these for `GroupNorm` or `InstanceNorm`. If you add custom layers to `model.py`, ensure they are DP-compatible.
* **Out of Memory (OOM) on GPU**: 
  Simulating 100 clients sequentially in a single process can eat up VRAM. If you hit an OOM error, lower the `batch_size` in `data.py` (e.g., to 128 or 64) or ensure you are clearing the CUDA cache between sigma runs in `main.py` by adding `torch.cuda.empty_cache()`.

---

## Citation

If you use this codebase or build upon our simulation methodology, please cite our paper:

```bibtex
@article{sylvester2024dpfedavg,
  title={Differentially Private Federated Learning for Secure Collaborative Analytics Across 100 Simulated Institutions},
  author={Sylvester, Benjamin},
  journal={Department of Computer Science, National Open University of Nigeria},
  year={2024}
}
```
"Note: check_epsilon.py computes the privacy budget assuming a uniform client dataset size of n samples. Actual per-client 
ϵ
ϵ may vary slightly due to partition size differences, but this represents a safe, conservative estimate."

---
*Built with PyTorch and Opacus. Questions, issues, or pull requests are welcome!*