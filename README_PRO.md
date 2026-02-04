# 🚀 DBRouting-Modular: RTX 5090 Server Guide

This guide details how to deploy and run the "Modular Reasoning Re-Ranking" pipeline on a remote Linux server (e.g., RTX 5090 / A100 node).

---

## 1. Connecting to the Server (SSH) 🔑

Open your local terminal (PowerShell, CMD, or Terminal) and run:

```powershell
ssh username@ip_address
# Example: ssh f20230448@10.1.19.142
```

- **Troubleshooting**: If it hangs, check if you need a VPN or specific network permissions.
- **Tip**: To stay alive during network drops, use `tmux` or `screen` immediately after logging in.

### Using TMUX (Highly Recommended)
Once logged in, start a persistent session:
```bash
tmux new -s dbrouting
```
*(To detach later: `Ctrl+B` then `D`)*
*(To reattach later: `tmux attach -t dbrouting`)*

---

## 2. Setting Up the Project 📦

### Option A: Via Git (Easiest)
if your server has internet access:

```bash
# 1. Clone the repository
git clone https://github.com/Vinay12345-neutron/DBRouting-Modular.git

# 2. Enter directory
cd DBRouting-Modular
```

### Option B: Via SCP (If you have local changes not pushed)
Run this from your **LOCAL** computer (new terminal):

```powershell
# Copy the entire directory to the server
scp -r d:\Tanmay_SOP\DBRouting-Modular username@ip_address:~/
```

---

## 3. Environment Setup 🐍

Running on Linux usually requires a virtual environment.

```bash
# 1. Load Python module (if on a cluster/HPC)
# module load python/3.10  <-- Uncomment if needed

# 2. Create venv
python3 -m venv venv

# 3. Activate venv
source venv/bin/activate

# 4. Install Dependencies
# Upgrade pip first
pip install --upgrade pip

# Install project requirements
pip install -r requirements.txt

# Install Critical GPU libraries
pip install bitsandbytes accelerate python-dotenv
```

---

## 4. Configuring for RTX 5090 (14B Model) ⚡

Ensure `reranker_optimized.py` is configured for high performance:

```python
# Check verify these lines in reranker_optimized.py using: nano reranker_optimized.py

LLM_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"  # High-performance model
USE_8BIT_QUANTIZATION = True                  # Efficient loading
MAX_SAMPLES = None                            # Run ALL queries
```

---

## 5. Running the Pipeline 🚀

### Step 1: Prepare Data (If missing)
Check if `processed_data/` exists.
```bash
ls -l processed_data/
```
If empty/missing, ensure you SCP'd the `data/` folder or run:
```bash
python prepare_data.py
```

### Step 2: Run Retrieval (If needed)
```bash
python retrieval.py
```

### Step 3: Run Reranker (Foreground test)
Run for a minute to verify it loads:
```bash
python reranker_optimized.py
```
*Ctrl+C once you see the progress bar moving.*

### Step 4: Run Reranker (Background Job) ✨
Run the full job in the background so you can disconnect:

```bash
# Option A: Inside TMUX (Just run it)
python reranker_optimized.py

# Option B: Using Nohup (if not using tmux)
nohup python reranker_optimized.py > run.log 2>&1 &
```

**Monitoring Progress:**
```bash
tail -f run.log   # or check the output json files via ls -l results/
nvidia-smi -l 5   # Watch GPU usage update every 5 seconds
```

---

## 6. Downloading Results 📥

After completion, copy results back to your local machine.
Run this on your **LOCAL** computer:

```powershell
# Download the results folder
scp -r username@ip_address:~/DBRouting-Modular/results d:\Tanmay_SOP\Results_From_Server
```
