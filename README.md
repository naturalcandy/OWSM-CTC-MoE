# OWSM-CTC MoE

Fine-tuning OWSM-CTC with Mixture-of-Experts for efficient speech recognition.

## Requirements

## Setup

### 1. Clone Repository
```bash
git clone <repo-url>
cd owsm-ctc-moe
```

### 2. Create Environment

From the root of the repository:
```bash
bash scripts/setup_env.sh
```

This will:
- Create a conda environment named `owsm-ctc-moe`
- Install PyTorch with CUDA support
- Install ESPnet and dependencies
- Install CUDA toolkit for compiling extensions

### 3. Activate Environment
```bash
conda activate owsm-ctc-moe
```

### 4. Install Flash Attention (Optional)
Flash Attention provides **2-4x training speedup**.
**Skip this step if using V100 or older GPUs.**


In your conda environment run: 
```bash
  (owsm-ctc-moe) user$ pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.0 \
  torchaudio==2.4.0
```

Then run:

```bash
# Installs FlashAttention from prebuilt wheel
pip install flash-attn==2.8.3 --no-build-isolation
```

### 5. Verify Installation
```bash
python scripts/verify_setup.py
```

## Additional Steps

- **Download LibriSpeech:** `bash scripts/download_librispeech.sh`