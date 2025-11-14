# Setup script for OWSM-CTC-MoE environment
# Usage: bash scripts/setup_env.sh
# Must run from repo root
set -euo pipefail

echo "Creating/updating conda environment..."
conda env create -f ./environment.yml 2>/dev/null || conda env update -f environment.yml --prune

echo ""
echo "Complete"
echo ""

echo "Activate with: conda activate owsm-ctc-moe"
echo "Quick check: python scripts/verify_setup.py"

echo "Follow README.md for FlashAttention installation"
echo ""
