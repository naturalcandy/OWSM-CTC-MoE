# This script Downloads the MLS (Multilingual LibriSpeech) dataset for s
# Italian, Portuguese, and Polish languages.
#
# These are 3 languages have highest WER on OWSM-CTC:
#   Polish:     31.6% WER,  104h train,  2.1h test,  1.6GB (opus)
#   Portuguese: 23.5% WER,  161h train,  3.7h test,  2.5GB (opus)
#   Italian:    22.1% WER,  247h train,  5.3h test,  3.8GB (opus)
#
#
# Usage:
#   ./scripts/download_mls.sh
#

set -e

DATA_DIR="${1:-data/mls}"
BASE_URL="https://dl.fbaipublicfiles.com/mls"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=============================================="
echo "MLS Download: Italian, Portuguese, Polish"
echo "=============================================="
echo "Output directory: $(pwd)"
echo ""
echo "Files to download:"
echo "  - mls_italian_opus.tar.gz    (3.8 GB)"
echo "  - mls_portuguese_opus.tar.gz (2.5 GB)"
echo "  - mls_polish_opus.tar.gz     (1.6 GB)"
echo "  - Total: ~8 GB"
echo ""

download_and_extract() {
    lang=$1
    filename="mls_${lang}_opus.tar.gz"
    url="${BASE_URL}/${filename}"
    
    if [ -d "mls_${lang}" ]; then
        echo "✓ ${lang} already exists, skipping"
        return
    fi
    
    echo ""
    echo ">>> Downloading ${lang}..."
    
    if [ ! -f "$filename" ]; then
        wget --progress=bar:force -c "$url"
    fi
    
    echo ">>> Extracting ${filename}..."
    tar -xzf "$filename"
    
    echo "✓ ${lang} done"
}

# Download each language
download_and_extract "polish"
download_and_extract "portuguese"  
download_and_extract "italian"

echo "Download complete."