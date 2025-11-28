"""
Download FLEURS dataset for multilingual MoE experiments.

Training languages (4 languages, ~40h):
- Spanish (es_419) - Romance
- Portuguese (pt_br) - Romance  
- Polish (pl_pl) - Slavic
- Czech (cs_cz) - Slavic

Held-out languages for generalization testing:
- Italian (it_it) - Romance
- French (fr_fr) - Romance
- Slovak (sk_sk) - Slavic
- Croatian (hr_hr) - Slavic
"""

import argparse
from datasets import load_dataset
from pathlib import Path


# Language configs
# Format: (fleurs_code, iso3_code, family, role)
LANGUAGES = {
    # Training languages
    "es_419": ("spa", "romance", "train"),
    "pt_br": ("por", "romance", "train"),
    "pl_pl": ("pol", "slavic", "train"),
    "cs_cz": ("ces", "slavic", "train"),
    # Held-out languages for generalization
    "it_it": ("ita", "romance", "heldout"),
    "fr_fr": ("fra", "romance", "heldout"),
    "sk_sk": ("slk", "slavic", "heldout"),
    "hr_hr": ("hrv", "slavic", "heldout"),
}

TRAIN_LANGUAGES = [k for k, v in LANGUAGES.items() if v[2] == "train"]
HELDOUT_LANGUAGES = [k for k, v in LANGUAGES.items() if v[2] == "heldout"]


def download_language(fleurs_code: str, cache_dir: str = None):
    iso3, family, role = LANGUAGES[fleurs_code]
    print(f"\n{'='*60}")
    print(f"Downloading: {fleurs_code} ({iso3}) - {family} - {role}")
    print(f"{'='*60}")
    
    try:
        ds = load_dataset(
            "google/fleurs",
            fleurs_code,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        
        print(f"  Train samples: {len(ds['train'])}")
        print(f"  Validation samples: {len(ds['validation'])}")
        print(f"  Test samples: {len(ds['test'])}")
        
        # Estimate hours (rough: ~10h per language)
        # More precise: count actual audio duration
        total_samples = len(ds['train']) + len(ds['validation']) + len(ds['test'])
        print(f"  Total samples: {total_samples}")
        print(f"  OK: {fleurs_code}")
        return True
        
    except Exception as e:
        print(f"  ERROR: {fleurs_code} - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download FLEURS languages for multilingual MoE experiments"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/fleurs",
        help="Cache directory for HuggingFace datasets",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Download only training languages (4 languages)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Specific language codes to download (e.g., es_419 pt_br)",
    )
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which languages to download
    if args.languages:
        languages = args.languages
    elif args.train_only:
        languages = TRAIN_LANGUAGES
    else:
        languages = list(LANGUAGES.keys())
    
    print("="*60)
    print("FLEURS Dataset Download")
    print("="*60)
    print(f"Cache directory: {cache_dir}")
    print(f"Languages to download: {languages}")
    
    # Download each language
    success = []
    failed = []
    
    for lang_code in languages:
        if lang_code not in LANGUAGES:
            print(f"WARNING: Unknown language code: {lang_code}")
            failed.append(lang_code)
            continue
            
        if download_language(lang_code, str(cache_dir)):
            success.append(lang_code)
        else:
            failed.append(lang_code)
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successful: {len(success)}/{len(languages)}")
    for lang in success:
        iso3, family, role = LANGUAGES[lang]
        print(f"  ✓ {lang} ({iso3}) - {family}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for lang in failed:
            print(f"  ✗ {lang}")
    
    print("\n" + "="*60)
    print("Language Family Summary:")
    print("="*60)
    
    romance_train = [l for l in success if LANGUAGES.get(l, (None, None, None))[1] == "romance" and LANGUAGES.get(l, (None, None, None))[2] == "train"]
    slavic_train = [l for l in success if LANGUAGES.get(l, (None, None, None))[1] == "slavic" and LANGUAGES.get(l, (None, None, None))[2] == "train"]
    romance_held = [l for l in success if LANGUAGES.get(l, (None, None, None))[1] == "romance" and LANGUAGES.get(l, (None, None, None))[2] == "heldout"]
    slavic_held = [l for l in success if LANGUAGES.get(l, (None, None, None))[1] == "slavic" and LANGUAGES.get(l, (None, None, None))[2] == "heldout"]
    
    print(f"Romance (train): {romance_train}")
    print(f"Slavic (train): {slavic_train}")
    print(f"Romance (held-out): {romance_held}")
    print(f"Slavic (held-out): {slavic_held}")
    
    # Estimate total hours
    train_hours = len([l for l in success if LANGUAGES.get(l, (None, None, None))[2] == "train"]) * 10
    print(f"\nEstimated training data: ~{train_hours}h")


if __name__ == "__main__":
    main()