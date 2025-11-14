import os
import sys
import torchaudio

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/librispeech"
    os.makedirs(data_dir, exist_ok=True)

    splits = [
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
    ]

    for split in splits:
        try:
            torchaudio.datasets.LIBRISPEECH(
                root=data_dir,
                url=split,
                download=True,
            )
            print("OK:", split)
        except Exception as e:
            print("ERROR:", split, e)

if __name__ == "__main__":
    '''Usage: python download_libri.py [data_dir]'''
    main()
