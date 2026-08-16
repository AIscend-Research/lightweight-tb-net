#!/usr/bin/env python3
"""Fetch the TB chest X-ray datasets into data/raw/.

There is no single command that grabs everything, because the sources have
different access rules. This script does what can be automated (Kaggle) and prints
exact instructions for the rest. Run `python scripts/download_data.py --help`.

Sources
-------
1. Aggregated Kaggle TB set  (fastest start; Montgomery+Shenzhen+NIAID+RSNA+Belarus
   already mixed, 3500 TB / 3500 normal, foldered Normal/ and Tuberculosis/):
       kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset
   Needs a Kaggle API token (~/.kaggle/kaggle.json). NOTE: provenance is only
   recoverable per-source for Montgomery/Shenzhen via filename prefixes; NIAID is
   ~all TB, RSNA ~all normal, Belarus all TB. Plan LOCO folds accordingly.

2. Raw NLM Montgomery + Shenzhen  (cleanest per-clinic provenance, both classes):
       https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/
       (Montgomery-County-CXR-Set/ , Shenzhen-Hospital-CXR-Set/)
   Filenames MCUCXR_*_0/1 (Montgomery) and CHNCXR_*_0/1 (Shenzhen); trailing
   0 = normal, 1 = TB. Also mirrored on Kaggle (search "Montgomery Shenzhen CXR").

3. RSNA Pneumonia Detection Challenge  (source of the 'rsna' normals):
       kaggle competitions download -c rsna-pneumonia-detection-challenge
   Competition rules apply; you must accept them on the site first. DICOM format.

4. NIAID TB Portals  (source of 'niaid' TB positives):
       https://data.tbportals.niaid.nih.gov/    (free account + data-use agreement)
   Not scriptable without credentials; register and export CXR images.

After downloading, run scripts/build_manifest.py to produce the unified manifest.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RAW = Path("data/raw")

INSTRUCTIONS = {
    "nlm": (
        "Raw NLM Montgomery + Shenzhen:\n"
        "  Browse https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/\n"
        "  Download Montgomery-County-CXR-Set/ and Shenzhen-Hospital-CXR-Set/ into data/raw/.\n"
        "  Labels are in the filename suffix (_0 = normal, _1 = TB)."
    ),
    "rsna": (
        "RSNA normals:\n"
        "  1) Accept the rules at kaggle.com/c/rsna-pneumonia-detection-challenge\n"
        "  2) kaggle competitions download -c rsna-pneumonia-detection-challenge -p data/raw/rsna\n"
        "  3) Use only the 'Normal' class rows (stage_2 labels csv) to match this project."
    ),
    "niaid": (
        "NIAID TB positives:\n"
        "  Register at https://data.tbportals.niaid.nih.gov/ , accept the data-use\n"
        "  agreement, and export chest X-ray images into data/raw/niaid/."
    ),
}


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def kaggle_aggregated() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / "kaggle_tb"
    code = _run(["kaggle", "datasets", "download", "-d",
                 "tawsifurrahman/tuberculosis-tb-chest-xray-dataset",
                 "-p", str(dest), "--unzip"])
    if code != 0:
        print("\nKaggle download failed. Install the CLI (`pip install kaggle`) and put your "
              "token at ~/.kaggle/kaggle.json (chmod 600). Then re-run.", file=sys.stderr)
    else:
        print(f"Done -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaggle-aggregated", action="store_true",
                    help="download the aggregated Kaggle TB dataset (needs kaggle token)")
    ap.add_argument("--print-instructions", action="store_true",
                    help="print manual steps for NLM / RSNA / NIAID")
    args = ap.parse_args()

    if args.kaggle_aggregated:
        kaggle_aggregated()
    if args.print_instructions or not any(vars(args).values()):
        print("\n".join(INSTRUCTIONS.values()))


if __name__ == "__main__":
    main()
