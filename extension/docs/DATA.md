# Data: clinics, sources, and the one trap to avoid

Four public clinics, each from a different machine and population. That
difference is the domain shift the project studies, so keep the images grouped
by source and never let one source's images leak across a split.

## The clinics

### Montgomery County (NLM)
138 frontal CXRs, 80 normal / 58 TB. 12-bit PNG, very high resolution
(~4000x4900), single Eureka CR machine. Both classes present, so it is a clean
leave-one-clinic-out fold. The label is the last digit of the filename
(`MCUCXR_0001_0.png` = normal, `..._1.png` = abnormal).

- Download: https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
- Also mirrored on the NLM index and on Kaggle/HuggingFace.
- Licence: de-identified, IRB-exempt, free for research. NLM asks that you cite
  the source and not redistribute outside your group.

### Shenzhen (NLM)
662 frontal CXRs, 326 normal / 336 TB. Philips DR, ~3000x3000 PNG. Both classes
present, another clean fold. Same filename-label convention (`CHNCXR_####_X.png`).

- Download: https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
- Licence: same terms as Montgomery.

### NIAID TB Portals
Roughly 3000 TB-positive CXRs from a TB patient registry. Because it is a
registry of TB patients, it is almost entirely positive (few or no normals).

- Access: register and accept the data-use agreement at
  https://data.tbportals.niaid.nih.gov/ . This is a click-through agreement, not
  an IRB submission, but it is a gate, so plan for it.
- Export images plus a `labels.csv` with `image_path,label` and drop them under
  `data/niaid/`.

### RSNA Pneumonia Detection Challenge
Around 30,000 CXRs (about 10,000 normal, the rest lung opacity). This is a
pneumonia dataset, not TB. Use it as a source of *normal* images, or as an
out-of-domain distractor, rather than as a TB clinic in its own right.

- Access: accept the competition rules on Kaggle, then download from
  https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data . Free for
  research; needs a Kaggle account. Images are DICOM (`.dcm`); `pydicom` reads
  them and the dataset loader handles the conversion.

## Read this before you run leave-one-clinic-out

There is a confounding trap in these datasets. In the common bundled "TB Chest
X-ray Database" (Qatar/Dhaka, on Kaggle and IEEE DataPort), the TB class is
drawn mostly from NIAID (plus NLM and Belarus) and the normal class mostly from
RSNA. If you build clinics that way, the clinic label becomes an almost perfect
proxy for the diagnosis: NIAID is nearly all TB, RSNA is nearly all normal.

Two consequences:

1. A model can "generalize" by learning the machine signature instead of the
   pathology, and you would not catch it, which is exactly the failure this
   project exists to expose.
2. When a single-class clinic is the held-out fold, sensitivity or specificity
   is undefined (there is no positive or no negative to score), and accuracy on
   it is easy to over-read.

So:

- Prefer sources that each contain both classes. Montgomery and Shenzhen do.
- If you use NIAID or RSNA, do it deliberately: mix RSNA normals and NIAID
  positives into balanced clinics, or treat RSNA as a normals-only / OOD source
  and say so in the paper. Do not hold out an all-TB or all-normal clinic and
  report it as a clean generalization result.
- `scripts/build_manifest.py` prints a class-balance table and flags single-class
  clinics. `tbtrust.data.splits.check_split` warns at split time. Heed both.

## After downloading

    python scripts/build_manifest.py --raw data/raw --out data/processed/manifest.csv
    python scripts/clinic_stats.py --manifest data/processed/manifest.csv   # domain-shift numbers

The clinic-stats table (median resolution, mean brightness, mean contrast per
clinic) is the concrete "how different are these machines" figure for the paper.

## Simulating the smartphone-capture gap

These clinics are clean digital exports, not smartphone photos of printed or
lightbox films. `tbtrust.data.degradation` closes that gap synthetically (blur,
glare, shadow, capture angle, compression, resolution loss) at continuous
severity, applied on the fly at load time — no degraded copies on disk and no
extra manifest to build. Set the severity where you need it:

    # a controlled robustness sweep at eval time
    tbtrust-eval --config configs/loco_montgomery.yaml --checkpoint outputs/montgomery/best.ckpt \
        eval.severity_sweep=[0.0,0.25,0.5,0.75,1.0]
    # randomized severity during training (the default)
    tbtrust-train --config configs/loco_montgomery.yaml degradation.train_low=0.0 degradation.train_high=1.0

See `docs/DEGRADATION.md` for the pipeline design, the strategy-comparison
ablation (physics vs. learned vs. real recaptures), and how the weak
degradation label is used both as a training target and as a validation check on
every uncertainty method.

## Citations to include

- Jaeger et al., "Two public chest X-ray datasets for computer-aided screening
  of pulmonary diseases," Quant Imaging Med Surg, 2014 (Montgomery, Shenzhen).
- NIAID TB Portals Program (tbportals.niaid.nih.gov).
- RSNA Pneumonia Detection Challenge (Kaggle).
- Rahman et al., "Reliable Tuberculosis Detection Using Chest X-ray..." if you
  use the bundled Qatar/Dhaka database.
