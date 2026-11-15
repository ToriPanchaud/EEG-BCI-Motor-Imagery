# EEG Motor Imagery Classification

Python code accompanying my article *Can AI Decode What You're Imagining? I Tried to Find Out*, using the PhysioNet EEG Motor Movement/Imagery Dataset.

The V2 pipeline compares CSP + LDA, Minimum Distance to Mean, and Tangent Space + Logistic Regression, with Euclidean or Riemannian alignment.

## Pipeline

1. `V2/download_data.py` — download runs 6, 10 and 14 for 109 subjects.
2. `V2/preprocess_data.py` — filter at 7–30 Hz, resample to 160 Hz, and extract epochs from 1–2 seconds after each cue.
3. `V2/run_experiments.py` — train, evaluate and export balanced accuracy.

Install `requirements.txt` with Python 3.11, then run the scripts in this order. Experiment settings are at the bottom of `run_experiments.py`.

The current default evaluates MDM and TS + LR, with and without RA, on 10 subjects and three classes. Run 6 provides labeled calibration; runs 10 and 14 are used for testing.

This minimal version covers the basic pipeline. It does not reproduce all results in the article.

Built as part of my TKS Focus on neuroscience and brain-computer interfaces.
