# EEG Motor Imagery Classification

A simple EEG motor imagery classification project using the PhysioNet EEG Motor Movement/Imagery Dataset.

The model uses Common Spatial Patterns (CSP) for feature extraction and Linear Discriminant Analysis (LDA) for classification.

The EEG signals are filtered between 7–30 Hz and evaluated using random train/test splits and leave-one-run-out cross-validation.

## Experiments

- Hands vs Feet: 95.54% mean balanced accuracy (4 CSP components).
- Rest vs Hands vs Feet: 57.28% mean balanced accuracy (8 CSP components).

Both results use leave-one-run-out cross-validation on subject 1, runs 6, 10 and 14.

To reproduce an experiment, change the `event_id` and `n_components` parameters in the Python script.

Built as part of my TKS Focus on neuroscience and brain-computer interfaces.
