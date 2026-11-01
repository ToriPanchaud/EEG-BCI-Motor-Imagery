import mne
import numpy as np
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

subject = 1
runs = [6, 10, 14]
l_freq = 7.0
h_freq = 30.0
tmin = 1
tmax = 2

files = eegbci.load_data(subject, runs)
raws = [mne.io.read_raw_edf(file, preload=True) for file in files]

for raw in raws:
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    raw.annotations.rename({"T0": "rest", "T1": "hands", "T2": "feet"})
    raw.set_eeg_reference(projection=True)
    raw.filter(l_freq, h_freq, fir_design="firwin", skip_by_annotation="edge")

X_parts = []
y_parts = []
group_parts = []

for run, raw in zip(runs, raws):
    epochs = mne.Epochs(
        raw,
        event_id=["hands", "feet"], # Choose the classes: ["hands", "feet"] or ["rest", "hands", "feet"]
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        proj=True,
        picks="eeg",
        preload=True,
    )
    X_parts.append(epochs.get_data())
    y_parts.append(epochs.events[:, -1])
    group_parts.append(np.full(len(epochs), run))

X = np.concatenate(X_parts)
y = np.concatenate(y_parts)
groups = np.concatenate(group_parts)

pipeline = Pipeline([
    ("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)), # Change n_components to test different numbers of CSP components
    ("LDA", LinearDiscriminantAnalysis()),
])

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
random_scores = cross_val_score(pipeline, X, y, cv=sss, scoring="balanced_accuracy")
print("Random 80/20 split:", random_scores)
print("Mean balanced accuracy:", np.mean(random_scores))

logo = LeaveOneGroupOut()
run_scores = cross_val_score(pipeline, X, y, groups=groups, cv=logo, scoring="balanced_accuracy")
print("Leave-one-run-out:", run_scores)
print("Mean balanced accuracy:", np.mean(run_scores))
