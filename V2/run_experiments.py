import mne
import numpy as np
from pathlib import Path
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from moabb.datasets.preprocessing import EuclideanAlignment
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
import csv
from datetime import datetime
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from pyriemann.preprocessing import Whitening

from threadpoolctl import threadpool_limits

mne.set_log_level("WARNING")

def load_data(subjects, runs, classes):
    processed_path = Path(__file__).resolve().parent / "data" / "processed"

    X_parts = []
    y_parts = []
    subject_parts = []
    run_parts = []

    for subject in subjects:
        for run in runs:
            name = f"S{subject:03d}R{run:02d}"
            epochs = mne.read_epochs(processed_path / f"{name}-epo.fif", preload=True)
            epochs = epochs[classes]

            if not X_parts:
                channel_names = epochs.ch_names
                times = epochs.times
            else:
                assert epochs.ch_names == channel_names, f"Channel mismatch: {name}"
                assert np.array_equal(epochs.times, times), f"Time mismatch: {name}"

            X_parts.append(epochs.get_data())
            y_parts.append(epochs.events[:, -1])
            subject_parts.append(np.full(len(epochs), subject))
            run_parts.append(np.full(len(epochs), run))

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    subject_groups = np.concatenate(subject_parts)
    run_groups = np.concatenate(run_parts)

    return X, y, subject_groups, run_groups

def make_split(subject_groups, run_groups, test_subject, calibration_runs, test_runs):
    assert max(calibration_runs) < min(test_runs), \
        "Calibration must precede testing."

    is_target = subject_groups == test_subject

    source_indices = np.where(~is_target)[0]

    calibration_indices = np.where(
        is_target & np.isin(run_groups, calibration_runs)
    )[0]

    test_indices = np.where(
        is_target & np.isin(run_groups, test_runs)
    )[0]

    assert len(source_indices) > 0, "No source subjects."
    assert len(calibration_indices) > 0, "No calibration epochs."
    assert len(test_indices) > 0, "No test epochs."

    return source_indices, calibration_indices, test_indices

def make_alignment(method):
    if method == "euclid":
        return EuclideanAlignment(estimator="lwf")
    if method == "riemann":
        return Whitening(metric="riemann")
    raise ValueError("Unknown alignment: " + method)

def align_subjects(X, subject_groups, method="euclid"):
    X_aligned = X.copy()

    for subject in np.unique(subject_groups):
        mask = subject_groups == subject

        alignment = make_alignment(method)
        X_aligned[mask] = alignment.fit_transform(X[mask])

    return X_aligned

def make_csp_lda(n_components):
    return Pipeline([
        ("CSP", CSP(n_components=n_components, reg=None, log=True, norm_trace=False)),
        ("LDA", LinearDiscriminantAnalysis()),
    ])

def make_mdm(from_covariances=False):
    steps = []
    if not from_covariances:
        steps.append(("Covariances", Covariances(estimator="lwf")))
    steps.append(("MDM", MDM(metric="riemann")))
    return Pipeline(steps)

def make_ts_lr(from_covariances=False):
    steps = []
    if not from_covariances:
        steps.append(("Covariances", Covariances(estimator="lwf")))
    steps.append(("TangentSpace", TangentSpace(metric="riemann", tsupdate=False)))
    steps.append(("LogisticRegression", LogisticRegression(max_iter=1000)))
    return Pipeline(steps)

def make_config(model, use_calibration=False, alignment="none",
                calibration_runs=(6,), test_runs=(10, 14)):
    if alignment not in ("none", "euclid", "riemann"):
        raise ValueError("Unknown alignment: " + alignment)
    if alignment == "riemann":
        if not isinstance(model.steps[0][1], (MDM, TangentSpace)):
            raise ValueError("Riemannian alignment requires from_covariances=True.")
    return {
        "model": model,
        "use_calibration": use_calibration,
        "alignment": alignment,
        "calibration_runs": calibration_runs,
        "test_runs": test_runs,
    }

def evaluate_model(X, y, subject_groups, run_groups, config):
    scores = []

    with threadpool_limits(limits=1):
        if config["alignment"] == "riemann":
            X = Covariances(estimator="lwf").fit_transform(X)

        for subject in np.unique(subject_groups):
            source_indices, calibration_indices, test_indices = make_split(
                subject_groups,
                run_groups,
                test_subject=subject,
                calibration_runs=config["calibration_runs"],
                test_runs=config["test_runs"],
            )

            X_train = X[source_indices]
            y_train = y[source_indices]

            X_calibration = X[calibration_indices]
            X_test = X[test_indices]

            if config["alignment"] != "none":
                X_train = align_subjects(
                    X_train,
                    subject_groups[source_indices],
                    method=config["alignment"],
                )

                target_alignment = make_alignment(config["alignment"])
                X_calibration = target_alignment.fit_transform(X_calibration)
                X_test = target_alignment.transform(X_test)

            if config["use_calibration"]:
                X_train = np.concatenate([X_train, X_calibration])
                y_train = np.concatenate([y_train, y[calibration_indices]])

            trained_model = clone(config["model"])
            trained_model.fit(X_train, y_train)

            predictions = trained_model.predict(X_test)
            score = balanced_accuracy_score(y[test_indices], predictions)
            scores.append(score)


    return np.array(scores)

def save_results(results, subject_groups, classes, runs, configurations):
    output_path = Path(__file__).resolve().parent / "results" / "metrics"
    output_path.mkdir(parents=True, exist_ok=True)

    date = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file = output_path / f"experiment_{date}.csv"

    with file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "configuration", "subject", "balanced_accuracy", "classes", "runs",
            "use_calibration", "alignment", "calibration_runs", "test_runs",
        ])

        for name, scores in results.items():
            config = configurations[name]

            for subject, score in zip(np.unique(subject_groups), scores):
                writer.writerow([
                    name, subject, score, classes, runs,
                    config["use_calibration"], config["alignment"],
                    list(config["calibration_runs"]), list(config["test_runs"]),
                ])

    print("Results saved:", file)
    return file

def run_experiments(subjects, runs, classes, configurations):
    X, y, subject_groups, run_groups = load_data(subjects, runs, classes)
    results = {}

    for name, config in configurations.items():
        results[name] = evaluate_model(X, y, subject_groups, run_groups, config)

    save_results(results, subject_groups, classes, runs, configurations)

    print("\nFINAL RESULTS")
    for name, scores in results.items():
        print("\nConfiguration:", name)
        print("Scores:", scores)
        print("Mean:", np.mean(scores))

    return results

if __name__ == "__main__":
    configurations = {
        "CSP4_no_calibration": make_config(make_csp_lda(4)),
        "CSP4_supervised_calibration": make_config(
            make_csp_lda(4), use_calibration=True),
        "EA_CSP4_unsupervised_calibration": make_config(
            make_csp_lda(4), alignment="euclid"),
        "EA_CSP4_supervised_calibration": make_config(
            make_csp_lda(4), use_calibration=True, alignment="euclid"),
        "MDM_supervised_calibration": make_config(
            make_mdm(), use_calibration=True),
        "TS_LR_supervised_calibration": make_config(
            make_ts_lr(), use_calibration=True),
        "RA_MDM_supervised_calibration": make_config(
            make_mdm(from_covariances=True), use_calibration=True, alignment="riemann"),
        "RA_TS_LR_supervised_calibration": make_config(
            make_ts_lr(from_covariances=True), use_calibration=True, alignment="riemann"),
    }

    results = run_experiments(
        subjects=range(1, 11),
        runs=[6, 10, 14],
        classes=["rest", "hands", "feet"],
        configurations={
            "MDM_supervised_calibration": configurations["MDM_supervised_calibration"],
            "TS_LR_supervised_calibration": configurations["TS_LR_supervised_calibration"],
            "RA_MDM_supervised_calibration": configurations["RA_MDM_supervised_calibration"],
            "RA_TS_LR_supervised_calibration": configurations["RA_TS_LR_supervised_calibration"],
        },
    )
