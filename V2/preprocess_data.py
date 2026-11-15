import mne
from pathlib import Path

build_path = Path(__file__).resolve().parent.parent
data_path = build_path / "data" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
processed_path = build_path / "V2" / "data" / "processed"
processed_path.mkdir(parents=True, exist_ok=True)

subjects = range(1, 110)
runs = [6, 10, 14]
event_id = {"feet": 1, "hands": 2, "rest": 3}

l_freq = 7.0
h_freq = 30.0
target_sfreq = 160.0
tmin = 1
tmax = 2

for subject in subjects:
    for run in runs:
        name = f"S{subject:03d}R{run:02d}"
        output_file = processed_path / f"{name}-epo.fif"
        file = data_path / f"S{subject:03d}" / f"{name}.edf"
        raw = mne.io.read_raw_edf(file, preload=True)

        eegbci.standardize(raw)
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
        raw.annotations.rename({"T0": "rest", "T1": "hands", "T2": "feet"})
        raw.set_eeg_reference(projection=True)
        raw.filter(l_freq, h_freq, fir_design="firwin", skip_by_annotation="edge")

        if raw.info["sfreq"] != target_sfreq:
            raw.resample(target_sfreq)

        epochs = mne.Epochs(
            raw,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            proj=True,
            picks="eeg",
            preload=True,
        )
        epochs.save(output_file, fmt="double", overwrite=True)

