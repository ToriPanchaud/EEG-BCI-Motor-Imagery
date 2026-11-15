from pathlib import Path
from mne.datasets import eegbci


data_path = Path(__file__).resolve().parent.parent / "data"
subjects = range(1, 110)
runs = [6, 10, 14]

for subject in subjects:
    eegbci.load_data(
        subject,
        runs,
        path=data_path,
        force_update=False,
        update_path=False,
        verbose=False,
    )
