import mne
from read_data import load_data
from mne import Epochs
from mne.preprocessing import ICA
from mne.channels import make_standard_montage
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import models, layers
from mne.decoding import CSP
from scipy.signal import welch

def insert_prev_event(blink_events_input, sampling_frequency):

    sample_indices = np.round(blink_events_input[:, 0] * sampling_frequency).astype(int)
    previous_event_ids_column = np.zeros_like(sample_indices)
    new_event_ids_column = blink_events_input[:, 1].astype(int)

    mne_events_array = np.c_[sample_indices, previous_event_ids_column, new_event_ids_column]

    return mne_events_array

eeg_data, sampling_fs, blink_events, corroupted_intervals = load_data(data_folder="EEG-VV", subjects=[1])

eeg_data = eeg_data[0]
blink_events = blink_events[0]
corroupted_intervals = corroupted_intervals[0]

ch_names = ['Fp1', 'Fp2'] 
ch_types = ['eeg', 'eeg']

info = mne.create_info(ch_names=ch_names, sfreq=sampling_fs, ch_types=ch_types)

raw = mne.io.RawArray(eeg_data, info, verbose=False)
annotations = mne.Annotations(
    onset = [],
    duration= [],
    description = []
)
raw.set_annotations(annotations)

filtered_raw = (raw.copy()).filter(l_freq=0.5, h_freq=30., fir_design='firwin')
filtered_raw.set_montage(make_standard_montage('standard_1020'))

for index, value in enumerate(blink_events):
    if value[1] == 2.0 or value[1] == 0.0:
        blink_events[index][1] = 1.0
blink_events = insert_prev_event(blink_events, sampling_fs)

blink_event_onsets = blink_events[:, 0] 
min_gap = int(1.0 * sampling_fs) 

# Create all possible event positions spaced 1 second apart
candidate_onsets = np.arange(min_gap, len(raw.times) - min_gap, min_gap)

# Remove candidates too close to any blink
non_blink_onsets = []
for onset in candidate_onsets:
    if np.all(np.abs(onset - blink_event_onsets) > min_gap):
        non_blink_onsets.append(onset)

non_blink_onsets = np.array(non_blink_onsets, dtype=int)

non_blink_events = np.column_stack([non_blink_onsets, np.zeros_like(non_blink_onsets), np.zeros_like(non_blink_onsets)])
all_events = np.vstack([blink_events, non_blink_events])

ica = ICA(n_components=len(ch_names), random_state=42, max_iter='auto')
ica.fit(filtered_raw)
ica.exclude = [0]
ica.apply(raw)

epochs = Epochs(filtered_raw, all_events,
                    event_id={'non-blink': 0, 'blink': 1},
                    tmin=-0.2, tmax=0.5,
                    baseline=None,
                    preload=True,
                    reject_by_annotation='omit'
)

labels = epochs.events[:, 2]

filtered_raw = epochs.get_data()

def compute_bandpower(data, sf, band):
    band = np.array(band)
    freqs, psd = welch(data, sf, nperseg=sf*2)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[:, idx_band], axis=1)  

delta = []
theta = []

for epoch in filtered_raw:
    delta.append(compute_bandpower(epoch, sampling_fs, [0., 4.]))
    theta.append(compute_bandpower(epoch, sampling_fs, [4., 8.]))

delta = np.array(delta)  # shape (n_epochs, n_channels)
theta = np.array(theta)

features = np.concatenate([delta, theta], axis=1)  # shape (n_epochs, 2 * n_channels)

# Apply CSP
csp = CSP(n_components=len(ch_names))
csp_features = csp.fit_transform(filtered_raw, labels)

# Normalize
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Concatentate
combined = np.concatenate([scaled_features, csp_features], axis=1)

x_temp, x_test, y_temp, y_test = train_test_split(combined, labels, test_size = 0.15, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size = 0.4, random_state=42)

mlp = models.Sequential([
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4), 
    layers.Dense(128, activation="relu"), 
    layers.Dropout(0.3), 

    layers.Flatten(),

    layers.Dense(2, activation="softmax")
])

epochs = 150

mlp.compile(
    optimizer = "adam",
    metrics = ['accuracy'],
    loss = "sparse_categorical_crossentropy"
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", 
    mode = "max", 
    restore_best_weights = True, 
    min_delta = 0.001,
    patience = epochs,
)

mlp.fit(x_train, y_train, epochs=epochs, validation_data=[x_val, y_val], callbacks=[early_stopping_cb])

test_loss, test_acc = mlp.evaluate(x_test, y_test)

print(f"Test Loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2f}")