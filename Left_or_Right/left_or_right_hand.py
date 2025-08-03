import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
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

def compute_bandpower(data, sf, band):
    band = np.array(band)
    freqs, psd = welch(data, sf, nperseg=sf*2)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[:, idx_band], axis=1)

all_features = []
all_labels = []
all_epoch_data = []

subject_ids = [1, 2, 4, 5, 8]  # Expandable, 3, 6, and 7 are bad subjects for these runs (ICA analyzed)

for subj in subject_ids:
    print(f"Processing Subject {subj}...")
    files = eegbci.load_data(subj, [6, 10, 14])
    raw_files = [read_raw_edf(f, preload=True) for f in files]
    raw = concatenate_raws(raw_files)
    # Filter for alpha and beta waves 
    raw.filter(8., 30., fir_design='firwin')

    channels_to_use = ['C3..', 'C4..', 'Cz..']
    raw.pick_channels(channels_to_use)
    raw.rename_channels({'C3..': 'C3', 'C4..': 'C4', 'Cz..': 'Cz'})
    raw.set_montage(make_standard_montage('standard_1020'))

    ica = ICA(n_components=len(channels_to_use), random_state=42, max_iter='auto')
    ica.fit(raw)

    # Remove ica depending on Subject
    if subj == 1:
        ica.exclude = [0,1]
    elif subj == 2:
        ica.exclude = [1,2]
    elif subj == 4:
        ica.exclude = [0,2]
    elif subj == 5:
        ica.exclude = [0,1]
    else: # Has to be 8
        ica.exclude = [0,1]
    ica.apply(raw)

    events, _ = mne.events_from_annotations(raw)
    epochs = Epochs(raw, events,
                    event_id={'T1': 2, 'T2': 3},
                    tmin=0, tmax=4,
                    baseline=None,
                    preload=True)

    fs = int(raw.info['sfreq'])
    epoch_data = epochs.get_data()

    for epoch in epoch_data:
        mu_power = compute_bandpower(epoch, fs, [8, 13])
        beta_power = compute_bandpower(epoch, fs, [13, 30])
        combined = np.concatenate([mu_power, beta_power])
        all_features.append(combined)

    all_labels.extend(np.where(epochs.events[:, 2] == 2, 0, 1))
    all_epoch_data.extend(epoch_data)

all_features = np.array(all_features)
all_labels = np.array(all_labels)
all_epoch_data = np.array(all_epoch_data)

csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
csp_features = csp.fit_transform(all_epoch_data, all_labels)  # epoch_data from your loader

scaler = StandardScaler()
features_scaled = scaler.fit_transform(all_features)

# Combine CSP (of non-scaled bandpower features) and scaled bandpower features
combined_features = np.concatenate([csp_features, features_scaled], axis=1)

x_temp, x_test, y_temp, y_test = train_test_split(combined_features, all_labels, test_size=0.15, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.3, random_state=42)

# Simple (Size Optimized) MLP Classifier
model = models.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='softmax'),
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

total_epochs = 150

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', 
    factor=0.5, 
    patience=25, 
    min_lr=0.01
)

# Just to restore best weights for testing in ram since model checkpoint saves on disk. 
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  
    patience=total_epochs,
    min_delta=0.001,  
    restore_best_weights=True
)

# Train
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=total_epochs, callbacks=[reduce_lr, early_stopping_cb])

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

'''Test Loss: 0.6118, Test Accuracy: 0.7353 (one iteration test accuracy got upto 79.4% and another upto 82.35%. But can also get down to 61% in some iterations)'''