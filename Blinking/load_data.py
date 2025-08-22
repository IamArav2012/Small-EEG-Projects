import os
import numpy as np
from scipy.signal import *
import csv

def load_data(data_folder, subjects):
    # Options to read: 'EEG-IO', 'EEG-VV', 'EEG-VR', 'EEG-MB'
    fs=250

    # function to read stimulations
    def decode_stim(data_path, file_stim):
        interval_corrupt = []
        blinks = []
        n_corrupt = 0
        with open(os.path.join(data_path,file_stim)) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                if row[0]=="corrupt":
                    n_corrupt = int(row[1])
                elif n_corrupt > 0:
                    if float(row[1]) == -1:
                        t_end = data_sig[-1,0]
                    else:
                        t_end = float(row[1])
                    interval_corrupt.append([float(row[0]), t_end])
                    n_corrupt = n_corrupt - 1
                elif row[0]=="blinks":
                    #check that n_corrupt is 0
                    if not n_corrupt==0:
                        print("!Error in parsing")
                else:
                    blinks.append([float(row[0]), int(row[1])])
        blinks = np.array(blinks)

        return interval_corrupt, blinks 

    # Reading data files

    eeg_data_channels_only_list = []
    groundtruth_blinks_list = []
    interval_corrupt_list = []

    if not isinstance(subjects, (tuple, list)):
        raise ValueError("Variable subjects must be a list or tuple containing specified subject indices")
    
    for file_idx in subjects: 
        list_of_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and '_data' in f]
        file_sig = list_of_files[file_idx]
        file_stim = list_of_files[file_idx].replace('_data','_labels')

        # Loading data
        if data_folder == 'EEG-IO' or data_folder == 'EEG-MB':
            data_sig = np.loadtxt(open(os.path.join(data_folder,file_sig), "rb"), delimiter=";", skiprows=1, usecols=(0,1,2))
        elif data_folder == 'EEG-VR' or data_folder == 'EEG-VV':
            data_sig = np.loadtxt(open(os.path.join(data_folder,file_sig), "rb"), delimiter=",", skiprows=5, usecols=(0,1,2))
            data_sig = data_sig[0:(int(200*fs)+1),:]
            data_sig = data_sig[:,0:3]
            data_sig[:,0] = np.array(range(0,len(data_sig)))/fs

        # Loading Stimulations
        interval_corrupt, groundtruth_blinks = decode_stim(data_folder, file_stim)
        eeg_data_channels_only = data_sig[:, 1:3].T 
        eeg_data_channels_only *= 1e-6

        eeg_data_channels_only_list.append(eeg_data_channels_only)
        groundtruth_blinks_list.append(groundtruth_blinks)
        interval_corrupt_list.append(interval_corrupt)

    return np.array(eeg_data_channels_only_list), fs, groundtruth_blinks_list, interval_corrupt_list