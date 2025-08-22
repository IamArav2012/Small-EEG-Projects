# Blinking.md
The `Blinking` folder consists of the following files: 
- `blinking.md` (This file)
- `blinking.py`
- `EEG-VV.zip`
- `load_data.py`

## EEG-VV.zip 
This zipped file contains a copy of the orginal EEG-VV provided by the creators of this dataset. To find the original folders visit [**this website.**](https://gnan.ece.gatech.edu/eeg-eyeblinks/)

## load_data.py 
This is a python file needed to load the eeg data from EEG-VV. It is a modified version f the `read_data.py` file provided by th orignial authors. The scrip thas been adopted for this specific implementation. 

# blinking.py
This is the main script which achieves:
```
Accuracy: 0.945 ± 0.0231

Loss: 0.1855 ± 0.0925
```
These were calculated using 20 different independent training trails. The model is lightweight and takes <1 minute on modern hardware. 

## Refrences
Agarwal, M., & Sivakumar, R. (2019). Blink: A fully automated unsupervised algorithm for eye-blink detection in EEG signals. In 57th Annual Allerton Conference on Communication, Control, and Computing (pp. 1045–1052). IEEE.

Agarwal, M., & Sivakumar, R. (2020). Charge for a whole day: Extending battery life for BCI wearables using a lightweight wake-up command. In Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems (pp. 1–12). ACM.

Gupta, E., Agarwal, M., & Sivakumar, R. (2020). Blink to get in: Biometric authentication for mobile devices using EEG signals. In 2020 IEEE International Conference on Communications (pp. 1–7). IEEE.