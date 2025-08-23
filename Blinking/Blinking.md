# Blinking.md  

The `Blinking` folder contains the following files:  

- `blinking.md` (this file)  
- `blinking.py`  
- `EEG-VV.zip`  
- `load_data.py`  

---

## EEG-VV.zip  

This archive contains a copy of the **EEG-VV** dataset provided by the original authors. To access the original dataset and documentation, visit [this website](https://gnan.ece.gatech.edu/eeg-eyeblinks/).  

---

## load_data.py  

This Python script loads EEG data from the EEG-VV dataset. It is a modified version of the `read_data.py` script released by the original authors, adapted for this specific implementation.  

---

## blinking.py  

This is the primary model script. It achieves the following performance benchmarks:  
```
Accuracy: 0.945 ± 0.0231

Loss: 0.1855 ± 0.0925
```

These values were obtained over 20 independent training trials. The model is lightweight and runs in under one minute on modern hardware.  

*For details on the preprocessing techniques used, see [Hand.md](/Left_or_Right/Hand.md).*  

---

## References  

- Agarwal, M., & Sivakumar, R. (2019). *Blink: A fully automated unsupervised algorithm for eye-blink detection in EEG signals.* In **57th Annual Allerton Conference on Communication, Control, and Computing** (pp. 1045–1052). IEEE.  

- Agarwal, M., & Sivakumar, R. (2020). *Charge for a whole day: Extending battery life for BCI wearables using a lightweight wake-up command.* In **Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems** (pp. 1–12). ACM.  

- Gupta, E., Agarwal, M., & Sivakumar, R. (2020). *Blink to get in: Biometric authentication for mobile devices using EEG signals.* In **2020 IEEE International Conference on Communications** (pp. 1–7). IEEE.