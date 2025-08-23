# Hand.md  

This model was trained on the built-in data archive library, `mne.datasets.eegbci`. The preprocessing pipeline includes the following techniques:  

- *Independent Component Analysis (ICA)*  
- *Bandpower Extraction*  
- *Common Spatial Pattern (CSP)*  
- *Feature Scaling*  

The model achieves the following performance benchmarks:  
```
Accuracy: 0.7673 ± 0.026

Loss: 0.4791 ± 0.038
```

These values were obtained from 20 independent training trials.  

---

## Explanations  

### Independent Component Analysis (ICA)  

ICA is a standard method in EEG analysis for artifact removal and noise reduction. It estimates independent components `S` by applying unmixing matrices `W` to the observed signals `X`:  

$$
S = W X
$$  

$$\text{where}$$

$$
S \in \mathbb{R}^{m \times t}, \quad X \in \mathbb{R}^{n \times t}, \quad n = m
$$  

This process is based on two assumptions:  

1. Raw EEG signals are approximately Gaussian.  
2. The extracted components are statistically independent (least Gaussian).  

Manual ICA inspection was performed for each subject to maximize artifact removal accuracy.  

---

### Bandpower Features  

Bandpower features were derived from Power Spectral Density ([PSD](https://en.wikipedia.org/wiki/Spectral_density)) estimates, capturing average power within specific frequency bands (e.g., *Delta*: 0.5–4 Hz, *Beta*: 12–30 Hz). Absolute values were used, as relative scaling is less effective here.  

PSD estimates were computed using the **Welch method** with a `hann` window to minimize spectral leakage. The procedure:  

1. Segment the EEG recording into shorter windows.  
2. Apply a windowing function (e.g., `hann`, `hamming`).  
3. Compute the periodogram for each segment:  

   $
   P_i(f) = \frac{1}{U} \left| \sum_{n=0}^{L-1} x_i'[n] \, e^{-j 2 \pi f n} \right|^2
   $  

4. Average all periodograms to obtain the Welch PSD estimate:  

   $
   \hat{S}_x(f) = \frac{1}{K} \sum_{i=0}^{K-1} P_i(f)
   $  

5. Compute bandpower by summing over the frequency bins in the desired range:  

   $
   \text{Bandpower} = \sum_{k=f_{\text{low}}}^{f_{\text{high}}} \hat{S}_x(f_k)
   $ 

---

### Feature Scaling  

Bandpower features were standardized using `StandardScaler()`. This ensures:  

- Comparability across frequency bands  
- Reduced internal covariate shift  
- Improved gradient stability during optimization  

Standardization is defined as:  

$$
X' = \frac{X - \mu}{\sigma}
$$  

$\mu = \text{mean}$

$ \sigma = \text{standard deviation}$

---

### Common Spatial Patterns (CSP)  

CSP is a feature extraction method widely used in EEG, especially for binary classification. It projects multichannel EEG signals into a space where the variance between two classes is maximally discriminated.  

CSP computes spatial filters `w` by solving the generalized eigenvalue problem:  

$$
C_1 w = \lambda C_2 w
$$  

where  

- $C_1, C_2$ are the normalized covariance matrices of the two classes  
- $w$ is an eigenvector (spatial filter)  
- $\lambda$ is the corresponding eigenvalue  

Filters with the largest eigenvalues maximize variance for **class 1** while minimizing it for **class 2**, and vice versa.  

In this project, CSP was applied during preprocessing to enhance class separability and retain the most discriminative features.