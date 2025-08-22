# Hand.md

This model is trained on the built-in data archive library, `mne.datasets.eegbvi`. The preprocessing of the data involves techiniques such as: 
- Independent Component Analysis (ICA)
- Bandpower extraction 
- Common Spatial Pattern (CSP)
- Feature Scaling 

The model achieves the following benchmarks:
```
Accuracy: 0.7673 ± 0.026

Loss: 0.4791 ± 0.038
```
These have been calculated using the benchmarks of 20 independent training trails. 

## Explanations
A brief explanation of each preprocessing technique will 
be given. 

### Independent Component Analysis (ICA)
This is a rudimentary technique used in EEG analysis for artifact removal and noise reduction. ICA is an iterative process where `n_components` are found by converging to unmixing matrices, denoted as `S`. Using these matrices we can directly find out the individual ICA components. This iterative process assumes 2 fundamental statistical principles: 

1. Raw EEG are gaussial in nature. 
2. Individual components are deemed as the most statiscally independent aka the least gaussial sources. 

For this implementation, manual ICA inspection were conducted on each subject to ensure maximum accuracy in artifact removal.

### Bandpower features 
We create Power Spectral Density ([PSD](https://en.wikipedia.org/wiki/Spectral_density)) maps and average power ranges with respect to brain wave frequencies (ex. *Delta:* 0.5–4 Hz, *Beta:* 12–30 Hz).

### Scaling 
`StandardScaler() has been utilized to scale bandpower features. Scaling these offer several benifits: 
- Converts absolute bandpower into bandpower arrays which repect the values of other bands. 
- Reduces the internal-shift among the data. 
- Allows for more stable gradient computation.

### Common Spatial Patterns (CSP)  
This is a fundamental technique used in EEG analysis for feature extraction, specifically in binary classification tasks. CSP is a linear spatial filtering method that projects multichannel EEG signals into a new space where the variance between two classes is maximally separated.  

The core idea of CSP is to compute spatial filters `W` that maximize variance for one class while simultaneously minimizing variance for the other. This optimization problem can be formulated as a generalized eigenvalue decomposition of the form:  

$$C_1 w = \lambda C_2 w$$

where `C₁` and `C₂` are the normalized covariance matrices of the two classes, `w` is an eigenvector (the spatial filter), and `λ` is the corresponding eigenvalue. Filters associated with the largest eigenvalues maximize variance for **class 1** while suppressing **class 2**, and vice versa for the smallest eigenvalues.  

In this project, CSP was employed during preprocessing to ensure the most discriminative features were retained.