# Music Genre Classification

Music is core to my identity; my father is a drummer, my mother is a singer, and I grew up playing the piano and saxophone in classical and jazz bands daily. Through my studies in audio-neuro interactions, I can’t help but *wholeheartedly* believe that music is core to the human condition — despite being based on waveforms and harmonics: math. In clinical practice, audio is critical to tasks in health like EEG readings and mental health diagnoses. As both the music and health industries now experience rapid, systemic, AI-related darwinism, it goes without question that ML is becoming a ***pillar***. 

This is a machine learning project classifying audio clips to music genres using logistic regression, random forest, and support vector machine models. It was synthesized to determine the extent to which genre separability is frequency dependent, as well as the interaction of these features.

---

## Overview

The music genre spectra dataset contains short-time Fourier transform (STFT), spectral centroid, spectral bandwidth, and mel-frequency cepstral coefficients (MFCCs). STFT and MFCCs are foundational to both voice-contingent mental health diagnosis and EEG, while spectral centroid and bandwidth are similarly fundamental in mental health diagnosis.

This project trains and evaluates three classical classification models on the
dataset, comparing their performance and examining where and why misclassifications occur within each. The goal is not to optimize predictive accuracy, but to demonstrate clean ML workflow and produce logical reasoning and evaluation.

---

## Repository Format
```
music-genre-classification/
│
├── data/
│   ├── data.csv
│   └── kaggle_train.csv
│
├── figures/
│   ├── centroid_vs_bandwidth.png
│   ├── confusion_matrix_lr.png
│   ├── genre_feature_profile_heatmap.png
│   ├── genre_heatmap.png
│   ├── heat_map.png
│   ├── mfcc_heatmap.png
│   ├── random_forest_lr.png
│   └── svm_lr.png
│
├── heat_map.py
├── logistic_regression.py
├── mfcc_heatmap.py
├── random_forest.py
├── scatterplot.py
├── svm.py
│
└── README.md
```

---

## Extracted features

All features were extracted from 30-second audio clips using the librosa library
and normalized prior to model training, ensuring a consistent feature scale
across all 1,000 samples.


- **STFT Means** —mean magnitude across frequency bins over time across frequency bins
- **Spectral Centroid** — brightness of the sound 
- **Spectral Bandwidth** — spread of frequencies around the centroid
- **MFCCs** — timbral texture and spectral shape 

Additionally, the dataset included tempo, beats, root mean square error (RMSE), rolloff, and zero crossing rate. These features were excluded after preliminary analysis indicated they reduced inter-genre separability, likely reflecting their sensitivity to within-genre variance rather than genre-defining structure.

---

## Models

Three classifiers were trained using scikit-learn with default hyperparameters:

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline; L2 regularization |
| Random Forest | 200 estimators; Gini impurity |
| Support Vector Machine | RBF kernel; C=1.0 |

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 67.5% |
| Random Forest | 64.5% |
| Support Vector Machine | 67.5% |

All three models performed particularly well on spectrally stable genres such as classical and metal, and consistently underperformed on rhythmically-defined genres such as reggae and disco. This reflects the broader limitation of frequency-domain features in capturing temporal structure.

Misclassification patterns are visualized in `figures/svm_lr.png`.

---

## Requirements

Python 3.8+ and the following packages:
```bash
pip install numpy pandas librosa scikit-learn matplotlib seaborn
```

---

## Repository Contents

All analysis and model training is contained within the main Python script.
Figures and confusion matrices are available in the `figures/` directory.
The accompanying paper is available in the `paper/` directory.

---

## Discussion

The results indicate that, while frequency-dependent features provide some level of differentiation, these models struggle with separating genres that are more rhythmically separable. This materializes through the three models performing practically identically, implying that the bottleneck is frequency-dependent feature effectiveness.

A surprising result is the models' ability to accurately classify blues despite its seemingly broader influence. Blues is one of the oldest forms of music that is still relevant, widely considered an ancestor to genres such as jazz, rock, and country. It is unexpected that these models, particularly random forest, can classify blues with very high accuracy. This can likely be attributed to reliance on 12-bar phrasing and the blues scale, implicating that blues is perhaps less rhythmically complex (comparatively) than anticipated. Expanding on blues, this invites further investigation on inter-genre variability.

---

## Author

Joshua Threlkeld  
[GitHub](https://github.com/joshthrelkeld) · [LinkedIn](https://linkedin.com/in/joshuathrelkeld)
