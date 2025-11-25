# Audio Classification for Gender Recognition

A deep learning project for classifying music snippets into 4 categories based on singer gender, developed as the final project for CSCI-SHU 360 Machine Learning.

**Final Accuracy: 80.42%**

## Overview

This project implements an ensemble of ResNet34 models to classify 3-second audio snippets into 4 gender-based categories. The pipeline includes vocal separation, custom mel-frequency feature extraction, and ensemble learning techniques.

## Project Structure

```
├── audiopreprocess.py          # Mel-frequency filter bank feature extraction
├── notebooks/                  # Jupyter notebooks
│   ├── ensemble-resnet-dense.ipynb  # Main training notebook (final model)
│   ├── old_resnet.ipynb        # Previous ResNet experiments
│   └── spleeter2.ipynb         # Vocal separation workflow
├── scripts/                    # Utility scripts
│   ├── label_process.py        # Convert labels to NPZ format
│   ├── mp3_sort_by_label.py    # Organize audio files by category
│   ├── sample_labels.py        # Generate sample submission files
│   └── waveform.py             # Generate waveform/spectrogram visualizations
├── docs/                       # Documentation and reports
│   ├── final_comp_report.pdf   # Final competition report
│   ├── final_comp_report.tex   # LaTeX source
│   └── final_competition_sp24.pdf  # Competition guidelines
├── pics/                       # Images and visualizations
├── junk/                       # Deprecated code
└── CLAUDE.md                   # AI assistant guidance
```

## Methodology

### 1. Preprocessing

The preprocessing pipeline transforms raw audio into features suitable for deep learning:

#### Vocal Separation
- Uses **Spleeter** to separate vocal tracks from background music
- Improves classification accuracy by ~5%
- Focuses the model on singer characteristics rather than instrumentation

#### Feature Extraction
Custom mel-frequency filter bank implementation:
- Sample rate: 16kHz
- Segment length: 3 seconds
- Pre-emphasis filter: α = 0.97
- Frame size: 25ms (400 samples)
- Frame stride: 10ms (160 samples)
- Hamming window applied to each frame
- FFT size: 1024
- Mel filter banks: 40 banks (0-8000 Hz)
- Output shape: **(299, 40)** per sample

Features are saved as individual `.npz` files for efficient loading during training.

### 2. Model Architecture

**Base Model: ResNet34**
- Modified input layer to accept 1-channel spectrograms (299×40)
- Dropout layer (p=0.2) before final classification
- 4-class output layer
- Pre-trained weights not used (trained from scratch)

**Ensemble Method:**
- 20 independent ResNet34 models
- Each model trained on the same data with different shuffling
- Predictions averaged across all models
- Ensemble improves accuracy by 2-3%

### 3. Training Configuration

- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-5)
- **Loss Function:** CrossEntropyLoss
- **Learning Rate Scheduler:** CosineAnnealingWarmRestarts
- **Batch Size:** 32
- **Epochs:** 10-20
- **Train/Validation Split:** 70/30 (random_state=8)
- **Platform:** Kaggle (GPU acceleration)

### 4. Advanced Techniques

- **Dropout:** Prevents overfitting (rate=0.2)
- **Learning Rate Scheduling:** Cosine annealing with warm restarts
- **Ensemble Learning:** 20-model ensemble for robust predictions
- **Data Augmentation:** Explored mixup (experimental)

## Dataset

- **Training Set:** 11,886 audio samples (3 seconds each)
- **Test Set:** 2,447 audio samples
- **Categories:** 4 classes (gender-based classification)
- **Format:** MP3 → Vocal separation → Mel filter banks → NPZ

## Quick Start

### Prerequisites

```bash
pip install librosa numpy matplotlib torch torchvision
```

### Feature Extraction

```bash
# Extract mel-frequency filter banks from vocal tracks
python audiopreprocess.py
```

### Training

**Note:** The main training notebook is designed to run on Kaggle with specific data paths.

1. Upload data to Kaggle: `/kaggle/input/compettioon/`
2. Open `notebooks/ensemble-resnet-dense.ipynb` on Kaggle
3. Run all cells to train the ensemble model

### Utilities

```bash
# Convert label text file to NPZ format
python scripts/label_process.py

# Organize MP3 files by label
python scripts/mp3_sort_by_label.py

# Generate waveform and spectrogram visualizations
python scripts/waveform.py

# Create sample submission CSV
python scripts/sample_labels.py
```

## Results

| Model | Accuracy |
|-------|----------|
| Single CNN (3 conv layers) | ~65% |
| Single ResNet34 | ~74% |
| ResNet34 Ensemble (4-5 models) | ~77-78% |
| **ResNet34 Ensemble (20 models)** | **80.42%** |

The ensemble approach with 20 models provided the best performance, though it is computationally expensive. The vocal separation preprocessing step was crucial, contributing ~5% accuracy improvement.

## Technical Highlights

1. **Custom Mel Filter Banks:** Hand-implemented mel-frequency feature extraction instead of using pre-built MFCC functions, providing fine-grained control over the audio representation.

2. **Vocal Isolation:** Spleeter-based vocal separation focuses the model on singer characteristics rather than background instrumentation.

3. **Large-Scale Ensemble:** Training 20 independent models and averaging predictions significantly improves robustness and generalization.

4. **Modified ResNet Architecture:** Adapted computer vision architecture (ResNet34) for audio classification by treating spectrograms as single-channel images.

## Limitations and Future Work

- **Hyperparameter Tuning:** Limited exploration of optimal learning rates, epochs, and model architectures
- **Computational Resources:** Training constrained by Kaggle platform limitations (random disconnections)
- **Model Architecture:** Could explore transformer-based models or other audio-specific architectures
- **Data Augmentation:** Further experimentation with mixup, SpecAugment, and time-stretching

## References

- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Spleeter: [Audio Source Separation](https://github.com/deezer/spleeter)
- Mel-Frequency Filter Banks: Standard audio feature extraction technique for speech and music processing
