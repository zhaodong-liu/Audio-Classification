# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an audio classification project for CSCI-SHU 360 Machine Learning. The system classifies audio files into 4 categories using deep learning models trained on mel-frequency filter bank features extracted from vocal tracks.

## Data Pipeline

### Audio Preprocessing
The project uses a custom mel-frequency filter bank implementation rather than librosa's built-in MFCC:

1. **Source Separation**: Audio files are processed with vocal separation (likely using Spleeter based on `spleeter2.ipynb`)
2. **Feature Extraction** (`audiopreprocess.py`):
   - Loads vocal tracks at 16kHz sample rate
   - Extracts 3-second segments
   - Applies pre-emphasis filter (alpha=0.97)
   - Windowing: 25ms frame size, 10ms stride
   - Computes 40 mel-frequency filter banks
   - Output shape: (299, 40) per audio sample
   - Features saved as `.npz` files with key `arr_0`

### Data Format
- **Training**: 11,886 samples, each with shape (299, 40)
- **Test**: 2,447 samples
- **Labels**: 4 categories (0-3), stored in `train_label.npz` with key `labels`
- Features are stored as individual NPZ files: `{index}feature.npz`

## Model Architecture

### Primary Model: Ensemble ResNet34
Located in `notebooks/ensemble-resnet-dense.ipynb`:

**Architecture**:
- 20 ResNet34 models in ensemble
- Modified first conv layer: accepts 1-channel input (grayscale spectrograms)
- Dropout layer before final classification (p=0.2)
- 4-class output with averaged predictions across ensemble

**Training Configuration**:
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-5)
- Scheduler: CosineAnnealingWarmRestarts (T_0=num_epochs*len(train_loader)//2)
- Loss: CrossEntropyLoss
- Batch size: 32
- Train/validation split: 70/30 (random_state=8)
- Each model in ensemble trains on the same data with different random shuffling

**Custom Dataset Class** (`MyDataset`):
- Converts (299, 40) features to (1, 299, 40) tensors (adds channel dimension)
- Returns float32 features and long labels

### Alternative Model: DenseNet121
Commented code available in the notebook for DenseNet121 ensemble (similar structure to ResNet approach).

## Development Workflow

### Running on Kaggle
**IMPORTANT**: The main training notebook (`notebooks/ensemble-resnet-dense.ipynb`) is designed to run on Kaggle with specific data paths (`/kaggle/input/`). It will not work locally without modifying data paths.

### Local Utilities
These scripts work locally for data preparation (located in `scripts/`):

- `audiopreprocess.py` (root): Extract mel filter bank features from vocal tracks
- `scripts/label_process.py`: Convert label text file to NPZ format
- `scripts/mp3_sort_by_label.py`: Organize MP3 files into label-based directories
- `scripts/waveform.py`: Generate waveform and spectrogram visualizations
- `scripts/sample_labels.py`: Create sample submission CSV files

## Key Implementation Details

### Feature Extraction Process (audiopreprocess.py:10-72)
1. Load 3-second audio segment at 16kHz
2. Apply pre-emphasis: `y[n] = y[n] - 0.97 * y[n-1]`
3. Frame the signal: 400 samples/frame (25ms), 160 sample stride (10ms)
4. Apply Hamming window to each frame
5. Compute FFT (NFFT=1024) to get power spectrum
6. Apply 40 mel filter banks (0Hz to 8000Hz)
7. Convert to log scale (dB)
8. Save as NPZ without normalization

### Training Loop Pattern
- Each model in the ensemble trains independently on shuffled data
- Learning rate updated after each batch using cosine annealing
- Validation accuracy computed after each epoch
- Models saved individually as `.pth` files

### Prediction Pipeline
1. Load test features (2447 samples)
2. Run inference through all ensemble models
3. Average logits across models
4. Take argmax for final prediction
5. Save results to CSV with columns: `id`, `category`

## File Organization

```
├── audiopreprocess.py          # Mel filter bank feature extraction
├── notebooks/                  # Jupyter notebooks
│   ├── ensemble-resnet-dense.ipynb  # Main training notebook (Kaggle)
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
│   ├── final_comp_report.aux   # LaTeX auxiliary file
│   ├── final_comp_report.log   # LaTeX compilation log
│   ├── final_comp_report.synctex.gz  # LaTeX sync file
│   └── final_competition_sp24.pdf    # Competition guidelines
├── pics/                       # Images and visualizations
├── junk/                       # Deprecated code
└── example_submission.csv      # Sample submission format
```

## Notes

- The feature extraction uses a custom implementation of mel filter banks rather than standard librosa MFCC
- All features are computed from isolated vocal tracks, not full mix audio
- The ensemble uses averaging at the logit level, not at the probability level
- Data paths in notebooks assume Kaggle directory structure
