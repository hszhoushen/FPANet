# Feature Pyramid Attention Network (FPANet)

Official PyTorch implementation for **“Feature Pyramid Attention Network for Audio-Visual Scene Classification.”**

FPANet combines hierarchical audio and visual features with a Feature Pyramid Attention Module (FPAM), emphasizing semantically relevant spatial regions and temporal information.

## Paper

- **Authors:** Liguang Zhou, Yuhongze Zhou, Xiaonan Qi, Junjie Hu, Tin Lun Lam, and Yangsheng Xu
- **Journal:** *CAAI Transactions on Intelligence Technology*, vol. 10, no. 2, pp. 359–374, 2025
- **First published:** November 26, 2024
- **DOI:** [10.1049/cit2.12375](https://doi.org/10.1049/cit2.12375)
- **Article:** [Wiley Online Library](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12375)

## Method Overview

The network extracts multi-scale features from a ResNet backbone and refines them with FPAM:

- **Dimension Alignment (DA):** aligns feature channels from different backbone stages.
- **Pyramid Spatial Attention (PSA):** emphasizes informative spatial regions.
- **Pyramid Channel Attention (PCA):** captures salient channel and temporal information.

Audio and visual representations can be used individually or fused for audio-visual scene classification.

## Repository Structure

- `train_fpam_esc.py`, `test_fpam_esc.py`: training and evaluation entry point for FPAM experiments
- `model/`: FPAM, audio feature extraction, graph components, and dataset utilities
- `resnet_audio/`: audio ResNet implementation
- `scripts/fpam/train/`: training recipes for ESC, Places365, DCASE 2021, and ADVANCE
- `scripts/fpam/test/`: evaluation recipes for DCASE 2021 and ADVANCE
- `dataset.py`, `data/`, `utils/`: data loading, normalization, logging, and helper utilities

## Installation

The repository does not provide a pinned requirements file. The original README lists these Python packages; install a PyTorch and torchvision build compatible with your CUDA environment as well:

```bash
pip install torch torchvision
pip install tensorboardX timm librosa pandas matplotlib pyyaml yacs termcolor opencv-python scikit-learn
```

Other dependencies may be needed depending on the selected dataset and code path.

## Datasets

The paper evaluates acoustic scene classification (ASC), visual scene classification (VSC), and audio-visual scene classification (AVSC):

- **ASC:** ESC-10 and ESC-50
- **VSC:** Places365-7 and Places365-14
- **AVSC:** TAU Audio-Visual Urban Scenes 2021 (AVUS, DCASE 2021) and ADVANCE

Dataset access:

- [ESC-10 and ESC-50](https://github.com/karolpiczak/ESC-50) (ESC-10 is a subset of ESC-50)
- [Places365](http://places2.csail.mit.edu/)
- [TAU Audio-Visual Urban Scenes 2021 development set](https://zenodo.org/records/4477542)
- [TAU Audio-Visual Urban Scenes 2021 evaluation set](https://zenodo.org/records/4767103)
- [ADVANCE dataset](https://zenodo.org/records/3828124)

Prepare the datasets and pretrained weights expected by the scripts. Dataset paths and splits may require local configuration; follow each dataset's license and terms of use.

## Training and Evaluation

Run commands from the repository root. The shell recipes select CUDA devices and contain experiment-specific options; inspect and adjust them for your machine and dataset paths before running.

Train on ADVANCE:

```bash
bash scripts/fpam/train/fpam_advance.sh
```

Train on the DCASE 2021 audio-visual dataset:

```bash
bash scripts/fpam/train/fpam_dcase2021-audio-visual.sh
```

Evaluate on DCASE 2021:

```bash
bash scripts/fpam/test/test_fpam_dcase2021_audio_visual.sh
```

Evaluation recipes require trained checkpoints. The scripts use the legacy `torch.distributed.launch` interface, so an older compatible PyTorch setup may be needed.

## Reported Results

The following results are reported in the paper:

- **AVUS:** 94.1% accuracy for audio-visual scene classification.
- **ADVANCE:** 95.89 F1-score for audio-visual scene classification; the paper reports a 28.8% relative improvement over the CTTA comparison method.

Reproduction depends on the official data splits, preprocessing, pretrained weights, and training configuration.

## Citation

```bibtex
@article{zhou2025feature,
  title   = {Feature pyramid attention network for audio-visual scene classification},
  author  = {Zhou, Liguang and Zhou, Yuhongze and Qi, Xiaonan and Hu, Junjie and Lam, Tin Lun and Xu, Yangsheng},
  journal = {CAAI Transactions on Intelligence Technology},
  volume  = {10},
  number  = {2},
  pages   = {359--374},
  year    = {2025},
  doi     = {10.1049/cit2.12375}
}
```

## License

This project is released under the [MIT License](LICENSE).
