# Feature Pyramid Attention Network (FPANet)

Official repository for **“Feature Pyramid Attention Network for Audio-Visual Scene Classification.”**

FPANet is an attention-based framework for acoustic, visual, and audio-visual scene classification. It combines hierarchical features from a ResNet backbone and uses a Feature Pyramid Attention Module (FPAM) to emphasize semantically relevant spatial regions and temporal information.

## Paper

- **Authors:** Liguang Zhou, Yuhongze Zhou, Xiaonan Qi, Junjie Hu, Tin Lun Lam, and Yangsheng Xu
- **Journal:** *CAAI Transactions on Intelligence Technology*, vol. 10, no. 2, pp. 359–374, 2025
- **First published:** November 26, 2024
- **DOI:** [10.1049/cit2.12375](https://doi.org/10.1049/cit2.12375)
- **Article:** [Wiley Online Library](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12375)

## Method Overview

The network extracts multi-scale features from a ResNet backbone and refines them with FPAM. FPAM has three components:

- **Dimension Alignment (DA):** aligns feature channels from different backbone stages.
- **Pyramid Spatial Attention (PSA):** emphasizes informative spatial regions.
- **Pyramid Channel Attention (PCA):** captures salient channel and temporal information.

Audio and visual representations can be used individually or fused for audio-visual scene classification. Attention visualizations in the paper show that the model can focus on meaningful regions while suppressing less informative input.

## Benchmarks

The paper evaluates FPANet on acoustic scene classification (ASC), visual scene classification (VSC), and audio-visual scene classification (AVSC):

- **ASC:** ESC-10 and ESC-50
- **VSC:** Places365-7 and Places365-14
- **AVSC:** TAU Audio-Visual Urban Scenes 2021 (AVUS, associated with the DCASE 2021 challenge) and ADVANCE

Dataset access:

- [ESC-10 and ESC-50](https://github.com/karolpiczak/ESC-50) (ESC-10 is a subset of ESC-50)
- [Places365](http://places2.csail.mit.edu/)
- [TAU Audio-Visual Urban Scenes 2021 development set](https://zenodo.org/records/4477542)
- [TAU Audio-Visual Urban Scenes 2021 evaluation set](https://zenodo.org/records/4767103)
- [ADVANCE dataset](https://zenodo.org/records/3828124)

Please follow each dataset's license and terms of use.

## Reported Results

Results below are reported in the paper and are included here as reference points:

- **AVUS:** 94.1% accuracy for audio-visual scene classification.
- **ADVANCE:** 95.89 F1-score for audio-visual scene classification; the paper reports a 28.8% relative improvement over the CTTA comparison method.

These numbers are paper results; reproducing them depends on the official data splits, preprocessing, pretrained weights, and training configuration.

## Repository Status

The `main` branch currently contains the MIT license only. Model source files, dataset preparation instructions, dependencies, and training/evaluation scripts are not yet available in this repository. This README summarizes the paper and its benchmarks; runnable usage instructions can be added when the implementation is released here.

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

This repository is released under the [MIT License](LICENSE).
