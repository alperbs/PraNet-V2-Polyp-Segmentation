# Changelog

## v1.0.0 - PraNet-V2 Evaluation Pipeline

### Added

- PraNet-V2 binary polyp segmentation inference pipeline
- GPU/CPU device selection
- Pretrained checkpoint loading
- Multi-dataset inference support
- CVC-300 evaluation
- CVC-ClinicDB evaluation
- Kvasir evaluation
- ETIS-LaribPolypDB evaluation
- Quantitative evaluation pipeline
- CSV evaluation results
- Example segmentation visualization
- Project documentation
- Python dependency specification

### Evaluation

The PraNet-V2 model was evaluated on four benchmark datasets.

| Dataset | Dice | IoU |
|---|---:|---:|
| CVC-300 | 0.898 | 0.827 |
| CVC-ClinicDB | 0.923 | 0.872 |
| Kvasir | 0.907 | 0.853 |
| ETIS-LaribPolypDB | 0.641 | 0.565 |

### Repository

The repository includes the source code, evaluation pipeline, evaluation results, dependency specification, and example segmentation output.

Large datasets, generated prediction masks, virtual environments, and pretrained model weights are excluded from version control.
