# PETformer: Ultra-Low-Dose Total-Body PET Imaging Without Structural Prior

## Abstract

Positron Emission Tomography (PET) is essential for non-invasive imaging of metabolic processes in healthcare applications. However, the use of radiolabeled tracers exposes patients to ionizing radiation, raising concerns about carcinogenic potential, and warranting efforts to minimize doses without sacrificing diagnostic quality. In this work, we present a novel neural network architecture, PETformer, designed for denoising ultra-low-dose PET images without requiring structural priors such as CT or MRI. 

Key Highlights:
- Utilizes a U-Net backbone.
- Incorporate multi-headed transposed attention (MDTA) blocks, kernel-basis attention (KBA) and channel attention (CA) mechanisms.
- Trained and validated on a dataset of 317 patients imaged on a total-body uEXPLORER PET/CT scanner.
- Achieved significant superiority over other established denoising algorithms across different dose-reduction factors.

For more details, please refer to the [paper (to be updated)](google.com).

## Architecture

A overview of the network architecture:
![architecture](./images/architecture.png)

## Results

![result1](./images/result1.png)
![result2](./images/result2.png)

For more details, please refer to the [paper (to be updated)](google.com).

## Pre-trained Weights

Pre-trained model weights will be made available post-publication.

## Usage

To use the PETformer model:

```python
from petformer import PETformer

model = PETformer(...)
model.load_state_dict(torch.load('path_to_pretrained_weights.pt'))
model.eval()

# image tensor
input_image = torch.randn(...) # Replace with your image tensor
denoised_image = model(input_image)
```

## Citation

If you found this work useful or used it in your research, please consider citing our paper:
(TBD)


## Contact

For questions or issues, please reach out to [yul055@ucsd.edu](mailto:yul055@ucsd.edu).
