# Third-Party Licenses

This project uses the following open-source components:

| Component | License | URL |
|-----------|---------|-----|
| SadTalker | Apache 2.0 | https://github.com/OpenTalker/SadTalker |
| GFPGAN | Apache 2.0 | https://github.com/TencentARC/GFPGAN |
| basicsr | Apache 2.0 | https://github.com/XPixelGroup/BasicSR |
| facexlib | Apache 2.0 | https://github.com/xinntao/facexlib |
| PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch |
| Flask | BSD-3-Clause | https://github.com/pallets/flask |
| boto3 | Apache 2.0 | https://github.com/boto/boto3 |

## NVIDIA CUDA

The Docker container uses `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` as a base image.
This image is subject to the [NVIDIA Deep Learning Container License](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license).
By building and using this container, you agree to NVIDIA's license terms.

## Model Weights

SadTalker model checkpoints (`vinthony/SadTalker-V002rc` on HuggingFace) are released
under Apache 2.0 by the original authors (Tencent).
