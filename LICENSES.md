# Third-Party Licenses

This project includes the following third-party dependencies. Users are
responsible for complying with each license when distributing or deploying
this software.

## Critical — GPL (viral copyleft)

| Package | License | Implication |
|---------|---------|-------------|
| [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox) | GPL-3.0 | Distribution of binaries requires full source disclosure. Internal use only — do NOT ship in production images without replacing or obtaining a commercial license. |

## Permissive — safe for commercial use

| Package | License |
|---------|---------|
| PyTorch | BSD-3-Clause |
| scikit-image | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| trimesh | MIT |
| pydicom | MIT |
| astropy | BSD-3-Clause |
| matplotlib | PSF (BSD-compatible) |
| Gpufit | MIT |
| tqdm | MIT/MPL-2.0 |
| psutil | BSD-3-Clause |
| PyYAML | MIT |
| tabulate | MIT |

## Infrastructure

| Component | License | Notes |
|-----------|---------|-------|
| NVIDIA CUDA Toolkit | [NVIDIA EULA](https://docs.nvidia.com/cuda/eula/) | Free for commercial use |
| NVIDIA Container Runtime | Apache-2.0 | |
| Python | PSF License | |
| Ubuntu (base image) | Various (mostly GPL/LGPL) | Standard OS, no linking concerns |

## DEBISim2

Original DEBISim2 code by Ankit Manerikar et al. is marked "Public Domain"
in source headers. Verify with the original author before commercial
distribution.

## Recommendation for production deployment

Replace ASTRA Toolbox with [TIGRE](https://github.com/CERN/TIGRE) (BSD-3)
or a custom Vulkan compute pipeline before distributing Docker images or
deploying to customer sites.
