<h1 align="center">mri registration qc</h1>

<p align="center">
  Checks the quality of a diffusion registration by comparing fractional anisotropy maps from a fixed and a registered moving tensor dataset with similarity metrics (NMI, NCC, MSE, SSIM) and QC figures.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.925"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.925-blue.svg" alt="Run on brainlife.io"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Checks the quality of a diffusion registration by comparing fractional anisotropy maps from a fixed and a registered moving tensor dataset with similarity metrics (NMI, NCC, MSE, SSIM) and QC figures.**

Reorients both images to RAS+, averages 4-D images over time, resamples the moving image into the fixed grid if needed, then computes normalised mutual information, normalised cross-correlation, mean squared error and SSIM with a qualitative label (excellent/good/fair/poor). It writes metrics.json and per-view PNG figures (fixed, moving, checkerboard with edge overlay, absolute difference), plus AFNI edge-alignment montages when AFNI is available. Use it after a registration App to check alignment; it does not change the images. This variant reads the FA maps of two tensor datasets; with thr_mask it also computes overlap metrics on thresholded masks.

| | |
|---|---|
| **Inputs** | `neuro/tensor` ×2 |
| **Outputs** | `generic/images` |
| **Container** | `gamorosino/tract_align:v1.0` |
| **Resources** | 1 CPUs · 2 h walltime |
| **Run it** | [doi.org/10.25663/brainlife.app.925](https://doi.org/10.25663/brainlife.app.925) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `tensor_moving` | `neuro/tensor` | yes | Registered (moving) tensor dataset; its FA map is compared with the fixed FA. Should already be registered to the fixed data. |
| `tensor_fixed` | `neuro/tensor` | yes | Reference (fixed) tensor dataset; its FA map is used. |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `figures` | `generic/images` | QC figures (PNG) for axial, coronal and sagittal views and the similarity metrics. |

<details>
<summary><b>Parameters</b> (2)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `checkerboard_tiles` | number | `8` | A visualization that alternates patches of two images in a checkerboard pattern to highlight local alignment differences. |
| `thr_mask` | number |  | A threshold-based mask applied to both fixed and moving images to compute overlap metrics (e.g., Dice), where a single value (LOW) selects voxels with values > LOW, and two values (LOW,HIGH) select voxels with LOW < value < HIGH. |

</details>

<!-- brainlife-app:end -->

## Versions on brainlife.io

This repository hosts several Apps, each built from its own branch.

| App | Branch | Run it | Status |
|---|---|---|---|
| MRI Registration QC (FA maps) | [`v0.7`](https://github.com/gamorosino/app-mri-registration-qc/tree/v0.7) | [doi:10.25663/brainlife.app.925](https://doi.org/10.25663/brainlife.app.925) | current |
| MRI Registration QC (T1w) | [`v0.7`](https://github.com/gamorosino/app-mri-registration-qc/tree/v0.7) | [doi:10.25663/brainlife.app.921](https://doi.org/10.25663/brainlife.app.921) | available |

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.925](https://doi.org/10.25663/brainlife.app.925), choose `neuro/tensor`, `neuro/tensor` from your project, set the parameters and press **Execute**. The results are saved to your project as `generic/images`.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/) (the App runs in `gamorosino/tract_align:v1.0`), [`jq`](https://jqlang.github.io/jq/).

1. Clone this repository and check out the App's branch (`git checkout v0.7`).
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {
       "moving": "/path/to/tensor_moving/moving",
       "fixed": "/path/to/tensor_fixed/fixed",
       "checkerboard_tiles": 8,
       "thr_mask": "<number>"
   }
   ```

3. Run `./main`.

## Troubleshooting

- **Error: 'fixed' image path is missing from config.json**: Provide the fixed (reference) input.
- **Error: 'moving' image path is missing from config.json**: Provide the moving (registered) input.

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fixed` | string | — | Path to the fixed (reference) NIfTI image (**required**) |
| `moving` | string | — | Path to the registered NIfTI image (**required**) |
| `mask` | string / null | null | Path to an optional binary brain mask in fixed space |
| `checkerboard_tiles` | integer | 8 | Number of checkerboard tiles per axis in the overlay figure |

---

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- MRI Registration QC (FA maps). brainlife.io App. [doi:10.25663/brainlife.app.925](https://doi.org/10.25663/brainlife.app.925)
- Hayashi, S., Caron, B. A., Heinsfeld, A. S., … & Pestilli, F. (2024). brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature methods*, 21(5), 809–813.
- Cox, R. W. (1996). AFNI: software for analysis and visualization of functional magnetic resonance neuroimages. Computers and Biomedical research, 29(3), 162-173. ---
- Cox. AFNI: Software for Analysis and Visualization of Functional Magnetic Resonance Neuroimages. *Computers and Biomedical Research* 29:162-173 (1996). [doi:10.1006/cbmr.1996.0014](https://doi.org/10.1006/cbmr.1996.0014)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

**Gabriele Amorosino**
(email:

---
