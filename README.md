# MRI Registration QC

An automated quality-check pipeline for MRI image registration. Given a **fixed
(reference)** image and a **moving (registered)** image — both in NIfTI format —
the app computes four voxel-based similarity metrics and produces visual QC figures
for each anatomical plane (axial, coronal, sagittal).

---

## Author

**Gabriele Amorosino**
(email: [gabriele.amorosino@utexas.edu](mailto:gabriele.amorosino@utexas.edu))

---

## Description

The pipeline performs:

1. **Image loading & canonicalisation** — Images are reoriented to RAS+ canonical
   orientation; 4-D images (e.g. DWI, fMRI) are reduced to their temporal mean
   before metric computation.
2. **Resampling** — If the moving image has a different voxel grid than the fixed
   image, it is resampled into the fixed space via affine-based trilinear
   interpolation.
3. **Metric computation** — Four complementary similarity measures:
   | Metric | Abbreviation | Interpretation |
   |--------|-------------|----------------|
   | Normalised Mutual Information | NMI | > 1, higher is better |
   | Normalised Cross-Correlation  | NCC | ∈ [-1, 1], closer to 1 is better |
   | Mean Squared Error            | MSE | ≥ 0, lower is better |
   | Structural Similarity Index   | SSIM | ∈ [-1, 1], closer to 1 is better |
4. **Quality label** — A qualitative label (`excellent`, `good`, `fair`, `poor`)
   derived from the combined metric thresholds.
5. **Visual QC figures** — For each anatomical view a 4-panel PNG is produced:
   - Fixed image
   - Registered (moving) image
   - Checkerboard + fixed-image edge overlay
   - Absolute difference map
6. **AFNI edge-alignment figures** *(optional, requires AFNI)* — When AFNI is
   available in the environment, `@djunct_edgy_align_check` is run to produce
   an additional set of edge-based alignment montages (`edgy_align.*.png`).
   These use `3dedge3` to outline meaningful anatomical edges of the fixed image
   overlaid on the moving image, providing an intuitive visual check of
   structural alignment.  If AFNI is not installed the step is skipped
   gracefully and the Python-based figures are still produced.
7. **Brainlife product.json** — The PNGs are embedded as base64 images in
   `product.json` so the results are immediately viewable in the brainlife.io UI.

An optional **brain mask** (in fixed image space) restricts all metric calculations
and mid-slice selection to the brain region.

---

## Requirements

To run the app, you only need one of:

- **Singularity / Apptainer**

All required software dependencies are already included in the container image, including:

- Python ≥ 3.8
- `nibabel`
- `numpy`
- `scipy`
- `matplotlib`
- `jq`
- **AFNI 26.1.00**

The pipeline uses `@djunct_edgy_align_check` to generate additional
edge-based alignment figures as part of the QC workflow, and AFNI is
therefore required by the current implementation.

---

## Usage

### Running on Brainlife.io

1. Go to [Brainlife.io](https://brainlife.io) and search for `app-mri-registration-qc`.
2. Click the **Execute** tab.
3. Select the following inputs:
   - **Fixed** — reference NIfTI image
   - **Moving** — registered NIfTI image (must already be in fixed image space)
   - **Mask** (optional) — binary brain mask in fixed image space
4. Submit the job to view the QC report and metrics.

### Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/gamorosino/app-mri-registration-qc.git
   cd app-mri-registration-qc
   ```

2. Prepare a `config.json` file (see `config.json.example`):
   ```json
   {
       "fixed":  "path/to/fixed.nii.gz",
       "moving": "path/to/registered.nii.gz",
       "mask":   null,
       "checkerboard_tiles": 8
   }
   ```

3. Run the pipeline:
   ```bash
   bash ./main
   ```

   Or invoke the Python script directly:
   ```bash
   python3 registration_qc.py \
       --fixed  path/to/fixed.nii.gz  \
       --moving path/to/registered.nii.gz \
       --outdir output/
   ```

---

## Outputs

| Path | Description |
|------|-------------|
| `output/metrics.json` | JSON file with NMI, NCC, MSE, SSIM values and quality label |
| `output/qc_axial.png` | 4-panel QC figure — axial view |
| `output/qc_coronal.png` | 4-panel QC figure — coronal view |
| `output/qc_sagittal.png` | 4-panel QC figure — sagittal view |
| `output/edgy_align.*.png` | AFNI edge-alignment montages (produced when AFNI is available) |
| `product.json` | Brainlife product file embedding all PNGs as base64 |

---

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fixed` | string | — | Path to the fixed (reference) NIfTI image (**required**) |
| `moving` | string | — | Path to the registered NIfTI image (**required**) |
| `mask` | string / null | null | Path to an optional binary brain mask in fixed space |
| `checkerboard_tiles` | integer | 8 | Number of checkerboard tiles per axis in the overlay figure |

---

## Citation

If you use this app in a publication, please cite:

- Hayashi, S., Caron, B. A., Heinsfeld, A. S., … & Pestilli, F. (2024).
  brainlife.io: a decentralized and open-source cloud platform to support
  neuroscience research. *Nature methods*, 21(5), 809–813.
- Cox, R. W. (1996). AFNI: software for analysis and visualization of functional magnetic resonance neuroimages.
  Computers and Biomedical research, 29(3), 162-173.
---
