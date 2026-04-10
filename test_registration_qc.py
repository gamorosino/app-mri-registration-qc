#!/usr/bin/env python3
"""
test_registration_qc.py — Unit and integration tests for registration_qc.py

Run with:
    python3 test_registration_qc.py
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import nibabel as nib

# Add repo root to path so we can import registration_qc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registration_qc as rqc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nifti(data: np.ndarray, affine=None) -> nib.Nifti1Image:
    if affine is None:
        affine = np.eye(4)
    return nib.Nifti1Image(data.astype(np.float32), affine)


def _save_nifti(data: np.ndarray, path: str, affine=None):
    img = _make_nifti(data, affine)
    nib.save(img, path)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestNormalize(unittest.TestCase):
    def test_unit_range(self):
        arr = np.array([0.0, 5.0, 10.0])
        out = rqc.normalize(arr)
        self.assertAlmostEqual(out.min(), 0.0)
        self.assertAlmostEqual(out.max(), 1.0)

    def test_constant_array(self):
        arr = np.ones((5, 5))
        out = rqc.normalize(arr)
        self.assertTrue(np.all(out == 0.0))


class TestMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.shape = (20, 20, 20)
        self.a = np.random.rand(*self.shape)
        self.b = self.a.copy()          # identical → perfect similarity
        self.c = np.random.rand(*self.shape)   # independent noise

    # MSE
    def test_mse_identical(self):
        self.assertAlmostEqual(rqc.compute_mse(self.a, self.b), 0.0, places=10)

    def test_mse_positive(self):
        self.assertGreater(rqc.compute_mse(self.a, self.c), 0.0)

    # NCC
    def test_ncc_identical(self):
        self.assertAlmostEqual(rqc.compute_ncc(self.a, self.b), 1.0, places=5)

    def test_ncc_range(self):
        val = rqc.compute_ncc(self.a, self.c)
        self.assertGreaterEqual(val, -1.0)
        self.assertLessEqual(val, 1.0)

    # NMI
    def test_nmi_identical_gt1(self):
        val = rqc.compute_nmi(self.a, self.b)
        self.assertGreater(val, 1.0)

    def test_nmi_positive(self):
        val = rqc.compute_nmi(self.a, self.c)
        self.assertGreater(val, 0.0)

    # SSIM
    def test_ssim_identical(self):
        val = rqc.compute_ssim(self.a, self.b)
        self.assertAlmostEqual(val, 1.0, places=5)

    def test_ssim_range(self):
        val = rqc.compute_ssim(self.a, self.c)
        self.assertGreaterEqual(val, -1.0)
        self.assertLessEqual(val, 1.0)

    # Metrics with mask
    def test_metrics_with_mask(self):
        mask = np.zeros(self.shape, dtype=bool)
        mask[5:15, 5:15, 5:15] = True
        ncc = rqc.compute_ncc(self.a, self.b, mask)
        self.assertAlmostEqual(ncc, 1.0, places=5)
        mse = rqc.compute_mse(self.a, self.b, mask)
        self.assertAlmostEqual(mse, 0.0, places=10)


class TestOverlap(unittest.TestCase):
    def test_perfect_overlap(self):
        m = np.ones((10, 10, 10), dtype=bool)
        res = rqc.compute_overlap(m, m)
        self.assertAlmostEqual(res["dice"], 1.0)
        self.assertAlmostEqual(res["jaccard"], 1.0)

    def test_no_overlap(self):
        a = np.zeros((10, 10, 10), dtype=bool)
        a[:5] = True
        b = np.zeros((10, 10, 10), dtype=bool)
        b[5:] = True
        res = rqc.compute_overlap(a, b)
        self.assertAlmostEqual(res["dice"], 0.0)
        self.assertAlmostEqual(res["jaccard"], 0.0)


class TestQualityLabel(unittest.TestCase):
    def test_excellent(self):
        m = {"ncc": 0.95, "nmi": 1.25, "ssim": 0.95}
        self.assertEqual(rqc._quality_label(m), "excellent")

    def test_poor(self):
        m = {"ncc": 0.1, "nmi": 1.01, "ssim": 0.1}
        self.assertEqual(rqc._quality_label(m), "poor")


class TestExtract3D(unittest.TestCase):
    def test_3d_passthrough(self):
        data = np.ones((5, 5, 5), dtype=np.float32)
        img = _make_nifti(data)
        out = rqc.extract_3d(img)
        self.assertEqual(out.shape, (5, 5, 5))

    def test_4d_mean(self):
        data = np.stack([np.ones((5, 5, 5)) * i for i in range(4)], axis=-1).astype(np.float32)
        img = _make_nifti(data)
        with self.assertWarns(UserWarning):
            out = rqc.extract_3d(img)
        self.assertEqual(out.shape, (5, 5, 5))
        np.testing.assert_allclose(out, np.ones((5, 5, 5)) * 1.5)


# ---------------------------------------------------------------------------
# Integration test — full pipeline on tiny synthetic NIfTI images
# ---------------------------------------------------------------------------

class TestRunQC(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        np.random.seed(0)
        shape = (30, 30, 30)
        fixed_data = np.random.rand(*shape).astype(np.float32)
        # Slightly perturbed version of fixed → good registration
        moving_data = fixed_data + np.random.randn(*shape).astype(np.float32) * 0.05
        mask_data = np.zeros(shape, dtype=np.float32)
        mask_data[10:20, 10:20, 10:20] = 1.0

        self.fixed_path  = os.path.join(self.tmpdir, "fixed.nii.gz")
        self.moving_path = os.path.join(self.tmpdir, "moving.nii.gz")
        self.mask_path   = os.path.join(self.tmpdir, "mask.nii.gz")
        self.out_dir     = os.path.join(self.tmpdir, "output")

        _save_nifti(fixed_data,  self.fixed_path)
        _save_nifti(moving_data, self.moving_path)
        _save_nifti(mask_data,   self.mask_path)

    def test_full_pipeline_no_mask(self):
        metrics = rqc.run_qc(
            fixed_path=self.fixed_path,
            moving_path=self.moving_path,
            output_dir=self.out_dir,
        )
        # Metrics are present
        for key in ("nmi", "ncc", "mse", "ssim", "quality"):
            self.assertIn(key, metrics)

        # NCC should be high for nearly-identical images
        self.assertGreater(metrics["ncc"], 0.9)

        # metrics.json was written
        json_path = os.path.join(self.out_dir, "metrics.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path) as fh:
            loaded = json.load(fh)
        self.assertAlmostEqual(loaded["ncc"], metrics["ncc"], places=6)

        # PNG figures were created for all three views
        for view in ("axial", "coronal", "sagittal"):
            png = os.path.join(self.out_dir, f"qc_{view}.png")
            self.assertTrue(os.path.exists(png), f"Missing: {png}")
            self.assertGreater(os.path.getsize(png), 0)

    def test_full_pipeline_with_mask(self):
        metrics = rqc.run_qc(
            fixed_path=self.fixed_path,
            moving_path=self.moving_path,
            mask_path=self.mask_path,
            output_dir=self.out_dir,
        )
        self.assertIn("quality", metrics)

    def test_full_pipeline_different_grids(self):
        """Moving image with a different voxel grid triggers resampling."""
        np.random.seed(1)
        # 2× zoom moving image (different grid)
        shape_big = (60, 60, 60)
        moving_data = np.random.rand(*shape_big).astype(np.float32)
        affine_big = np.diag([0.5, 0.5, 0.5, 1.0])
        moving_path = os.path.join(self.tmpdir, "moving_big.nii.gz")
        nib.save(nib.Nifti1Image(moving_data, affine_big), moving_path)

        metrics = rqc.run_qc(
            fixed_path=self.fixed_path,
            moving_path=moving_path,
            output_dir=self.out_dir,
        )
        for key in ("nmi", "ncc", "mse", "ssim"):
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
