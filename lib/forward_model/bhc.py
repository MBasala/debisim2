"""Beam Hardening Correction (BHC) for polychromatic CT.

Precomputes a lookup table that maps polychromatic sinogram values
to their monochromatic equivalent. Applied in the sinogram domain
before FBP reconstruction.

Theory:
    A monochromatic beam at effective energy E_eff through path length L
    of water produces sinogram value:
        s_mono = mu_water(E_eff) * L

    A polychromatic beam through the same path produces:
        s_poly = -log(sum(spectrum[e] * exp(-mu_water[e] * density * L)))

    The BHC maps s_poly → s_mono via a precomputed LUT indexed by s_poly.

    Since we control the simulation physics, the LUT is computed
    analytically from the spectrum and water mu curve — no physical
    phantom calibration needed.

Metal Artifact Reduction (MAR):
    Dense metals (Fe, Cu, Zn, steel) cause severe beam hardening that
    water-based BHC cannot correct. The MAR pipeline:
    1. Segment metal regions from uncorrected reconstruction
    2. Forward-project the metal-only image
    3. Replace metal-affected sinogram bins with interpolated values
    4. Apply water BHC to the corrected sinogram
    5. Reconstruct, then add back the metal contribution

    This is the Normalized MAR (NMAR) approach from Meyer et al. (2010).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class BeamHardeningCorrector:
    """Precomputed water-based beam hardening correction.

    Parameters
    ----------
    spectrum : array, shape (n_kev,)
        Normalized photon spectrum (sum to 1 or unnormalized — both work).
    mu_water : array, shape (n_kev,)
        Mass attenuation coefficient of water at each keV (cm²/g).
    kev_range : array, shape (n_kev,)
        Photon energies in keV corresponding to spectrum and mu_water.
    water_density : float
        Water density in g/cm³ (default 1.0).
    n_lut_bins : int
        Number of bins in the correction LUT (default 4096).
    max_path_cm : float
        Maximum water path length to simulate (default 50 cm).
    """

    def __init__(
        self,
        spectrum: NDArray[np.float64],
        mu_water: NDArray[np.float64],
        kev_range: NDArray[np.float64],
        water_density: float = 1.0,
        n_lut_bins: int = 4096,
        max_path_cm: float = 50.0,
    ) -> None:
        self.n_lut_bins = n_lut_bins

        spec = np.asarray(spectrum, dtype=np.float64)
        mu_w = np.asarray(mu_water, dtype=np.float64)
        kev = np.asarray(kev_range, dtype=np.float64)

        n = min(len(spec), len(mu_w), len(kev))
        spec = spec[:n]
        mu_w = mu_w[:n]
        kev = kev[:n]

        lac_water = mu_w * water_density  # LAC at each keV
        spec_sum = np.sum(spec)

        # Build paired (s_poly, s_linear) curves over water path lengths.
        #
        # s_poly(L)   = -log(sum(spec * exp(-lac * L)) / sum(spec))
        #               This is what the detector actually measures.
        #
        # s_linear(L) = mu_eff * L
        #               This is what a perfectly linear (monochromatic)
        #               detector would measure, where mu_eff is the
        #               effective polyenergetic LAC at L=0 (thin-sample
        #               limit, before hardening distorts the curve).
        #
        # mu_eff = lim(L->0) s_poly(L) / L
        #        = sum(spec * lac) / sum(spec)
        #
        # The BHC LUT maps s_poly -> s_linear, undoing the nonlinearity
        # that beam hardening introduces for thick objects.

        mu_eff = np.sum(spec * lac_water) / spec_sum

        # Compute effective energy for diagnostic reporting (mean photon energy)
        e_eff = np.sum(kev * spec) / spec_sum if spec_sum > 0 else np.mean(kev)

        n_paths = 10000
        path_lengths = np.linspace(0, max_path_cm, n_paths)

        s_poly = np.zeros(n_paths, dtype=np.float64)
        s_linear = np.zeros(n_paths, dtype=np.float64)

        for i, L in enumerate(path_lengths):
            transmitted = np.sum(spec * np.exp(-lac_water * L))
            if transmitted > 0:
                s_poly[i] = -np.log(transmitted / spec_sum)
            else:
                s_poly[i] = s_poly[i - 1] if i > 0 else 0.0
            s_linear[i] = mu_eff * L

        # Build uniformly-spaced LUT indexed by s_poly
        self.s_poly_max = s_poly[-1]
        self.s_poly_min = 0.0
        self.bin_size = self.s_poly_max / n_lut_bins

        lut_poly_vals = np.linspace(0, self.s_poly_max, n_lut_bins + 1)
        self.lut = np.interp(lut_poly_vals, s_poly, s_linear)

        self.mu_mono = mu_eff
        self.e_eff = e_eff

    def correct(
        self, sinogram: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Apply beam hardening correction to a sinogram.

        Parameters
        ----------
        sinogram : array — polychromatic sinogram values

        Returns
        -------
        corrected : array — linearized (monochromatic-equivalent) sinogram
        """
        # Clamp to LUT range
        sino_clamped = np.clip(sinogram, 0, self.s_poly_max - 1e-10)

        # LUT index
        idx_float = sino_clamped / self.bin_size
        idx = idx_float.astype(np.int64)
        frac = idx_float - idx

        # Linear interpolation between LUT bins
        idx = np.clip(idx, 0, self.n_lut_bins - 1)
        corrected = self.lut[idx] * (1.0 - frac) + self.lut[idx + 1] * frac

        return corrected

    @classmethod
    def from_debisim(
        cls,
        mu_handler,
        spectrum_path: str,
        max_kev: int = 151,
        **kwargs,
    ) -> BeamHardeningCorrector:
        """Build a BHC from DEBISim's mu database and spectrum file.

        Parameters
        ----------
        mu_handler : MuDatabaseHandler
            DEBISim's material database.
        spectrum_path : str
            Path to spectrum file (2-column: keV, intensity).
        max_kev : int
            Maximum keV to read from spectrum.
        """
        spec_data = np.loadtxt(spectrum_path)
        if spec_data.ndim == 1:
            # Single-column spectrum: intensities only, keV implied from 10
            kev = np.arange(10, 10 + min(max_kev, len(spec_data)))
            spectrum = spec_data[:max_kev]
        else:
            kev = spec_data[:max_kev, 0] if spec_data.shape[1] >= 2 else np.arange(10, 10 + max_kev)
            spectrum = spec_data[:max_kev, 1] if spec_data.shape[1] >= 2 else spec_data[:max_kev, 0]

        water = mu_handler.material('water')
        mu_water = water['mu'][:max_kev]
        density = float(water['density'])

        return cls(
            spectrum=spectrum,
            mu_water=mu_water,
            kev_range=kev,
            water_density=density,
            **kwargs,
        )


class MetalArtifactReductor:
    """Normalized Metal Artifact Reduction (NMAR).

    Based on: Meyer et al., "Normalized metal artifact reduction (NMAR)
    in computed tomography," Medical Physics 37(10), 2010.

    Pipeline:
    1. Reconstruct uncorrected image (with artifacts)
    2. Threshold to segment metal voxels (HU > metal_threshold)
    3. Forward-project metal-only image → metal sinogram
    4. Create metal trace mask in sinogram (bins affected by metal)
    5. Replace metal-affected bins with interpolated values
    6. Apply water BHC to corrected sinogram
    7. Reconstruct corrected image
    8. Re-insert metal voxels from original reconstruction

    Parameters
    ----------
    projector : callable
        Forward projection function: image → sinogram.
    reconstructor : callable
        Reconstruction function: sinogram → image.
    bhc : BeamHardeningCorrector
        Water-based BHC to apply after metal removal.
    metal_threshold_hu : float
        HU threshold for metal segmentation (default 3000).
    dilation_pixels : int
        Dilate the metal mask by this many pixels to include
        neighboring artifacts (default 3).
    """

    def __init__(
        self,
        projector,
        reconstructor,
        bhc: BeamHardeningCorrector,
        metal_threshold_hu: float = 3000.0,
        dilation_pixels: int = 3,
    ) -> None:
        self.project = projector
        self.reconstruct = reconstructor
        self.bhc = bhc
        self.metal_threshold = metal_threshold_hu
        self.dilation_pixels = dilation_pixels

    def correct(
        self,
        sinogram: NDArray[np.float64],
        mu_w: float,
    ) -> NDArray[np.float64]:
        """Apply NMAR correction to a sinogram.

        Parameters
        ----------
        sinogram : array, shape (n_rows, n_views, n_det)
            Uncorrected polychromatic sinogram.
        mu_w : float
            Water LAC for HU conversion.

        Returns
        -------
        corrected_image : array
            Reconstructed image with metal artifacts reduced.
        """
        from scipy.ndimage import binary_dilation

        # Step 1: Initial reconstruction (uncorrected)
        image_uncorrected = self.reconstruct(sinogram)

        # Step 2: Segment metal voxels
        # Convert to HU for thresholding
        hu = (image_uncorrected - mu_w) / mu_w * 1000
        metal_mask = hu > self.metal_threshold

        if not np.any(metal_mask):
            # No metal detected — just apply water BHC
            sino_corrected = self.bhc.correct(sinogram)
            return self.reconstruct(sino_corrected)

        # Step 3: Create metal-only image
        metal_image = np.where(metal_mask, image_uncorrected, 0.0)

        # Step 4: Forward-project metal to get metal sinogram trace
        metal_sino = self.project(metal_image)
        metal_trace = metal_sino > 0.01  # binary mask of affected bins

        # Dilate the trace to catch edge artifacts
        if self.dilation_pixels > 0:
            struct = np.ones((1, 1, 2 * self.dilation_pixels + 1))
            metal_trace = binary_dilation(metal_trace, structure=struct)

        # Step 5: Normalize sinogram for interpolation
        # Create a "prior" sinogram from non-metal reconstruction
        non_metal_image = np.where(metal_mask, 0.0, image_uncorrected)
        prior_sino = self.project(non_metal_image)
        prior_sino = np.clip(prior_sino, 1e-10, None)

        # Normalize: sinogram / prior (removes spatial structure)
        normalized = sinogram / prior_sino

        # Interpolate over metal trace in the normalized domain
        # (linear interpolation along detector axis)
        normalized_interp = normalized.copy()
        for row in range(normalized.shape[0]):
            for view in range(normalized.shape[1]):
                line = normalized[row, view, :]
                mask_1d = metal_trace[row, view, :]
                if not np.any(mask_1d):
                    continue
                # Linear interpolation over masked bins
                x = np.arange(len(line))
                valid = ~mask_1d
                if np.sum(valid) < 2:
                    continue
                normalized_interp[row, view, :] = np.interp(
                    x, x[valid], line[valid]
                )

        # Denormalize: multiply back by prior
        sino_replaced = normalized_interp * prior_sino

        # Step 6: Apply water BHC
        sino_corrected = self.bhc.correct(sino_replaced)

        # Step 7: Reconstruct
        corrected_image = self.reconstruct(sino_corrected)

        # Step 8: Re-insert metal from original
        corrected_image = np.where(metal_mask, image_uncorrected, corrected_image)

        return corrected_image
