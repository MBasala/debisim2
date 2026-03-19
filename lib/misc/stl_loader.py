# -----------------------------------------------------------------------------
"""
Utility functions for loading STL files and discovering object pools
for the DEBISim2 baggage simulation pipeline.
"""
# -----------------------------------------------------------------------------

import os
import numpy as np
import trimesh
from lib.misc.util import read_fits_data


# Conversion factors from declared mesh units to millimetres
_UNIT_TO_MM = {
    'mm': 1.0,
    'cm': 10.0,
    'in': 25.4,
    'inches': 25.4,
    'm': 1000.0,
    'meters': 1000.0,
    'metres': 1000.0,
}


def load_stl_as_mask(stl_path, target_voxel_size=None, mm_per_voxel=1.0,
                     mesh_units='m'):
    """
    Load an STL file and voxelize it into a 3D boolean mask.

    The voxelization pitch (real-world size of each voxel) is determined by
    one of two modes:

    * **Native-unit mode** (default): ``mm_per_voxel`` sets the pitch
      directly so that the mask preserves the mesh's real-world dimensions.
      The ``mesh_units`` parameter declares what unit system the STL file
      uses (default ``'m'`` — metres) and the mesh is scaled to millimetres
      before voxelization.
    * **Legacy mode**: if ``target_voxel_size`` is given it overrides
      ``mm_per_voxel`` and normalises the longest axis to that many voxels
      (old behaviour, kept for backward compatibility).

    Caches the result as a .npy file alongside the STL for fast subsequent
    loads.  The cache key includes the pitch and units so that changing
    parameters invalidates stale caches.

    :param stl_path:            path to the STL file
    :param target_voxel_size:   (legacy) target size in voxels for longest
                                axis — overrides mm_per_voxel if given
    :param mm_per_voxel:        real-world size of each voxel in mm
                                (default 1.0, i.e. 1 voxel = 1 mm)
    :param mesh_units:          unit system of the STL file — one of
                                ``'mm'``, ``'cm'``, ``'in'``/``'inches'``,
                                ``'m'``/``'meters'`` (default ``'m'``)
    :return:                    3D boolean numpy array
    """
    # ---- Determine cache path (includes pitch+units for uniqueness) ------
    if target_voxel_size is not None:
        cache_tag = f'_tvs{target_voxel_size}'
    else:
        cache_tag = f'_mpv{mm_per_voxel}_{mesh_units}'
    cache_path = stl_path + cache_tag + '.npy'

    if os.path.exists(cache_path):
        stl_mtime = os.path.getmtime(stl_path)
        cache_mtime = os.path.getmtime(cache_path)
        if cache_mtime > stl_mtime:
            return np.load(cache_path)

    scene_or_mesh = trimesh.load(stl_path)

    # ---- Convert mesh to millimetres -------------------------------------
    scale_to_mm = _UNIT_TO_MM.get(mesh_units, 1.0)
    if scale_to_mm != 1.0:
        if isinstance(scene_or_mesh, trimesh.Scene):
            for g in scene_or_mesh.geometry.values():
                if isinstance(g, trimesh.Trimesh):
                    g.apply_scale(scale_to_mm)
        else:
            scene_or_mesh.apply_scale(scale_to_mm)

    # ---- Collect individual mesh bodies ----------------------------------
    # STL files for mechanical objects (firearms, tools, etc.) are often
    # composed of multiple separate watertight parts (slide, frame, barrel,
    # trigger, magazine …).  We must voxelize and fill each part on its own
    # so that air gaps between parts are preserved, rather than merging
    # everything into one non-manifold blob.
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [g for g in scene_or_mesh.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
    else:
        # Single mesh — split into connected components
        meshes = scene_or_mesh.split(only_watertight=False)
        if len(meshes) == 0:
            meshes = [scene_or_mesh]

    # ---- Compute a common voxel grid from the overall bounding box -------
    full_bounds = np.array([
        np.min([m.bounds[0] for m in meshes], axis=0),
        np.max([m.bounds[1] for m in meshes], axis=0),
    ])
    full_extents = full_bounds[1] - full_bounds[0]

    if target_voxel_size is not None:
        # Legacy mode: normalise longest axis to N voxels
        pitch = max(full_extents) / target_voxel_size
    else:
        # Native-unit mode: 1 voxel = mm_per_voxel mm
        pitch = mm_per_voxel

    origin = full_bounds[0]

    # Grid dimensions
    grid_shape = np.ceil(full_extents / pitch).astype(int)
    # Clamp minimum grid size to 2 to avoid degenerate masks
    grid_shape = np.maximum(grid_shape, 2)
    mask = np.zeros(grid_shape, dtype=bool)

    # ---- Voxelize + fill each part independently, then OR into mask ------
    for part in meshes:
        if not part.is_watertight:
            trimesh.repair.fix_normals(part)
            trimesh.repair.fill_holes(part)
            trimesh.repair.fix_winding(part)
            trimesh.repair.fix_inversion(part)

        try:
            part_vox = part.voxelized(pitch)
        except ValueError:
            # trimesh's default 'subdivide' method can fail on large meshes.
            # Fall back to the 'binvox' or ray-based approach.
            try:
                part_vox = part.voxelized(pitch, method='ray')
            except Exception:
                # Last resort: skip this part
                continue

        if part_vox.matrix.sum() == 0:
            # Part is too thin to register at this pitch — skip
            continue

        if part.is_watertight:
            part_filled = part_vox.fill().matrix.astype(bool)
        else:
            from scipy.ndimage import binary_fill_holes
            part_filled = binary_fill_holes(
                part_vox.matrix.astype(bool))

        # Map this part's local voxel grid into the common grid
        part_origin = part_vox.transform[:3, 3]
        offset = np.round((part_origin - origin) / pitch).astype(int)
        # Clamp to grid bounds
        ps = part_filled.shape
        for ax in range(3):
            if offset[ax] < 0:
                part_filled = part_filled[
                    tuple(slice(-offset[ax], None) if a == ax
                          else slice(None) for a in range(3))]
                offset[ax] = 0
        end = np.minimum(offset + np.array(part_filled.shape), grid_shape)
        slc = tuple(slice(offset[a], end[a]) for a in range(3))
        trimmed = part_filled[
            :end[0]-offset[0], :end[1]-offset[1], :end[2]-offset[2]]
        mask[slc] |= trimmed

    if not mask.any():
        # No parts registered at this pitch — create a minimal 2×2×2 solid
        # so downstream code doesn't crash on an empty mask.
        import warnings
        warnings.warn(
            f"STL '{stl_path}' produced an empty mask at pitch={pitch:.3f}mm. "
            f"The mesh may be too small or too thin for this voxel size.")
        mask = np.ones((2, 2, 2), dtype=bool)

    np.save(cache_path, mask)
    return mask


def orient_mask_upright(mask):
    """
    Orient a 3D boolean mask so its tallest axis aligns with axis 0
    (the gravity/vertical axis in DEBISim2).

    This ensures that liquid fill (which fills along axis 0) produces
    a waterline parallel to the ground plane.

    :param mask:    3D boolean numpy array
    :return:        reoriented 3D boolean numpy array
    """
    shape = mask.shape
    tallest_axis = np.argmax(shape)

    if tallest_axis == 0:
        return mask
    else:
        # Swap the tallest axis with axis 0
        return np.moveaxis(mask, tallest_axis, 0)


def load_mesh_file(file_path, target_voxel_size=None, mm_per_voxel=1.0,
                   mesh_units='m'):
    """
    Load a 3D shape file as a boolean mask. Dispatches based on extension.

    Supported formats:
    - .stl          -> voxelized via trimesh
    - .fits / .fits.gz -> loaded via astropy FITS reader

    :param file_path:           path to the mesh/mask file
    :param target_voxel_size:   (legacy) target voxel count for longest axis
    :param mm_per_voxel:        real-world size of each voxel in mm
    :param mesh_units:          unit system of mesh files (default 'm')
    :return:                    3D numpy array (boolean for STL, data for FITS)
    """
    lower = file_path.lower()
    if lower.endswith('.stl'):
        return load_stl_as_mask(file_path,
                                target_voxel_size=target_voxel_size,
                                mm_per_voxel=mm_per_voxel,
                                mesh_units=mesh_units)
    elif lower.endswith('.fits') or lower.endswith('.fits.gz'):
        return read_fits_data(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def discover_pool(pool_dir):
    """
    Scan a directory for supported 3D shape files (.stl, .fits.gz).

    :param pool_dir:    directory to scan
    :return:            sorted list of absolute file paths; empty list if
                        directory is missing or empty
    """
    if not os.path.isdir(pool_dir):
        return []

    supported_ext = ('.stl', '.fits', '.fits.gz')
    files = []
    for f in os.listdir(pool_dir):
        lower = f.lower()
        if any(lower.endswith(ext) for ext in supported_ext):
            files.append(os.path.join(pool_dir, f))

    return sorted(files)
