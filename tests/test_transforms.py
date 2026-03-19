"""
Unit tests for geometric transformations: shape creation, rotation,
scaling, voxelization, mask orientation, and STL loading utilities.
"""

import os
import numpy as np
import pytest
from numpy import array

from lib.bag_generator.baggage_creator_3d import Object3D
from lib.bag_generator.shape_list_handle import ShapeListHandle
from lib.misc.stl_loader import orient_mask_upright, discover_pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_slh = ShapeListHandle()


def _make_object3d(shape, geom, pose=None):
    """Create an Object3D from shape type and geometry dict."""
    if pose is None:
        pose = array([0, 0, 0])
    obj_dict = _slh.create_sim_object(geom=geom, shape=shape,
                                       obj_material='Al', label=99)
    return Object3D(obj_dict=obj_dict, obj_pose=pose)


# ===========================================================================
# orient_mask_upright
# ===========================================================================

class TestOrientMaskUpright:

    def test_already_upright(self):
        """Mask with tallest axis already on axis 0 is unchanged."""
        mask = np.ones((10, 5, 5), dtype=bool)
        result = orient_mask_upright(mask)
        assert result.shape == (10, 5, 5)
        assert np.array_equal(result, mask)

    def test_tallest_on_axis1(self):
        """Tallest axis on axis 1 should move to axis 0."""
        mask = np.ones((5, 20, 5), dtype=bool)
        result = orient_mask_upright(mask)
        assert result.shape[0] == 20
        assert result.shape == (20, 5, 5)

    def test_tallest_on_axis2(self):
        """Tallest axis on axis 2 should move to axis 0."""
        mask = np.ones((3, 3, 15), dtype=bool)
        result = orient_mask_upright(mask)
        assert result.shape[0] == 15

    def test_preserves_voxel_count(self):
        """Total True voxels should not change."""
        mask = np.random.rand(4, 10, 6) > 0.5
        result = orient_mask_upright(mask)
        assert result.sum() == mask.sum()

    def test_cube_is_identity(self):
        """A cubic mask should return unchanged (axis 0 wins ties)."""
        mask = np.ones((8, 8, 8), dtype=bool)
        result = orient_mask_upright(mask)
        assert result.shape == (8, 8, 8)

    def test_preserves_content(self):
        """Specific voxels should be findable after reorientation."""
        mask = np.zeros((4, 12, 6), dtype=bool)
        mask[2, 5, 3] = True
        result = orient_mask_upright(mask)
        # Tallest is axis 1 (12), moved to axis 0
        assert result.shape == (12, 4, 6)
        assert result.sum() == 1
        assert result[5, 2, 3] == True


# ===========================================================================
# discover_pool
# ===========================================================================

class TestDiscoverPool:

    def test_empty_dir(self, tmp_path):
        result = discover_pool(str(tmp_path))
        assert result == []

    def test_nonexistent_dir(self, tmp_path):
        result = discover_pool(str(tmp_path / 'nope'))
        assert result == []

    def test_finds_stl_files(self, tmp_path):
        (tmp_path / 'a.stl').write_text('dummy')
        (tmp_path / 'b.stl').write_text('dummy')
        (tmp_path / 'c.txt').write_text('ignored')
        result = discover_pool(str(tmp_path))
        assert len(result) == 2
        assert all(f.endswith('.stl') for f in result)

    def test_finds_fits_gz(self, tmp_path):
        (tmp_path / 'mask.fits.gz').write_text('dummy')
        result = discover_pool(str(tmp_path))
        assert len(result) == 1

    def test_returns_sorted(self, tmp_path):
        for name in ['c.stl', 'a.stl', 'b.stl']:
            (tmp_path / name).write_text('dummy')
        result = discover_pool(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert basenames == ['a.stl', 'b.stl', 'c.stl']

    def test_case_insensitive_extension(self, tmp_path):
        (tmp_path / 'model.STL').write_text('dummy')
        result = discover_pool(str(tmp_path))
        assert len(result) == 1


# ===========================================================================
# Ellipsoid creation
# ===========================================================================

class TestCreateEllipsoid:

    def test_sphere_is_roughly_cubic(self):
        """A sphere (equal semi-axes) should produce a roughly cubic mask."""
        geom = dict(center=array([0, 0, 0]),
                    axes=array([10, 10, 10]),
                    rot=array([0, 0, 0]))
        obj = _make_object3d('E', geom)
        assert obj.data.ndim == 3
        # Dimensions should be approximately equal for a sphere
        assert abs(obj.dim[0] - obj.dim[1]) <= 2
        assert abs(obj.dim[1] - obj.dim[2]) <= 2

    def test_elongated_ellipsoid(self):
        """An elongated ellipsoid should have one dominant axis."""
        geom = dict(center=array([0, 0, 0]),
                    axes=array([5, 5, 20]),
                    rot=array([0, 0, 0]))
        obj = _make_object3d('E', geom)
        # Third axis should be longest
        assert obj.dim[2] > obj.dim[0]

    def test_rotation_preserves_voxel_count(self):
        """Rotation should approximately preserve the number of True voxels."""
        geom_no_rot = dict(center=array([0, 0, 0]),
                           axes=array([8, 6, 4]),
                           rot=array([0, 0, 0]))
        geom_rot = dict(center=array([0, 0, 0]),
                        axes=array([8, 6, 4]),
                        rot=array([45, 30, 60]))
        obj_no_rot = _make_object3d('E', geom_no_rot)
        obj_rot = _make_object3d('E', geom_rot)
        count_no_rot = obj_no_rot.data.sum()
        count_rot = obj_rot.data.sum()
        # Allow 15% tolerance due to nearest-neighbor interpolation
        assert abs(count_rot - count_no_rot) / count_no_rot < 0.15

    def test_data_contains_label(self):
        """Ellipsoid data values should be 0 (empty) or the assigned label."""
        geom = dict(center=array([0, 0, 0]),
                    axes=array([6, 6, 6]),
                    rot=array([0, 0, 0]))
        obj = _make_object3d('E', geom)
        unique_vals = set(np.unique(obj.data))
        assert unique_vals <= {0, 99}  # label=99 from _make_object3d

    def test_data_is_nonempty(self):
        geom = dict(center=array([0, 0, 0]),
                    axes=array([5, 5, 5]),
                    rot=array([0, 0, 0]))
        obj = _make_object3d('E', geom)
        assert obj.data.sum() > 0


# ===========================================================================
# Cuboid creation
# ===========================================================================

class TestCreateCuboid:

    def test_axis_aligned_box(self):
        """A non-rotated box should have dimensions matching input."""
        geom = dict(center=array([0, 0, 0]),
                    dim=array([10, 20, 30]),
                    rot=array([0, 0, 0]))
        obj = _make_object3d('B', geom)
        # Dimensions should be close to (10, 20, 30) + 1 for inclusive range
        np.testing.assert_allclose(obj.dim, [11, 21, 31], atol=2)

    def test_cube(self):
        """A cube should produce equal-ish dimensions."""
        geom = dict(center=array([0, 0, 0]),
                    dim=array([15, 15, 15]),
                    rot=array([0, 0, 0]))
        obj = _make_object3d('B', geom)
        assert abs(obj.dim[0] - obj.dim[1]) <= 1
        assert abs(obj.dim[1] - obj.dim[2]) <= 1

    def test_90_degree_rotation_swaps_axes(self):
        """Rotating 90 degrees about one axis should swap two dimensions."""
        geom_orig = dict(center=array([0, 0, 0]),
                         dim=array([10, 20, 5]),
                         rot=array([0, 0, 0]))
        geom_rot = dict(center=array([0, 0, 0]),
                        dim=array([10, 20, 5]),
                        rot=array([90, 0, 0]))
        obj_orig = _make_object3d('B', geom_orig)
        obj_rot = _make_object3d('B', geom_rot)
        # After 90-degree rotation about axes (1,0), dims should permute
        orig_sorted = sorted(obj_orig.dim)
        rot_sorted = sorted(obj_rot.dim)
        np.testing.assert_allclose(orig_sorted, rot_sorted, atol=2)

    def test_rotation_preserves_voxel_count(self):
        geom_no_rot = dict(center=array([0, 0, 0]),
                           dim=array([8, 12, 6]),
                           rot=array([0, 0, 0]))
        geom_rot = dict(center=array([0, 0, 0]),
                        dim=array([8, 12, 6]),
                        rot=array([30, 45, 60]))
        obj0 = _make_object3d('B', geom_no_rot)
        obj1 = _make_object3d('B', geom_rot)
        count0 = obj0.data.sum()
        count1 = obj1.data.sum()
        assert abs(count1 - count0) / count0 < 0.15

    def test_data_contains_label(self):
        """Cuboid data values should be 0 (empty) or the assigned label."""
        geom = dict(center=array([0, 0, 0]),
                    dim=array([5, 5, 5]),
                    rot=array([0, 0, 0]))
        obj = _make_object3d('B', geom)
        unique_vals = set(np.unique(obj.data))
        assert unique_vals <= {0, 99}


# ===========================================================================
# Custom object (M shape) — rotation + scale
# ===========================================================================

class TestCreateCustomObject:

    def _make_custom(self, mask, rot=(0, 0, 0), scale=1.0):
        """Helper to create a custom object from a binary mask."""
        geom = dict(
            center=array([0, 0, 0]),
            dim=array(mask.shape, dtype=float),
            rot=array(rot),
            scale=scale,
            src='test_inline',
            mask=mask.copy(),
        )
        return _make_object3d('M', geom)

    def test_identity_transform(self):
        """No rotation, scale=1.0 should preserve the filled voxel count."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:8, 2:8, 2:8] = True
        obj = self._make_custom(mask, rot=(0, 0, 0), scale=1.0)
        # After bounding-box crop, the filled region is preserved
        assert (obj.data > 0).sum() == mask.sum()

    def test_scale_up_increases_voxels(self):
        """scale > 1.0 should increase total voxel count."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[3:7, 3:7, 3:7] = True
        obj_s1 = self._make_custom(mask, scale=1.0)
        obj_s2 = self._make_custom(mask, scale=2.0)
        assert obj_s2.data.sum() > obj_s1.data.sum()

    def test_scale_down_decreases_voxels(self):
        """scale < 1.0 should decrease total voxel count."""
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[5:15, 5:15, 5:15] = True
        obj_s1 = self._make_custom(mask, scale=1.0)
        obj_half = self._make_custom(mask, scale=0.5)
        assert obj_half.data.sum() < obj_s1.data.sum()

    def test_rotation_preserves_count(self):
        mask = np.zeros((12, 12, 12), dtype=bool)
        mask[3:9, 3:9, 3:9] = True
        obj0 = self._make_custom(mask, rot=(0, 0, 0))
        obj1 = self._make_custom(mask, rot=(45, 30, 60))
        count0 = obj0.data.sum()
        count1 = obj1.data.sum()
        assert abs(count1 - count0) / count0 < 0.15

    def test_output_is_nonzero(self):
        """Custom object from a solid mask should produce nonzero data."""
        mask = np.ones((5, 5, 5), dtype=bool)
        obj = self._make_custom(mask)
        assert (obj.data > 0).sum() > 0

    def test_empty_mask_produces_fallback(self):
        """An all-zero mask should produce a small fallback solid."""
        mask = np.zeros((5, 5, 5), dtype=bool)
        obj = self._make_custom(mask)
        # Should have at least a 2x2x2 fallback
        assert obj.data.sum() >= 8


# ===========================================================================
# ShapeListHandle.create_sim_object
# ===========================================================================

class TestCreateSimObject:

    def test_ellipsoid_dict(self):
        geom = dict(center=array([0, 0, 0]),
                    axes=array([5, 5, 5]),
                    rot=array([0, 0, 0]))
        result = _slh.create_sim_object(geom=geom, shape='E',
                                         obj_material='Fe', label=42)
        assert result['shape'] == 'E'
        assert result['material'] == 'Fe'
        assert result['label'] == 42
        assert result['lqd_flag'] == False

    def test_box_dict(self):
        geom = dict(center=array([0, 0, 0]),
                    dim=array([10, 10, 10]),
                    rot=array([0, 0, 0]))
        result = _slh.create_sim_object(geom=geom, shape='B',
                                         obj_material='water', label=7)
        assert result['shape'] == 'B'
        assert result['material'] == 'water'

    def test_liquid_flag(self):
        geom = dict(center=array([0, 0, 0]),
                    dim=array([10, 10, 10]),
                    rot=array([0, 0, 0]))
        lqd = dict(lqd_level=0.8, lqd_material='water',
                   cntr_thickness=2, lqd_label=50)
        result = _slh.create_sim_object(geom=geom, shape='B',
                                         obj_material='pyrex', label=10,
                                         lqd_flag=True, lqd_param=lqd)
        assert result['lqd_flag'] == True
        assert result['lqd_param']['lqd_material'] == 'water'
        assert result['lqd_param']['lqd_level'] == 0.8

    def test_custom_shape_dict(self):
        mask = np.ones((5, 5, 5), dtype=bool)
        geom = dict(center=array([0, 0, 0]),
                    dim=array([5, 5, 5], dtype=float),
                    rot=array([0, 0, 0]),
                    scale=1.0,
                    src='test.stl',
                    mask=mask)
        result = _slh.create_sim_object(geom=geom, shape='M',
                                         obj_material='Al', label=1)
        assert result['shape'] == 'M'
        assert result['geom']['src'] == 'test.stl'


# ===========================================================================
# Object3D pose and metadata
# ===========================================================================

class TestObject3DMetadata:

    def test_pose_is_stored(self):
        geom = dict(center=array([0, 0, 0]),
                    dim=array([5, 5, 5]),
                    rot=array([0, 0, 0]))
        obj = _make_object3d('B', geom, pose=array([100, 200, 300]))
        np.testing.assert_array_equal(obj.pose, [100, 200, 300])

    def test_material_stored(self):
        geom = dict(center=array([0, 0, 0]),
                    axes=array([5, 5, 5]),
                    rot=array([0, 0, 0]))
        obj_dict = _slh.create_sim_object(geom=geom, shape='E',
                                           obj_material='Cu', label=3)
        obj = Object3D(obj_dict=obj_dict, obj_pose=array([0, 0, 0]))
        assert obj.obj_dict['material'] == 'Cu'

    def test_label_stored(self):
        geom = dict(center=array([0, 0, 0]),
                    dim=array([5, 5, 5]),
                    rot=array([0, 0, 0]))
        obj_dict = _slh.create_sim_object(geom=geom, shape='B',
                                           obj_material='air', label=77)
        obj = Object3D(obj_dict=obj_dict, obj_pose=array([0, 0, 0]))
        assert obj.obj_dict['label'] == 77

    def test_shape_type_stored(self):
        geom = dict(center=array([0, 0, 0]),
                    axes=array([5, 5, 5]),
                    rot=array([0, 0, 0]))
        obj = _make_object3d('E', geom)
        assert obj.shape == 'E'
