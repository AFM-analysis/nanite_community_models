import numpy as np
import pytest


def old_helper_retract_fraction(tip_position, force, fraction_force):
    fraction = int(0.8 * len(force))
    max_force = np.max(force[:fraction])
    max_position = -1 * tip_position[0]
    fit_stop = int(np.argwhere(force < (max_force * fraction_force))[0])
    position_seg = tip_position[:fit_stop].copy()
    force_seg = force[:fit_stop].copy()
    return position_seg, force_seg, max_position, max_force


def new_helper_retract_fraction(tip_position, force, fraction_force):
    """Reference implementation using NumPy 1.x and 2.x compatible indexing."""
    fraction = int(0.8 * len(force))
    max_force = np.max(force[:fraction])
    max_position = -1 * tip_position[0]

    # new part
    _mask = force < (max_force * fraction_force)
    if not np.any(_mask):
        raise ValueError("No force values below threshold")
    fit_stop = int(np.argmax(_mask))

    position_seg = tip_position[:fit_stop].copy()
    force_seg = force[:fit_stop].copy()
    return position_seg, force_seg, max_position, max_force


def test_helper_behavior_differs_between_numpy_1_and_2():
    # Deterministic example with a clear cutoff.
    tip_position = np.linspace(0.0, -9.0, 10)
    force = np.array([10.0, 9.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0])
    fraction_force = 0.5  # threshold = 5.0 -> first below threshold at index 4

    major = int(np.__version__.split(".", 1)[0])
    if major < 2:
        old = old_helper_retract_fraction(tip_position, force, fraction_force)
        new = new_helper_retract_fraction(tip_position, force, fraction_force)
        np.testing.assert_allclose(old[0], new[0])
        np.testing.assert_allclose(old[1], new[1])
        assert old[2] == new[2]
        assert old[3] == new[3]
    else:
        # Current implementation creates errors on NumPy 2.x
        # ("only length-1 arrays..." / "only 0-d arrays...").
        with pytest.raises(TypeError):
            new_helper_retract_fraction(tip_position, force, fraction_force)
