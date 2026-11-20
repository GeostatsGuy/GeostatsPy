"""Coordinate-level regression checks for 2D SGS hard conditioning (#53)."""

import numpy as np
import pandas as pd
import pytest

from geostatspy import geostats


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def simulate(df, nx=3, ny=3, xmn=0.0, ymn=0.0, xsiz=1.0, ysiz=1.0,
             seed=73073, nsim=1):
    vario = dict(nug=0.0, nst=1, it1=1, cc1=1.0, azi1=0.0,
                 hmaj1=3.0, hmin1=3.0, it2=1, cc2=0.0, azi2=0.0,
                 hmaj2=0.0, hmin2=0.0)
    return geostats.sgsim(
        df, "X", "Y", "Value", wcol=-1, scol=-1, tmin=-1e21, tmax=1e21,
        itrans=0, ismooth=0, dftrans=0, tcol=0, twtcol=0,
        zmin=-5.0, zmax=5.0, ltail=1, ltpar=-5.0, utail=1, utpar=5.0,
        nsim=nsim, nx=nx, xmn=xmn, xsiz=xsiz, ny=ny, ymn=ymn, ysiz=ysiz,
        seed=seed, ndmin=0, ndmax=1, nodmax=8, mults=0, nmult=2, noct=-1,
        ktype=0, colocorr=0.0, sec_map=0, vario=vario,
    )


def test_issue_53_single_center_datum():
    df = pd.DataFrame({"X": [1.0], "Y": [1.0], "Value": [0.25]})
    result = simulate(df)
    assert result.shape == (1, 3, 3)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) < 10  # No UNEST/conditioning flags remain.
    assert result[0, 1, 1] == 0.25


@pytest.mark.parametrize("ix,iy", [(0, 0), (4, 0), (0, 2), (4, 2), (2, 1)])
def test_rectangular_grid_datum_at_correct_coordinate(ix, iy):
    # Non-square grid and nonzero origin expose row-stride and row-wrap errors.
    df = pd.DataFrame({"X": [10.0 + ix], "Y": [-4.0 + 2 * iy],
                       "Value": [0.375]})
    result = simulate(df, nx=5, ny=3, xmn=10.0, ymn=-4.0, ysiz=2.0)
    assert result.shape == (1, 3, 5)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) < 10
    # Public array rows run from highest to lowest y.
    assert result[0, 2 - iy, ix] == 0.375


def test_fully_conditioned_rectangular_grid_preserves_every_datum():
    rows = [(10.0 + ix, -4.0 + 2 * iy, (5 * iy + ix) / 20.0)
            for iy in range(3) for ix in range(5)]
    df = pd.DataFrame(rows, columns=["X", "Y", "Value"])
    result = simulate(df, nx=5, ny=3, xmn=10.0, ymn=-4.0, ysiz=2.0, nsim=2)
    expected = np.array([[0.50, 0.55, 0.60, 0.65, 0.70],
                         [0.25, 0.30, 0.35, 0.40, 0.45],
                         [0.00, 0.05, 0.10, 0.15, 0.20]])
    np.testing.assert_array_equal(result, np.stack([expected, expected]))


def test_seed_replay_and_multiple_realizations():
    df = pd.DataFrame({"X": [1.0], "Y": [1.0], "Value": [0.25]})
    first = simulate(df, nsim=2)
    np.testing.assert_array_equal(first, simulate(df, nsim=2))
    assert not np.array_equal(first[0], first[1])
    assert not np.array_equal(first, simulate(df, nsim=2, seed=73074))
    np.testing.assert_array_equal(first[:, 1, 1], [0.25, 0.25])
