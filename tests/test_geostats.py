import pytest
import numpy as np
import pandas as pd
from geostatspy.geostats import backtr, gcum, declus


def test_backtr():
    # Setup
    df = pd.DataFrame({'values': [0.1, 0.5, 0.9]})
    vr = np.array([0, 0.3, 0.6, 1.0])
    vrg = np.array([0, 0.3, 0.7, 1.0])
    zmin, zmax = 0, 1.0
    ltail, ltpar, utail, utpar = 1, 1.5, 1, 1.5

    # Expected output
    expected_backtr = np.array([0.1, 0.5, 0.9])  # Simplified expected result for demonstration

    # Test
    result = backtr(df, 'values', vr, vrg, zmin, zmax, ltail, ltpar, utail, utpar)
    assert np.allclose(result, expected_backtr, atol=0.1), "Back transformation did not match expected values"


def test_gcum():
    # Test value
    x_values = np.array([-1.96, 0, 1.96])
    # Expected probabilities for these z-scores
    expected_probs = np.array([0.025, 0.5, 0.975])

    # Test
    results = np.array([gcum(x) for x in x_values])
    assert np.allclose(results, expected_probs, atol=0.01), "Cumulative probabilities did not match expected values"


def test_declus():
    df = pd.read_csv('tests/test_data/test_declus.csv')

    ncell = 100
    cmin, cmax = 1, 5000
    noff = 10
    iminmax = 1

    wts, cell_sizes, dmeans = declus(df, 'X', 'Y', 'Por', iminmax=iminmax, noff=noff,
                                     ncell=ncell, cmin=cmin, cmax=cmax, verbose=False)

    correct_wts = np.array([0.35, 0.27, 0.92, 2.06,  0.27])
    correct_dmeans = np.array([18.38, 18.38, 17.14, 16.29, 16.05])

    assert np.allclose(wts[:5], correct_wts, atol=0.1), "Declus weights did not match expected values"
    assert np.allclose(dmeans[:5], correct_dmeans, atol=0.1), "Declus declustered means did not match expected values"
