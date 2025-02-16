import pytest
import os
import pandas as pd
from geostatspy.GSLIB import kb2d, sgsim


def test_kb2d():
    # Create dummy data
    data = {
        'xcol': [0, 1, 2, 3, 4],
        'ycol': [0, 1, 2, 3, 4],
        'varcol': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    df = pd.DataFrame(data)

    # Variogram model parameters
    var_model = {
        'nug': 0.0,
        'nst': 2,
        'hmaj1': 600,
        'hmaj2': 2300,
        'hmin1': 500,
        'hmin2': 2200,
        'azi1': 5,
        'azi2': 10,
        'it1': 2,
        'it2': 3,
        'cc1': 0.08,
        'cc2': 0.22
    }

    # Setup kb2d parameters
    nx = 10
    ny = 10
    hsiz = 1.0

    # Call kb2d with dry_run=True to generate parameter file without executing kriging
    kb2d(df, 'xcol', 'ycol', 'varcol', nx, ny, hsiz, var_model, dry_run=True)

    # Read back the parameters from the file
    params = {}
    with open('kb2d.par', 'r') as file:
        for i, line in enumerate(file):
            split_line = line.strip().split()
            if i == 9:
                assert split_line[0] == 'tmp.dat', "Incorrect data file in kb2d.par"
            elif i == 10:
                assert int(split_line[0]) == nx, "Incorrect nx in kb2d.par"
                assert float(split_line[1]) == 0.5, "Incorrect xmn in kb2d.par"
                assert float(split_line[2]) == hsiz, "Incorrect xsiz in kb2d.par"
            elif i == 11:
                assert int(split_line[0]) == ny, "Incorrect ny in kb2d.par"
                assert float(split_line[1]) == 0.5, "Incorrect ymn in kb2d.par"
                assert float(split_line[2]) == hsiz, "Incorrect ysiz in kb2d.par"
            elif i == 13:
                assert int(split_line[0]) == 1, "Incorrect ndmin in kb2d.par"
                assert int(split_line[1]) == 30, "Incorrect ndmax in kb2d.par"
            elif i == 14:
                assert float(split_line[0]) == 2300, "Incorrect max_radius in kb2d.par"
            elif i == 16:
                params['nst'] = int(split_line[0])
                params['nug'] = float(split_line[1])
            elif i == 17:
                params['it1'] = int(split_line[0])
                params['cc1'] = float(split_line[1])
                params['azi1'] = float(split_line[2])
                params['hmaj1'] = float(split_line[3])
                params['hmin1'] = float(split_line[4])
            elif i == 18:
                params['it2'] = int(split_line[0])
                params['cc2'] = float(split_line[1])
                params['azi2'] = float(split_line[2])
                params['hmaj2'] = float(split_line[3])
                params['hmin2'] = float(split_line[4])

    for key, value in var_model.items():
        assert params[key] == value, f"Incorrect {key} in variogram model in kb2d.par"

    # Clean up temporary files
    os.remove('kb2d.par')
    os.remove('data_temp.dat')


def test_sgsim():
    # Create dummy data
    data = {
        'xcol': [0, 1, 2, 3, 4],
        'zcol': [0, 1, 2, 3, 4],
        'ycol': [0, 1, 2, 3, 4],
        'wtcol': [1, 1, 1, 1, 1],
        'varcol': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    df = pd.DataFrame(data)

    # Variogram model parameters for sgsim
    var_model = {
        'nug': 0.05,
        'nst': 2,
        'c': [0.5, 0.45],
        'tstr': [1, 2],
        'ahmax': [2200, 600],
        'ahmin': [2100, 500],
        'ahvert': [400, 50],
        'ang1': [5, 8],
        'ang2': [80, 83],
        'ang3': [30, 33],
    }

    # Setup sgsim parameters
    nx, ny, nz = 10, 10, 10
    dx, dy, dz = 0.5, 0.5, 0.5

    # Call sgsim with dry_run=True to generate sgsim.par
    sgsim(1, df, 'varcol', var_model,
          xcol='xcol', ycol='ycol', zcol='zcol', wtcol='wtcol',
          nx=nx, ny=ny, nz=30, dx=dx, dy=dy, dz=dz,
          ndmin=3, ndmax=30, dry_run=True)

    # Read back the parameters from the file
    params = {}
    with open('sgsim.par', 'r') as file:
        for i, line in enumerate(file):
            split_line = line.strip().split()
            if i == 5:
                assert int(split_line[0]) == 2, "Incorrect xcol in sgsim.par"
                assert int(split_line[1]) == 3, "Incorrect ycol in sgsim.par"
                assert int(split_line[2]) == 4, "Incorrect zcol in sgsim.par"
                assert int(split_line[3]) == 1, "Incorrect vcol in sgsim.par"
                assert int(split_line[4]) == 5, "Incorrect wtcol in sgsim.par"
            elif i == 12:
                assert float(split_line[0]) == 0.1, "Incorrect lower tail in sgsim.par"
                assert float(split_line[1]) == 0.5, "Incorrect upper tail in sgsim.par"
            elif i == 18:
                assert int(split_line[0]) == 1, "Incorrect nsim in sgsim.par"
            elif i == 19:
                assert int(split_line[0]) == 10, "Incorrect nx in sgsim.par"
                assert float(split_line[1]) == 0.25, "Incorrect xmin in sgsim.par"
                assert float(split_line[2]) == 0.5, "Incorrect dx in sgsim.par"
            elif i == 20:
                assert int(split_line[0]) == 10, "Incorrect ny in sgsim.par"
                assert float(split_line[1]) == 0.25, "Incorrect ymin in sgsim.par"
                assert float(split_line[2]) == 0.5, "Incorrect dy in sgsim.par"
            elif i == 21:
                assert int(split_line[0]) == 30, "Incorrect nz in sgsim.par"
                assert float(split_line[1]) == 0.25, "Incorrect zmin in sgsim.par"
                assert float(split_line[2]) == 0.5, "Incorrect dz in sgsim.par"
            elif i == 23:
                assert int(split_line[0]) == 3, "Incorrect ndmin in sgsim.par"
                assert int(split_line[1]) == 30, "Incorrect ndmax in sgsim.par"
            elif i == 24:
                assert int(split_line[0]) == 12, "Incorrect # of simulated nodes in sgsim.par"
            elif i == 28:
                assert float(split_line[0]) == var_model['ahmax'][0], "Incorrect search radius"
                assert float(split_line[1]) == var_model['ahmin'][0], "Incorrect search radius"
                assert float(split_line[2]) == var_model['ahvert'][0], "Incorrect search radius"
            elif i == 29:
                assert float(split_line[0]) == var_model['ang1'][0], "Incorrect search ellipsoid"
                assert float(split_line[1]) == var_model['ang2'][0], "Incorrect search ellipsoid"
                assert float(split_line[2]) == var_model['ang3'][0], "Incorrect search ellipsoid"
            elif i == 34:
                assert int(split_line[0]) == 2, "Incorrect nst effect in sgsim.par"
                assert float(split_line[1]) == var_model['nug'], "Incorrect nugget in sgsim.par"
            elif i == 35:
                assert int(split_line[0]) == 1, "Incorrect it1 in sgsim.par"
                assert float(split_line[1]) == var_model['c'][0], "Incorrect cc1 in sgsim.par"
                assert float(split_line[2]) == var_model['ang1'][0], "Incorrect ang1 in sgsim.par"
                assert float(split_line[3]) == var_model['ang2'][0], "Incorrect ang2 in sgsim.par"
                assert float(split_line[4]) == var_model['ang3'][0], "Incorrect ang3 in sgsim.par"
            elif i == 36:
                assert float(split_line[0]) == var_model['ahmax'][0], 'Incorrect ahmax in sgsim.par'
                assert float(split_line[1]) == var_model['ahmin'][0], 'Incorrect ahmin in sgsim.par'
                assert float(split_line[2]) == var_model['ahvert'][0], 'Incorrect ahvert in sgsim.par'
            elif i == 37:
                assert int(split_line[0]) == 2, "Incorrect it2 in sgsim.par"
                assert float(split_line[1]) == var_model['c'][1], "Incorrect cc2 in sgsim.par"
                assert float(split_line[2]) == var_model['ang1'][1], "Incorrect ang1 in sgsim.par"
                assert float(split_line[3]) == var_model['ang2'][1], "Incorrect ang2 in sgsim.par"
                assert float(split_line[4]) == var_model['ang3'][1], "Incorrect ang3 in sgsim.par"
            elif i == 38:
                assert float(split_line[0]) == var_model['ahmax'][1], 'Incorrect ahmax in sgsim.par'
                assert float(split_line[1]) == var_model['ahmin'][1], 'Incorrect ahmin in sgsim.par'
                assert float(split_line[2]) == var_model['ahvert'][1], 'Incorrect ahvert in sgsim.par'

    # Clean up temporary files
    os.remove('sgsim.par')
    os.remove('data_temp.dat')