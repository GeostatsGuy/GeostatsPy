"""
This file includes the reimplementations of GSLIB functionality in Python. While
this code will not be as well-tested and robust as the original GSLIB, it does
provide the opportunity to build 2D spatial modeling projects in Python without
the need to rely on compiled Fortran code from GSLIB. If you want to use the
GSLIB compiled code called from Python workflows use the functions available
with geostatspy.GSLIB.

This file includes the (1) GSLIB subroutines (converted to Python), followed by
the (2) functions: declus, gam, gamv, nscore, kb2d (more added all the time)
Note: some GSLIB subroutines are not included as they were replaced by available
NumPy and SciPy functionality or they were not required as we don't have to deal
with graphics and files in the same manner as GSLIB.

The original GSLIB code is from GSLIB: Geostatistical Library by Deutsch and
Journel, 1998. The reimplementation is by Michael Pyrcz, Associate Professor,
the University of Texas at Austin.
"""

import math  # for trig functions etc.
from bisect import bisect  # for maintaining array elements sorted

import numpy as np  # for ndarrays
import numpy.linalg as linalg  # for linear algebra
import scipy.spatial as sp  # for fast nearest neighbor search
from numba import jit  # for numerical speed up

def backtr(df,vcol,vr,vrg,zmin,zmax,ltail,ltpar,utail,utpar):   
    """Back transform an entire DataFrame column with a provided transformation table and tail extrapolation.

    :param df: the source DataFrame
    :param vcol: the column with the variable to transfrom
    :param vr: the transformation table, 1D ndarray with the original values
    :param vrg: the transformation table, 1D ndarray with the trasnformed variable
    :param zmin: lower trimming limits
    :param zmax: upper trimming limits
    :param ltail: lower tail value
    :param ltpar: lower tail extrapolation parameter
    :param utail: upper tail value
    :param utpar: upper tail extrapolation parameter
    :return: TODO
    """    
    
    EPSLON=1.0e-20
    nd = len(df); nt = len(vr) # number of data to transform and number of data in table
    backtr = np.zeros(nd)
    vrgs = df[vcol].values
    # Value in the lower tail?    1=linear, 2=power, (3 and 4 are invalid):
    for id in range(0,nd):
        if vrgs[id] <= vrg[0]:
            backtr[id] = vr[0]
            cdflo  = gcum(vrg[0])
            cdfbt  = gcum(vrgs[id])
            if ltail == 1:
                backtr[id] = powint(0.0,cdflo,zmin,vr[0],cdfbt,1.0)
            elif ltail == 2:
                cpow   = 1.0 / ltpar
                backtr[id] = powint(0.0,cdflo,zmin,vr[0],cdfbt,cpow)
        # Value in the upper tail?     1=linear, 2=power, 4=hyperbolic:
        elif vrgs[id] >= vrg[nt-1]:
            backtr[id] = vr[nt-1]
            cdfhi  = gcum(vrg[nt-1])
            cdfbt  = gcum(vrgs[id])
            if utail == 1:
                backtr[id] = powint(cdfhi,1.0,vr[nt-1],zmax,cdfbt,1.0)
            elif utail == 2: 
                cpow   = 1.0 / utpar
                backtr[id] = powint(cdfhi,1.0,vr[nt-1],zmax,cdfbt,cpow)
            elif utail == 4:
                plambda = (vr[nt-1]**utpar)*(1.0-gcum(vrg[nt-1]))
                backtr[id] = (plambda/(1.0-gcum(vrgs)))**(1.0/utpar)
        else:
            # Value within the transformation table:
            j = locate(vrg,1,nt,vrgs[id])
            j = max(min((nt-2),j),1)
            backtr[id] = powint(vrg[j],vrg[j+1],vr[j],vr[j+1],vrgs[id],1.0)
    return backtr

def gcum(x):
    """Calculate the cumulative probability of the standard normal distribution.

    :param x: the value from the standard normal distribution 
    :return: TODO
    """    
    
    z = x
    if z < 0: 
        z = -z
    t  = 1./(1.+ 0.2316419*z)
    gcum = t*(0.31938153   + t*(-0.356563782 + t*(1.781477937 + t*(-1.821255978 + t*1.330274429))))
    e2   = 0.
    #  6 standard deviations out gets treated as infinity:
    if z <= 6.: 
        e2 = np.exp(-z*z/2.)*0.3989422803
    gcum = 1.0- e2 * gcum
    if x >= 0:
        return gcum
    gcum = 1.0 - gcum
    return gcum

def locate(xx,iis,iie,x):
    """Return value `j` such that `x` is between `xx[j]` and `xx[j+1]`, where
    `xx` is an array of length `n`, and `x` is a given value. `xx` must be
    monotonic, either increasing or decreasing (GSLIB version).

    :param xx: array
    :param iis: start point
    :param iie: end point
    :param x: given value
    :return: TODO
    """
    
    n = len(xx)
# Initialize lower and upper methods:
    if iis <= 0: 
        iis = 0
    if iie >= n:
        iie = n-1
    jl = iis-1
    ju = iie
    if xx[n-1] <= x:
        j = iie
        return j
# If we are not done then compute a midpoint:
    while (ju-jl) > 1: 
        jm = int((ju+jl)/2)
# Replace the lower or upper limit with the midpoint:
        if (xx[iie] > xx[iis]) == (x > xx[jm]):
            jl = jm
        else:
            ju = jm
# Return with the array index:
    j = jl
    return j

def dlocate(xx, iis, iie, x):
    """Return value `j` such that `x` is between `xx[j]` and `xx[j+1]`, where
    `xx` is an array of length `n`, and `x` is a given value. `xx` must be
    monotonic, either increasing or decreasing (updated with Python bisect)

    :param xx: array
    :param iis: start point
    :param iie: end point
    :param x: given value
    :return: TODO
    """
    n = len(xx)
    if iie <= iis:
        iis = 0
        iie = n - 1
    array = xx[iis: iie - 1]  # this is accounting for swith to 0,...,n-1 index
    j = bisect(array, x)
    return j

def powint(xlow,xhigh,ylow,yhigh,xval,power):
    """Power-based interpolator 

    :param xlow: x lower interval
    :param xhigh: x upper interval
    :param ylow: y lower interval
    :param yhigh: y upper interval
    :param xval: value on x
    :param power: power for interpolation
    :return: TODO
    """    
    EPSLON=1.0e-20
    if (xhigh-xlow) < EPSLON:
        powint = (yhigh+ylow)/2.0
    else:
        powint = ylow + (yhigh-ylow)*(((xval-xlow)/(xhigh-xlow))**power)
    return powint


def dsortem(ib, ie, a, iperm, b=0, c=0, d=0, e=0, f=0, g=0, h=0):
    """Sort array in ascending order.

    :param ib: start index
    :param ie: end index
    :param a: array
    :param iperm: 0 no other array is permuted.
                  1 array b is permuted according to array a.
                  2 arrays b, c are permuted.
                  3 arrays b, c, d are permuted.
                  4 arrays b, c, d, e are permuted.
                  5 arrays b, c, d, e, f are permuted.
                  6 arrays b, c, d, e, f, g are permuted.
                  7 arrays b, c, d, e, f, g, h are permuted.
                 >7 no other array is permuted.
    :param b: array to be permuted according to array a.
    :param c: array to be permuted according to array a.
    :param d: array to be permuted according to array a.
    :param e: array to be permuted according to array a.
    :param f: array to be permuted according to array a.
    :param g: array to be permuted according to array a.
    :param h: array to be permuted according to array a.
    :return: a: the array, a portion of which has been sorted.
             b, c, d, e, f, g, h: arrays permuted according to array a (see
             iperm)
    """
    a = a[ib:ie]
    inds = a.argsort()
    a = np.copy(a[inds])  # deepcopy forces pass to outside scope
    if iperm == 1:
        return a
    b_slice = b[ib:ie]
    b = b_slice[inds]
    if iperm == 2:
        return a, b
    c_slice = c[ib:ie]
    c = c_slice[inds]
    if iperm == 3:
        return a, b, c
    d_slice = d[ib:ie]
    d = d_slice[inds]
    if iperm == 4:
        return a, b, c, d
    e_slice = e[ib:ie]
    e = e_slice[inds]
    if iperm == 5:
        return a, b, c, d, e
    f_slice = f[ib:ie]
    f = f_slice[inds]
    if iperm == 6:
        return a, b, c, d, e, f
    g_slice = g[ib:ie]
    g = g_slice[inds]
    if iperm == 7:
        return a, b, c, d, e, f, g  # TODO: changed from 'a, b, c, d, e, f, h'
    h_slice = h[ib:ie]
    h = h_slice[inds]
    return a, b, c, d, e, f, g, h  # TODO: changed from 'a, b, c, d, e, f, h'


def gauinv(p):
    """Compute the inverse of the standard normal cumulative distribution
    function.

    :param p: cumulative probability value
    :return: TODO
    """
    lim = 1.0e-10
    p0 = -0.322_232_431_088
    p1 = -1.0
    p2 = -0.342_242_088_547
    p3 = -0.020_423_121_024_5
    p4 = -0.000_045_364_221_014_8
    q0 = 0.099_348_462_606_0
    q1 = 0.588_581_570_495
    q2 = 0.531_103_462_366
    q3 = 0.103_537_752_850
    q4 = 0.003_856_070_063_4

    # Check for an error situation
    if p < lim:
        xp = -1.0e10
        return xp
    if p > (1.0 - lim):
        xp = 1.0e10
        return xp

    # Get k for an error situation
    pp = p
    if p > 0.5:
        pp = 1 - pp
    xp = 0.0
    if p == 0.5:
        return xp

    # Approximate the function
    y = np.sqrt(np.log(1.0 / (pp * pp)))
    xp = float(
        y
        + ((((y * p4 + p3) * y + p2) * y + p1) * y + p0)
        / ((((y * q4 + q3) * y + q2) * y + q1) * y + q0)
    )
    if float(p) == float(pp):
        xp = -xp
    return xp


def gcum(x):
    """Evaluate the standard normal cdf given a normal deviate `x`. `gcum` is
    the area under a unit normal curve to the left of `x`. The results are
    accurate only to about 5 decimal places.

    :param x: TODO
    :return: TODO
    """
    z = x
    if z < 0:
        z = -z
    t = 1.0 / (1.0 + 0.231_641_9 * z)
    gcum_ = t * (
        0.319_381_53
        + t
        * (
            -0.356_563_782
            + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))
        )
    )
    e2 = 0.0
    # Standard deviations out gets treated as infinity
    if z <= 6:
        e2 = np.exp(-z * z / 2.0) * 0.398_942_280_3
    gcum_ = 1.0 - e2 * gcum_
    if x >= 0.0:
        return gcum_
    gcum_ = 1.0 - gcum_
    return gcum_


def dpowint(xlow, xhigh, ylow, yhigh, xval, pwr):
    """Power interpolate the value of `y` between (`xlow`, `ylow`) and
    (`xhigh`, `yhigh`) for a value of `x` and a power `pwr`.

    :param xlow: TODO
    :param xhigh: TODO
    :param ylow: TODO
    :param yhigh: TODO
    :param xval: TODO
    :param pwr: power
    :return: TODO
    """
    EPSLON = 1.0e-20
    if (xhigh - xlow) < EPSLON:
        dpowint_ = (yhigh + ylow) / 2.0
    else:
        dpowint_ = ylow + (yhigh - ylow) * (
            ((xval - xlow) / (xhigh - xlow)) ** pwr
        )
    return dpowint_


@jit(nopython=True)
def setup_rotmat(c0, nst, it, cc, ang, pmx):
    """Setup rotation matrix.

    :param c0: nugget constant (isotropic)
    :param nst: number of nested structures (max. 4)
    :param it: TODO
    :param cc: multiplicative factor of each nested structure
    :param ang: TODO
    :param pmx: TODO
    :return: TODO
    """
    PI = 3.141_592_65
    DTOR = PI / 180.0

    # The first time around, re-initialize the cosine matrix for the variogram
    # structures
    rotmat = np.zeros((4, nst))
    maxcov = c0
    for js in range(0, nst):
        azmuth = (90.0 - ang[js]) * DTOR
        rotmat[0, js] = math.cos(azmuth)
        rotmat[1, js] = math.sin(azmuth)
        rotmat[2, js] = -1 * math.sin(azmuth)
        rotmat[3, js] = math.cos(azmuth)
        if it[js] == 4:
            maxcov = maxcov + pmx
        else:
            maxcov = maxcov + cc[js]
    return rotmat, maxcov


@jit(nopython=True)
def cova2(x1, y1, x2, y2, nst, c0, pmx, cc, aa, it, ang, anis, rotmat, maxcov):
    """Calculate the covariance associated with a variogram model specified by a
    nugget effect and nested variogram structures.

    :param x1: x coordinate of first point
    :param y1: y coordinate of first point
    :param x2: x coordinate of second point
    :param y2: y coordinate of second point
    :param nst: number of nested structures (maximum of 4)
    :param c0: isotropic nugget constant (TODO: not used)
    :param pmx: TODO
    :param cc: multiplicative factor of each nested structure
    :param aa: parameter `a` of each nested structure
    :param it: TODO
    :param ang: TODO: not used
    :param anis: TODO
    :param rotmat: rotation matrices
    :param maxcov: TODO
    :return: TODO
    """
    EPSLON = 0.000001

    # Check for very small distance
    dx = x2 - x1
    dy = y2 - y1

    if (dx * dx + dy * dy) < EPSLON:
        cova2_ = maxcov
        return cova2_

    # Non-zero distance, loop over all the structures
    cova2_ = 0.0
    for js in range(0, nst):
        # Compute the appropriate structural distance
        dx1 = dx * rotmat[0, js] + dy * rotmat[1, js]
        dy1 = (dx * rotmat[2, js] + dy * rotmat[3, js]) / anis[js]
        h = math.sqrt(max((dx1 * dx1 + dy1 * dy1), 0.0))
        if it[js] == 1:
            # Spherical model
            hr = h / aa[js]
            if hr < 1.0:
                cova2_ = cova2_ + cc[js] * (1.0 - hr * (1.5 - 0.5 * hr * hr))
        elif it[js] == 2:
            # Exponential model
            cova2_ = cova2_ + cc[js] * np.exp(-3.0 * h / aa[js])
        elif it[js] == 3:
            # Gaussian model
            hh = -3.0 * (h * h) / (aa[js] * aa[js])
            cova2_ = cova2_ + cc[js] * np.exp(hh)
        elif it[js] == 4:
            # Power model
            cov1 = pmx - cc[js] * (h ** aa[js])
            cova2_ = cova2_ + cov1
    return cova2_


def ksol_numpy(neq, a, r):
    """Find solution of a system of linear equations.

    :param neq: number of equations
    :param a: upper triangular left hand side matrix
    :param r: right hand side matrix
    :return: solution array, same dimension as `r`
    """
    a = a[0: neq * neq]  # trim the array
    a = np.reshape(a, (neq, neq))  # reshape to 2D
    ainv = linalg.inv(a)  # invert matrix
    r = r[0: neq]  # trim the array
    s = np.matmul(ainv, r)  # matrix multiplication
    return s

def correct_trend(trend):
    """Correct a indicator based trend model for closure (probabilities sum to 1.0).

    :param trend: ndarray [ny,nx,ncut]
    :return: nadarray [ny,nx,ncut] corrected for closure
    """
    ny = trend.shape[0]
    nx = trend.shape[1]
    ncut = trend.shape[2]
    for iy in range(0,ny):
        for ix in range(0,nx):
            sum = 0.0
            for ic in range(0,ncut):
                sum = sum + trend[iy,ix,ic]
            if sum > 0.0:
                for icut in range(0,ncut):
                    trend[iy,ix,ic] = trend[iy,ix,ic] / sum
    return trend

def ordrel(ivtype,ncut,ccdf):
    """Correct a indicator based CDF for order relations.

    :param ivtype: variable type, 0 - categorical and 1 - continuous
    :param ncut: number of categories or thresholds
    :param ccdf: input cumulative distribution function
    :return: cumulative distribution function correct for order relations
    """
#    print('input ordering relations'); print(ccdf)
    ccdfo = np.zeros(ncut)
    ccdf1 = np.zeros(ncut)
    ccdf2 = np.zeros(ncut) # do we need MAXCUT = 100 for these 2?

# Make sure conditional cdf is within [0,1]:
    for i in range(0,ncut):
        if ccdf[i] < 0.0:
            ccdf1[i] = 0.0
            ccdf2[i] = 0.0
        elif ccdf[i] > 1.0:
            ccdf1[i] = 1.0
            ccdf2[i] = 1.0
        else:
            ccdf1[i] = ccdf[i]
            ccdf2[i] = ccdf[i]
#    print('ordering relations'); print(ccdf1,ccdf2)

# Correct sequentially up, then down, and then average:
    if ivtype == 0:
        sumcdf = 0.0
        for i in range(0,ncut):
            sumcdf = sumcdf + ccdf1[i]
        if sumcdf <= 0.0: sumcdf = 1.0
        for i in range(0,ncut):
            ccdfo[i] = ccdf1[i] / sumcdf
    else:
        for i in range(1,ncut):
            if ccdf1[i] < ccdf1[i-1]: ccdf1[i] = ccdf1[i-1]
        for i in range(ncut-2,0,-1):
            if ccdf2[i] > ccdf2[i+1]: ccdf2[i] = ccdf2[i+1]
        for i in range(0,ncut):
            ccdfo[i] = 0.5*(ccdf1[i]+ccdf2[i])

# Return with corrected CDF:
    return ccdfo

def declus(df, xcol, ycol, vcol, iminmax, noff, ncell, cmin, cmax):
    """GSLIB's DECLUS program (Deutsch and Journel, 1998) converted from the
    original Fortran to Python by Michael Pyrcz, the University of Texas at
    Austin (Jan, 2019).

    Note this was simplified to 2D only.

    :param df: pandas DataFrame with the spatial data
    :param xcol: name of the x coordinate column
    :param ycol: name of the y coordinate column
    :param vcol: name of the property column
    :param iminmax: 1 / True: for use cell size with max decluster mean
                    0 / False: for declustered mean minimizing cell size
    :param noff: number of offsets
    :param ncell: number of cell sizes
    :param cmin: min cell size
    :param cmax: max cell size
    :return: TODO
    """
    # Load data and set up arrays
    nd = len(df)
    x = df[xcol].values
    y = df[ycol].values
    v = df[vcol].values
    wt = np.zeros(nd)
    wtopt = np.ones(nd)
    index = np.zeros(nd, np.int32)
    xcs_mat = np.zeros(ncell + 2)  # we use 1,...,n for this array
    vrcr_mat = np.zeros(ncell + 2)  # we use 1,...,n for this array
    anisy = 1.0  # hard code the cells to 2D isotropic
    roff = float(noff)

    # Calculate extents
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    # Calculate summary statistics
    vmean = np.mean(v)
    vstdev = np.std(v)
    vmin = np.min(v)
    vmax = np.max(v)
    xcs_mat[0] = 0.0
    vrcr_mat[0] = vmean
    vrop = vmean  # include the naive case

    print(f"There are {nd} data with:")
    print(f"   mean of      {vmean} ")
    print(f"   min and max  {vmin} and {vmax}")
    print(f"   standard dev {vstdev} ")

    # Define a "lower" origin to use for the cell sizes
    xo1 = xmin - 0.01
    yo1 = ymin - 0.01

    # Define the increment for the cell size
    xinc = (cmax - cmin) / ncell
    yinc = xinc

    # Loop over "ncell+1" cell sizes in the grid network
    ncellx = int((xmax - (xo1 - cmin)) / cmin) + 1
    ncelly = int((ymax - (yo1 - cmin * anisy)) / cmin) + 1
    ncellt = ncellx * ncelly
    cellwt = np.zeros(ncellt)
    xcs = cmin - xinc
    ycs = (cmin * anisy) - yinc

    # Main loop over cell sizes
    # 0 index is the 0.0 cell, note n + 1 in Fortran
    for lp in range(1, ncell + 2):
        xcs = xcs + xinc
        ycs = ycs + yinc

        # Initialize the weights to zero
        wt.fill(0.0)

        # Determine the maximum number of grid cells in the network
        ncellx = int((xmax - (xo1 - xcs)) / xcs) + 1
        ncelly = int((ymax - (yo1 - ycs)) / ycs) + 1
        ncellt = float(ncellx * ncelly)  # TODO: not used

        # Loop over all the origin offsets selected
        xfac = min((xcs / roff), (0.5 * (xmax - xmin)))
        yfac = min((ycs / roff), (0.5 * (ymax - ymin)))
        for kp in range(1, noff + 1):
            xo = xo1 - (float(kp) - 1.0) * xfac
            yo = yo1 - (float(kp) - 1.0) * yfac

            # Initialize the cumulative weight indicators
            cellwt.fill(0.0)

            # Determine which cell each datum is in
            for i in range(0, nd):
                icellx = int((x[i] - xo) / xcs) + 1
                icelly = int((y[i] - yo) / ycs) + 1
                icell = icellx + (icelly - 1) * ncellx
                index[i] = icell
                cellwt[icell] = cellwt[icell] + 1.0

            # The weight assigned to each datum is inversely proportional to the
            # number of data in the cell. We first need to get the sum of
            # weights so that we can normalize the weights to sum to one
            sumw = 0.0
            for i in range(0, nd):
                ipoint = index[i]
                sumw = sumw + (1.0 / cellwt[ipoint])
            sumw = 1.0 / sumw

            # Accumulate the array of weights (that now sum to one)
            for i in range(0, nd):
                ipoint = index[i]
                wt[i] = wt[i] + (1.0 / cellwt[ipoint]) * sumw

        # End loop over all offsets

        # Compute the weighted average for this cell size
        sumw = 0.0
        sumwg = 0.0
        for i in range(0, nd):
            sumw = sumw + wt[i]
            sumwg = sumwg + wt[i] * v[i]
        vrcr = sumwg / sumw
        vrcr_mat[lp] = vrcr
        xcs_mat[lp] = xcs

        # See if this weighting is optimal
        if iminmax and vrcr < vrop or not iminmax and vrcr > vrop or ncell == 1:
            best = xcs  # TODO: not used
            vrop = vrcr
            wtopt = wt.copy()  # deep copy

    # End main loop over all cell sizes

    # Get the optimal weights
    sumw = 0.0
    for i in range(0, nd):
        sumw = sumw + wtopt[i]
    wtmin = np.min(wtopt)  # TODO: not used
    wtmax = np.max(wtopt)  # TODO: not used
    facto = float(nd) / sumw
    wtopt = wtopt * facto
    return wtopt, xcs_mat, vrcr_mat


def gam(array, tmin, tmax, xsiz, ysiz, ixd, iyd, nlag, isill):
    """GSLIB's GAM program (Deutsch and Journel, 1998) converted from the
    original Fortran to Python by Michael Pyrcz, the University of Texas at
    Austin (Jan, 2019).

    :param array: 2D gridded data / model
    :param tmin: property trimming limit
    :param tmax: property trimming limit
    :param xsiz: grid cell extents in x direction
    :param ysiz: grid cell extents in y direction
    :param ixd: lag offset in grid cells
    :param iyd: lag offset in grid cells
    :param nlag: number of lags to calculate
    :param isill: 1 for standardize sill
    :return: TODO
    """
    if array.ndim == 2:
        ny, nx = array.shape
    elif array.ndim == 1:
        ny, nx = 1, len(array)

    nvarg = 1  # for multiple variograms repeat the program
    nxy = nx * ny  # TODO: not used
    mxdlv = nlag

    # Allocate the needed memory
    lag = np.zeros(mxdlv)
    vario = np.zeros(mxdlv)
    hm = np.zeros(mxdlv)
    tm = np.zeros(mxdlv)
    hv = np.zeros(mxdlv)  # TODO: not used
    npp = np.zeros(mxdlv)
    ivtail = np.zeros(nvarg + 2)
    ivhead = np.zeros(nvarg + 2)
    ivtype = np.zeros(nvarg + 2)
    ivtail[0] = 0
    ivhead[0] = 0
    ivtype[0] = 0

    # Summary statistics for the data after trimming
    inside = (array > tmin) & (array < tmax)
    avg = array[(array > tmin) & (array < tmax)].mean()  # TODO: not used
    stdev = array[(array > tmin) & (array < tmax)].std()
    var = stdev ** 2.0
    vrmin = array[(array > tmin) & (array < tmax)].min()  # TODO: not used
    vrmax = array[(array > tmin) & (array < tmax)].max()  # TODO: not used
    num = ((array > tmin) & (array < tmax)).sum()  # TODO: not used

    # For the fixed seed point, loop through all directions
    for iy in range(0, ny):
        for ix in range(0, nx):
            if inside[iy, ix]:
                vrt = array[iy, ix]
                ixinc = ixd
                iyinc = iyd
                ix1 = ix
                iy1 = iy
                for il in range(0, nlag):
                    ix1 = ix1 + ixinc
                    if 0 <= ix1 < nx:
                        iy1 = iy1 + iyinc
                        if 1 <= iy1 < ny:
                            if inside[iy1, ix1]:
                                vrh = array[iy1, ix1]
                                npp[il] = npp[il] + 1
                                tm[il] = tm[il] + vrt
                                hm[il] = hm[il] + vrh
                                vario[il] = vario[il] + ((vrh - vrt) ** 2.0)

    # Get average values for gam, hm, tm, hv, and tv, then compute the correct
    # "variogram" measure
    for il in range(0, nlag):
        if npp[il] > 0:
            rnum = npp[il]
            lag[il] = np.sqrt((ixd * xsiz * il) ** 2 + (iyd * ysiz * il) ** 2)
            vario[il] = vario[il] / float(rnum)
            hm[il] = hm[il] / float(rnum)
            tm[il] = tm[il] / float(rnum)

            # Standardize by the sill
            if isill == 1:
                vario[il] = vario[il] / var

            # Semivariogram
            vario[il] = 0.5 * vario[il]
    return lag, vario, npp


def gamv(
    df,
    xcol,
    ycol,
    vcol,
    tmin,
    tmax,
    xlag,
    xltol,
    nlag,
    azm,
    atol,
    bandwh,
    isill,
):
    """GSLIB's GAMV program (Deutsch and Journel, 1998) converted from the
    original Fortran to Python by Michael Pyrcz, the University of Texas at
    Austin (Jan, 2019).

    Note simplified for 2D, semivariogram only and one direction at a time.

    :param df: pandas DataFrame with the spatial data
    :param xcol: name of the x coordinate column
    :param ycol: name of the y coordinate column
    :param vcol: name of the property column
    :param tmin: property trimming limit
    :param tmax: property trimming limit
    :param xlag: lag distance
    :param xltol: lag distance tolerance
    :param nlag: number of lags to calculate
    :param azm: azimuth
    :param atol: azimuth tolerance
    :param bandwh: horizontal bandwidth / maximum distance offset orthogonal to
                   azimuth
    :param isill: 1 for standardize sill
    :return: TODO
    """
    # Load the data
    # Trim values outside tmin and tmax
    df_extract = df.loc[(df[vcol] >= tmin) & (df[vcol] <= tmax)]
    nd = len(df_extract)  # TODO: not used
    x = df_extract[xcol].values
    y = df_extract[ycol].values
    vr = df_extract[vcol].values

    # Summary statistics for the data after trimming
    avg = vr.mean()  # TODO: not used
    stdev = vr.std()
    sills = stdev ** 2.0
    ssq = sills  # TODO: not used
    vrmin = vr.min()  # TODO: not used
    vrmax = vr.max()  # TODO: not used

    # Define the distance tolerance if it isn't already
    if xltol < 0.0:
        xltol = 0.5 * xlag

    # Loop over combinatorial of data pairs to calculate the variogram
    dis, vario, npp = variogram_loop(
        x, y, vr, xlag, xltol, nlag, azm, atol, bandwh
    )

    # Standardize sill to one by dividing all variogram values by the variance
    for il in range(0, nlag + 2):
        if isill == 1:
            vario[il] = vario[il] / sills

        # Apply 1/2 factor to go from variogram to semivariogram
        vario[il] = 0.5 * vario[il]

    return dis, vario, npp


@jit(nopython=True)
def variogram_loop(x, y, vr, xlag, xltol, nlag, azm, atol, bandwh):
    """Calculate the variogram by looping over combinatorial of data pairs.

    :param x: x values
    :param y: y values
    :param vr: property values
    :param xlag: lag distance
    :param xltol: lag distance tolerance
    :param nlag: number of lags to calculate
    :param azm: azimuth
    :param atol: azimuth tolerance
    :param bandwh: horizontal bandwidth / maximum distance offset orthogonal to
                   azimuth
    :return: TODO
    """
    # Allocate the needed memory
    nvarg = 1
    mxdlv = nlag + 2  # in gamv the npp etc. arrays go to nlag + 2
    dis = np.zeros(mxdlv)
    lag = np.zeros(mxdlv)  # TODO: not used
    vario = np.zeros(mxdlv)
    hm = np.zeros(mxdlv)
    tm = np.zeros(mxdlv)
    hv = np.zeros(mxdlv)  # TODO: not used
    npp = np.zeros(mxdlv)
    ivtail = np.zeros(nvarg + 2)
    ivhead = np.zeros(nvarg + 2)
    ivtype = np.ones(nvarg + 2)
    ivtail[0] = 0
    ivhead[0] = 0
    ivtype[0] = 0

    EPSLON = 1.0e-20
    nd = len(x)
    # The mathematical azimuth is measured counterclockwise from EW and
    # not clockwise from NS as the conventional azimuth is
    azmuth = (90.0 - azm) * math.pi / 180.0
    uvxazm = math.cos(azmuth)
    uvyazm = math.sin(azmuth)
    if atol <= 0.0:
        csatol = math.cos(45.0 * math.pi / 180.0)
    else:
        csatol = math.cos(atol * math.pi / 180.0)

    # Initialize the arrays for each direction, variogram, and lag
    nsiz = nlag + 2  # TODO: not used
    dismxs = ((float(nlag) + 0.5 - EPSLON) * xlag) ** 2

    # Main loop over all pairs
    for i in range(0, nd):
        for j in range(0, nd):
            
            # Definition of the lag corresponding to the current pair
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dxs = dx * dx
            dys = dy * dy
            hs = dxs + dys
            if hs <= dismxs:
                if hs < 0.0:
                    hs = 0.0
                h = np.sqrt(hs)

                # Determine which lag this is and skip if outside the defined
                # distance tolerance
                if h <= EPSLON:
                    lagbeg = 0
                    lagend = 0
                else:
                    lagbeg = -1
                    lagend = -1
                    for ilag in range(1, nlag + 1):
                        # reduced to -1
                        if (
                            (xlag * float(ilag - 1) - xltol)
                            <= h
                            <= (xlag * float(ilag - 1) + xltol)
                        ):
                            if lagbeg < 0:
                                lagbeg = ilag
                            lagend = ilag
                if lagend >= 0:
                    # Definition of the direction corresponding to the current
                    # pair. All directions are considered (overlapping of
                    # direction tolerance cones is allowed)

                    # Check for an acceptable azimuth angle
                    dxy = np.sqrt(max((dxs + dys), 0.0))
                    if dxy < EPSLON:
                        dcazm = 1.0
                    else:
                        dcazm = (dx * uvxazm + dy * uvyazm) / dxy

                    # Check the horizontal bandwidth criteria (maximum deviation
                    # perpendicular to the specified direction azimuth)
                    band = uvxazm * dy - uvyazm * dx

                    # Apply all the previous checks at once to avoid a lot of
                    # nested if statements
                    if (abs(dcazm) >= csatol) and (abs(band) <= bandwh):
                        # Check whether or not an omni-directional variogram is
                        # being computed
                        omni = False
                        if atol >= 90.0:
                            omni = True

                        # For this variogram, sort out which is the tail and
                        # the head value
                        iv = 0  # hardcoded just one variogram
                        it = ivtype[iv]  # TODO: not used
                        if dcazm >= 0.0:
                            vrh = vr[i]
                            vrt = vr[j]
                            if omni:
                                vrtpr = vr[i]
                                vrhpr = vr[j]
                        else:
                            vrh = vr[j]
                            vrt = vr[i]
                            if omni:
                                vrtpr = vr[j]
                                vrhpr = vr[i]

                        # Reject this pair on the basis of missing values

                        # Data was trimmed at the beginning

                        # The Semivariogram (all other types of measures are
                        # removed for now)
                        for il in range(lagbeg, lagend + 1):
                            npp[il] = npp[il] + 1
                            dis[il] = dis[il] + h
                            tm[il] = tm[il] + vrt
                            hm[il] = hm[il] + vrh
                            vario[il] = vario[il] + ((vrh - vrt) * (vrh - vrt))
                            if omni:
                                npp[il] = npp[il] + 1.0
                                dis[il] = dis[il] + h
                                tm[il] = tm[il] + vrtpr
                                hm[il] = hm[il] + vrhpr
                                vario[il] = vario[il] + (
                                    (vrhpr - vrtpr) * (vrhpr - vrtpr)
                                )

    # Get average values for gam, hm, tm, hv, and tv, then compute the correct
    # "variogram" measure
    for il in range(0, nlag + 2):
        i = il
        if npp[i] > 0:
            rnum = npp[i]
            dis[i] = dis[i] / rnum
            vario[i] = vario[i] / rnum
            hm[i] = hm[i] / rnum
            tm[i] = tm[i] / rnum

    return dis, vario, npp

def varmapv(df,xcol,ycol,vcol,tmin,tmax,nxlag,nylag,dxlag,dylag,minnp,isill):
    """Calculate the variogram map from irregularly spaced data.

    :param df: DataFrame with the spatial data, xcol, ycol, vcol coordinates and property columns
    :param xcol: DataFrame column with x coordinate
    :param ycol: DataFrame column with y coordinate
    :param vcol: DataFrame column with value of interest
    :param tmin: lower trimming limit
    :param tmax: upper trimming limit
    :param nxlag: number of lags in the x direction
    :param nxlag: number of lags in the y direction
    :param dxlag: size of the lags in the x direction
    :param dylag: size of the lags in the y direction
    :param minnp: minimum number of pairs to calculate a variogram value
    :param isill: standardize sill to be 1.0
    :return: TODO
    """
    # Load the data
    df_extract = df.loc[(df[vcol] >= tmin) & (df[vcol] <= tmax)]    # trim values outside tmin and tmax
    nd = len(df_extract)
    x = df_extract[xcol].values
    y = df_extract[ycol].values
    vr = df_extract[vcol].values  
    
    # Summary statistics for the data after trimming
    avg = vr.mean()
    stdev = vr.std()
    sills = stdev**2.0
    ssq = sills
    vrmin = vr.min()
    vrmax = vr.max() 
    
    # Initialize the summation arrays
    npp = np.zeros((nylag*2+1,nxlag*2+1))
    gam = np.zeros((nylag*2+1,nxlag*2+1))
    nppf = np.zeros((nylag*2+1,nxlag*2+1))
    gamf = np.zeros((nylag*2+1,nxlag*2+1))
    hm = np.zeros((nylag*2+1,nxlag*2+1))
    tm = np.zeros((nylag*2+1,nxlag*2+1))
    hv = np.zeros((nylag*2+1,nxlag*2+1))
    tv = np.zeros((nylag*2+1,nxlag*2+1))
    
    # First fix the location of a seed point: 
    for i in range(0,nd):     
        # Second loop over the data: 
        for j in range(0,nd): 
            # The lag:
            ydis = y[j] - y[i]
            iyl = nylag + int(ydis/dylag)
            if iyl < 0 or iyl > nylag*2: # acocunting for 0,...,n-1 array indexing
                continue
            xdis = x[j] - x[i]
            ixl = nxlag + int(xdis/dxlag)
            if ixl < 0 or ixl > nxlag*2: # acocunting for 0,...,n-1 array indexing
                continue             
            # We have an acceptable pair, therefore accumulate all the statistics
            # that are required for the variogram:
            npp[iyl,ixl] = npp[iyl,ixl] + 1 # our ndarrays read from the base to top, so we flip
            tm[iyl,ixl] = tm[iyl,ixl] + vr[i]
            hm[iyl,ixl] = hm[iyl,ixl] + vr[j]
            tv[iyl,ixl] = tm[iyl,ixl] + vr[i]*vr[i]
            hv[iyl,ixl] = hm[iyl,ixl] + vr[j]*vr[j]
            gam[iyl,ixl] = gam[iyl,ixl] + ((vr[i]-vr[j])*(vr[i]-vr[j]))
            
    # Get average values for gam, hm, tm, hv, and tv, then compute
    # the correct "variogram" measure:
    for iy in range(0,nylag*2+1): 
        for ix in range(0,nxlag*2+1): 
            if npp[iy,ix] <= minnp:
                gam[iy,ix] = -999.
                hm[iy,ix]  = -999.
                tm[iy,ix]  = -999.
                hv[iy,ix]  = -999.
                tv[iy,ix]  = -999.
            else:
                rnum = npp[iy,ix]
                gam[iy,ix] = gam[iy,ix] / (2*rnum) # semivariogram
                hm[iy,ix] = hm[iy,ix] / rnum
                tm[iy,ix] = tm[iy,ix] / rnum
                hv[iy,ix] = hv[iy,ix] / rnum - hm[iy,ix]*hm[iy,ix]
                tv[iy,ix] = tv[iy,ix] / rnum - tm[iy,ix]*tm[iy,ix]                
                # Attempt to standardize:
            if isill > 0:
                gamf[iy,ix] = gamf[iy,ix]/sills
    for iy in range(0,nylag*2+1): 
        for ix in range(0,nxlag*2+1):             
            gamf[iy,ix] = gam[nylag*2-iy,ix]
            nppf[iy,ix] = npp[nylag*2-iy,ix]        
    return gamf, nppf

def vmodel(
    nlag,
    xlag,
    azm,
    vario
):
    """GSLIB's VMODEL program (Deutsch and Journel, 1998) converted from the
    original Fortran to Python by Michael Pyrcz, the University of Texas at
    Austin (Mar, 2019).
    :param nlag: number of variogram lags 
    :param xlag: size of the lags
    :param axm: direction by 2D azimuth, 000 is y positive, 090 is x positive 
    :param vario: dictionary with the variogram parameters
    :return:
    """
    
# Parameters
    MAXNST=4
    DEG2RAD=3.14159265/180.0 
    MAXROT=MAXNST+1
    EPSLON = 1.0e-20
    VERSION= 1.01
  
# Declare arrays
    index = np.zeros(nlag+1)
    h = np.zeros(nlag+1)
    gam = np.zeros(nlag+1)
    cov = np.zeros(nlag+1)
    ro = np.zeros(nlag+1)
    
# Load the variogram
    nst = vario["nst"]
    cc = np.zeros(nst)
    aa = np.zeros(nst)
    it = np.zeros(nst)
    ang = np.zeros(nst)
    anis = np.zeros(nst)
    
    c0 = vario["nug"]
    cc[0] = vario["cc1"]
    it[0] = vario["it1"]
    ang[0] = vario["azi1"]
    aa[0] = vario["hmaj1"]
    anis[0] = vario["hmin1"] / vario["hmaj1"]
    if nst == 2:
        cc[1] = vario["cc2"]
        it[1] = vario["it2"]
        ang[1] = vario["azi2"]
        aa[1] = vario["hmaj2"]
        anis[1] = vario["hmin2"] / vario["hmaj2"]
                    
    xoff = math.sin(DEG2RAD*azm)*xlag
    yoff = math.cos(DEG2RAD*azm)*xlag
    print(' x,y,z offsets = ' + str(xoff) + ',' + str(yoff))
    rotmat, maxcov = geostats.setup_rotmat(c0, nst, it, cc, ang, 99999.9)   
          
    
    xx = 0.0; yy = 0.0      
    for il in range(0,nlag+1):
        index[il] = il
        cov[il] = geostats.cova2(0.0,0.0,xx,yy,nst,c0,9999.9,cc,aa,it,ang,anis,rotmat,maxcov)
        gam[il] = maxcov - cov[il]
        ro[il]  = cov[il]/maxcov
        h[il]   = math.sqrt(max((xx*xx+yy*yy),0.0))
        xx = xx + xoff
        yy = yy + yoff

# finished
    return index,h,gam,cov,ro

def nscore(
    df, vcol, wcol=None, ismooth=False, dfsmooth=None, smcol=0, smwcol=0
):
    """GSLIB's NSCORE program (Deutsch and Journel, 1998) converted from the
    original Fortran to Python by Michael Pyrcz, the University of Texas at
    Austin (Jan, 2019).

    :param df: pandas DataFrame with the spatial data
    :param vcol: name of the variable column
    :param wcol: name of the weight column, if None assumes equal weighting
    :param ismooth: if True then use a reference distribution
    :param dfsmooth: pandas DataFrame required if reference distribution is used
    :param smcol: reference distribution property (required if reference
                  distribution is used)
    :param smwcol: reference distribution weight (required if reference
                   distribution is used)
    :return: TODO
    """
    # Set constants
    np.random.seed(73073)
    pwr = 1.0  # interpolation power, hard coded to 1.0 in GSLIB
    EPSILON = 1.0e-20

    # Decide which file to use for establishing the transformation table
    if ismooth:
        nd = len(dfsmooth)
        vr = dfsmooth[smcol].values
        wt_ns = np.ones(nd)
        if smwcol != 0:
            wt_ns = dfsmooth[smwcol].values
    else:
        nd = len(df)
        vr = df[vcol].values
        wt_ns = np.ones(nd)
        if wcol is not None:
            wt_ns = df[wcol].values
    twt = np.sum(wt_ns)

    # Sort data by value
    istart = 0
    iend = nd
    vr, wt_ns = dsortem(istart, iend, vr, 2, wt_ns)

    # Compute the cumulative probabilities and write transformation table
    wtfac = 1.0 / twt
    oldcp = 0.0
    cp = 0.0
    for j in range(istart, iend):
        w = wtfac * wt_ns[j]
        cp = cp + w
        wt_ns[j] = (cp + oldcp) / 2.0
        vrrg = gauinv(wt_ns[j])
        vrg = float(vrrg)
        oldcp = cp

        # Now, reset the weight to the normal scores value
        wt_ns[j] = vrg

    # Normal scores transform
    nd_trans = len(df)
    ns = np.zeros(nd_trans)
    val = df[vcol].values
    for i in range(0, nd_trans):
        vrr = val[i] + np.random.rand() * EPSILON

        # Now, get the normal scores value for "vrr"
        j = dlocate(vr, 1, nd, vrr)
        j = min(max(1, j), (nd - 1))
        ns[i] = dpowint(vr[j], vr[j + 1], wt_ns[j], wt_ns[j + 1], vrr, pwr)

    return ns, vr, wt_ns


def kb2d(
    df,
    xcol,
    ycol,
    vcol,
    tmin,
    tmax,
    nx,
    xmn,
    xsiz,
    ny,
    ymn,
    ysiz,
    nxdis,
    nydis,
    ndmin,
    ndmax,
    radius,
    ktype,
    skmean,
    vario,
):
    """GSLIB's KB2D program (Deutsch and Journel, 1998) converted from the
    original Fortran to Python by Michael Pyrcz, the University of Texas at
    Austin (Jan, 2019).

    :param df: pandas DataFrame with the spatial data
    :param xcol: name of the x coordinate column
    :param ycol: name of the y coordinate column
    :param vcol: name of the property column
    :param tmin: property trimming limit
    :param tmax: property trimming limit
    :param nx: definition of the grid system (x axis)
    :param xmn: definition of the grid system (x axis)
    :param xsiz: definition of the grid system (x axis)
    :param ny: definition of the grid system (y axis)
    :param ymn: definition of the grid system (y axis)
    :param ysiz: definition of the grid system (y axis)
    :param nxdis: number of discretization points for a block
    :param nydis: number of discretization points for a block
    :param ndmin: minimum number of data points to use for kriging a block
    :param ndmax: maximum number of data points to use for kriging a block
    :param radius: maximum isotropic search radius
    :param ktype:
    :param skmean:
    :param vario:
    :return:
    """
    
# Constants
    UNEST = -999.
    EPSLON = 1.0e-10
    VERSION = 2.907
    first = True
    PMX = 9999.0    
    MAXSAM = ndmax + 1
    MAXDIS = nxdis * nydis
    MAXKD = MAXSAM + 1
    MAXKRG = MAXKD * MAXKD
    
# load the variogram
    nst = vario['nst']
    cc = np.zeros(nst); aa = np.zeros(nst); it = np.zeros(nst)
    ang = np.zeros(nst); anis = np.zeros(nst)
    
    c0 = vario['nug']; 
    cc[0] = vario['cc1']; it[0] = vario['it1']; ang[0] = vario['azi1']; 
    aa[0] = vario['hmaj1']; anis[0] = vario['hmin1']/vario['hmaj1'];
    if nst == 2:
        cc[1] = vario['cc2']; it[1] = vario['it2']; ang[1] = vario['azi2']; 
        aa[1] = vario['hmaj2']; anis[1] = vario['hmin2']/vario['hmaj2'];
    
# Allocate the needed memory:   
    xdb = np.zeros(MAXDIS)
    ydb = np.zeros(MAXDIS)
    xa = np.zeros(MAXSAM)
    ya = np.zeros(MAXSAM)
    vra = np.zeros(MAXSAM)
    dist = np.zeros(MAXSAM)
    nums = np.zeros(MAXSAM)
    r = np.zeros(MAXKD)
    rr = np.zeros(MAXKD)
    s = np.zeros(MAXKD)
    a = np.zeros(MAXKRG)
    kmap = np.zeros((nx,ny))
    vmap = np.zeros((nx,ny))

# Load the data
    df_extract = df.loc[(df[vcol] >= tmin) & (df[vcol] <= tmax)]    # trim values outside tmin and tmax
    nd = len(df_extract)
    ndmax = min(ndmax,nd)
    x = df_extract[xcol].values
    y = df_extract[ycol].values
    vr = df_extract[vcol].values
    
# Make a KDTree for fast search of nearest neighbours   
    dp = list((y[i], x[i]) for i in range(0,nd))
    data_locs = np.column_stack((y,x))
    tree = sp.cKDTree(data_locs, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True)

# Summary statistics for the data after trimming
    avg = vr.mean()
    stdev = vr.std()
    ss = stdev**2.0
    vrmin = vr.min()
    vrmax = vr.max()

# Set up the discretization points per block.  Figure out how many
# are needed, the spacing, and fill the xdb and ydb arrays with the
# offsets relative to the block center (this only gets done once):
    ndb  = nxdis * nydis
    if ndb > MAXDIS: 
        print('ERROR KB2D: Too many discretization points ')
        print('            Increase MAXDIS or lower n[xy]dis')
        return kmap
    xdis = xsiz  / max(float(nxdis),1.0)
    ydis = ysiz  / max(float(nydis),1.0)
    xloc = -0.5*(xsiz+xdis)
    i    = -1   # accounting for 0 as lowest index
    for ix in range(0,nxdis):       
        xloc = xloc + xdis
        yloc = -0.5*(ysiz+ydis)
        for iy in range(0,nydis): 
            yloc = yloc + ydis
            i = i+1
            xdb[i] = xloc
            ydb[i] = yloc

# Initialize accumulators:
    cbb  = 0.0
    rad2 = radius*radius

# Calculate Block Covariance. Check for point kriging.
    rotmat, maxcov = setup_rotmat(c0,nst,it,cc,ang,PMX)
    cov = cova2(xdb[0],ydb[0],xdb[0],ydb[0],nst,c0,PMX,cc,aa,it,ang,anis,rotmat,maxcov)
# Keep this value to use for the unbiasedness constraint:
    unbias = cov
    first  = False
    if ndb <= 1:
        cbb = cov
    else:
        for i in range(0,ndb): 
            for j in range(0,ndb): 
                cov = cova2(xdb[i],ydb[i],xdb[j],ydb[j],nst,c0,PMX,cc,aa,it,ang,anis,rotmat,maxcov)
            if i == j: 
                cov = cov - c0
            cbb = cbb + cov
        cbb = cbb/real(ndb*ndb)

# MAIN LOOP OVER ALL THE BLOCKS IN THE GRID:
    nk = 0
    ak = 0.0
    vk = 0.0
    for iy in range(0,ny):
        yloc = ymn + (iy-0)*ysiz  
        for ix in range(0,nx):
            xloc = xmn + (ix-0)*xsiz
            current_node = (yloc,xloc)
        
# Find the nearest samples within each octant: First initialize
# the counter arrays:
            na = -1   # accounting for 0 as first index
            dist.fill(1.0e+20)
            nums.fill(-1)
            dist, nums = tree.query(current_node,ndmax) # use kd tree for fast nearest data search
            # remove any data outside search radius
            na = len(dist)
            nums = nums[dist<radius]
            dist = dist[dist<radius] 
            na = len(dist)        

# Is there enough samples?
            if na + 1 < ndmin:   # accounting for min index of 0
                est  = UNEST
                estv = UNEST
                print('UNEST at ' + str(ix) + ',' + str(iy))
            else:

# Put coordinates and values of neighborhood samples into xa,ya,vra:
                for ia in range(0,na):
                    jj = int(nums[ia])
                    xa[ia]  = x[jj]
                    ya[ia]  = y[jj]
                    vra[ia] = vr[jj]
                    
# Handle the situation of only one sample:
                if na == 0:  # accounting for min index of 0 - one sample case na = 0
                    cb1 = cova2(xa[0],ya[0],xa[0],ya[0],nst,c0,PMX,cc,aa,it,ang,anis,rotmat,maxcov)
                    xx  = xa[0] - xloc
                    yy  = ya[0] - yloc

# Establish Right Hand Side Covariance:
                    if ndb <= 1:
                        cb = cova2(xx,yy,xdb[0],ydb[0],nst,c0,PMX,cc,aa,it,ang,anis,rotmat,maxcov)
                    else:
                        cb  = 0.0
                        for i in range(0,ndb):                  
                            cb = cb + cova2(xx,yy,xdb[i],ydb[i],nst,c0,PMX,cc,aa,it,ang,anis,rotmat,maxcov)
                            dx = xx - xdb(i)
                            dy = yy - ydb(i)
                            if (dx*dx+dy*dy) < EPSLON:
                                cb = cb - c0
                            cb = cb / real(ndb)
                    if ktype == 0:
                        s[0] = cb/cbb
                        est  = s[0]*vra[0] + (1.0-s[0])*skmean
                        estv = cbb - s[0] * cb
                    else:
                        est  = vra[0]
                        estv = cbb - 2.0*cb + cb1
                else:

# Solve the Kriging System with more than one sample:
                    neq = na + ktype # accounting for first index of 0
#                    print('NEQ' + str(neq))
                    nn  = (neq + 1)*neq/2

# Set up kriging matrices:
                    iin=-1 # accounting for first index of 0
                    for j in range(0,na):

# Establish Left Hand Side Covariance Matrix:
                        for i in range(0,na):  # was j - want full matrix                    
                            iin = iin + 1
                            a[iin] = cova2(xa[i],ya[i],xa[j],ya[j],nst,c0,PMX,cc,aa,it,ang,anis,rotmat,maxcov) 
                        if ktype == 1:
                            iin = iin + 1
                            a[iin] = unbias
                        xx = xa[j] - xloc
                        yy = ya[j] - yloc

# Establish Right Hand Side Covariance:
                        if ndb <= 1:
                            cb = cova2(xx,yy,xdb[0],ydb[0],nst,c0,PMX,cc,aa,it,ang,anis,rotmat,maxcov)
                        else:
                            cb  = 0.0
                            for j1 in range(0,ndb):    
                                cb = cb + cova2(xx,yy,xdb[j1],ydb[j1],nst,c0,PMX,cc,aa,it,ang,anis,rotmat,maxcov)
                                dx = xx - xdb[j1]
                                dy = yy - ydb[j1]
                                if (dx*dx+dy*dy) < EPSLON:
                                    cb = cb - c0
                            cb = cb / real(ndb)
                        r[j]  = cb
                        rr[j] = r[j]

# Set the unbiasedness constraint:
                    if ktype == 1:
                        for i in range(0,na):
                            iin = iin + 1
                            a[iin] = unbias
                        iin      = iin + 1
                        a[iin]   = 0.0
                        r[neq-1]  = unbias
                        rr[neq-1] = r[neq]

# Solve the Kriging System:
#                    print('NDB' + str(ndb))
#                    print('NEQ' + str(neq) + ' Left' + str(a) + ' Right' + str(r))
#                    stop
                    s = ksol_numpy(neq,a,r)
                    ising = 0 # need to figure this out
#                    print('weights' + str(s))
#                    stop
                
            
# Write a warning if the matrix is singular:
                    if ising != 0:
                        print('WARNING KB2D: singular matrix')
                        print('              for block' + str(ix) + ',' + str(iy)+ ' ')
                        est  = UNEST
                        estv = UNEST
                    else:

# Compute the estimate and the kriging variance:
                        est  = 0.0
                        estv = cbb
                        sumw = 0.0
                        if ktype == 1: 
                            estv = estv - (s[na])*unbias
                        for i in range(0,na):                          
                            sumw = sumw + s[i]
                            est  = est  + s[i]*vra[i]
                            estv = estv - s[i]*rr[i]
                        if ktype == 0: 
                            est = est + (1.0-sumw)*skmean
            kmap[ny-iy-1,ix] = est
            vmap[ny-iy-1,ix] = estv
            if est > UNEST:
                nk = nk + 1
                ak = ak + est
                vk = vk + est*est

# END OF MAIN LOOP OVER ALL THE BLOCKS:

    if nk >= 1:
        ak = ak / float(nk)
        vk = vk/float(nk) - ak*ak
        print('  Estimated   ' + str(nk) + ' blocks ')
        print('      average   ' + str(ak) + '  variance  ' + str(vk))

    return kmap, vmap

def ik2d(
    df,xcol,ycol,vcol,ivtype,koption,ncut,thresh,gcdf,trend,tmin,tmax,nx,xmn,xsiz,ny,ymn,ysiz,ndmin,ndmax,radius,ktype,vario): 

     """A 2D version of GSLIB's IK3D Indicator Kriging program (Deutsch and Journel, 1998) converted from the
    original Fortran to Python by Michael Pyrcz, the University of Texas at
    Austin (March, 2019).

    :param df: pandas DataFrame with the spatial data
    :param xcol: name of the x coordinate column
    :param ycol: name of the y coordinate column
    :param vcol: name of the property column (cateogorical or continuous - note continuous is untested)
    :param ivtype: variable type, 0 - categorical, 1 - continuous
    :param koption: kriging option, 0 - estimation, 1 - cross validation (under construction)
    :param ncut: number of categories or continuous thresholds
    :param thresh: an ndarray with the category labels or continuous thresholds
    :param gcdf: global CDF, not used if trend is present
    :param trend: an ndarray [ny,ny,ncut] with the local trend proportions or cumulative CDF values
    :param tmin: property trimming limit
    :param tmax: property trimming limit
    :param nx: definition of the grid system (x axis)
    :param xmn: definition of the grid system (x axis)
    :param xsiz: definition of the grid system (x axis)
    :param ny: definition of the grid system (y axis)
    :param ymn: definition of the grid system (y axis)
    :param ysiz: definition of the grid system (y axis)
    :param nxdis: number of discretization points for a block
    :param nydis: number of discretization points for a block
    :param ndmin: minimum number of data points to use for kriging a block
    :param ndmax: maximum number of data points to use for kriging a block
    :param radius: maximum isotropic search radius
    :param ktype: kriging type, 0 - simple kriging and 1 - ordinary kriging
    :param vario: list with all of the indicator variograms (sill of 1.0) in consistent order with above parameters
    :return:
    """
    
# Find the needed paramters:
    PMX = 9999.9
    MAXSAM = ndmax + 1
    MAXEQ = MAXSAM + 1
    mik = 0  # full indicator kriging
    use_trend = False
    if trend.shape[0] == nx and trend.shape[1] == ny and trend.shape[2] == ncut: use_trend = True
    
# load the variogram
    MAXNST = 2
    nst = np.zeros(ncut,dtype=int); c0 = np.zeros(ncut); cc = np.zeros((MAXNST,ncut)) 
    aa = np.zeros((MAXNST,ncut),dtype=int); it = np.zeros((MAXNST,ncut),dtype=int) 
    ang = np.zeros((MAXNST,ncut)); anis = np.zeros((MAXNST,ncut))

    for icut in range(0,ncut):
        nst[icut] = int(vario[icut]['nst'])
        c0[icut] = vario[icut]['nug']; cc[0,icut] = vario[icut]['cc1']; it[0,icut] = vario[icut]['it1']; 
        ang[0,icut] = vario[icut]['azi1']; 
        aa[0,icut] = vario[icut]['hmaj1']; anis[0,icut] = vario[icut]['hmin1']/vario[icut]['hmaj1'];
        if nst[icut] == 2:
            cc[1,icut] = vario[icut]['cc2']; it[1,icut] = vario[icut]['it2']; ang[1,icut] = vario[icut]['azi2']; 
            aa[1,icut] = vario[icut]['hmaj2']; anis[1,icut] = vario[icut]['hmin2']/vario[icut]['hmaj2'];

# Load the data
    df_extract = df.loc[(df[vcol] >= tmin) & (df[vcol] <= tmax)]    # trim values outside tmin and tmax
    MAXDAT = len(df_extract)
    MAXCUT = ncut
    MAXNST = 2
    MAXROT = MAXNST*MAXCUT+ 1
    ikout = np.zeros((nx,ny,ncut))
    maxcov = np.zeros(ncut)
            
    # Allocate the needed memory:   
    xa = np.zeros(MAXSAM)
    ya = np.zeros(MAXSAM)
    vra = np.zeros(MAXSAM)
    dist = np.zeros(MAXSAM)
    nums = np.zeros(MAXSAM)
    r = np.zeros(MAXEQ)
    rr = np.zeros(MAXEQ)
    s = np.zeros(MAXEQ)
    a = np.zeros(MAXEQ*MAXEQ)
    ikmap = np.zeros((nx,ny,ncut))
    vr = np.zeros((MAXDAT,MAXCUT+1))
    
    nviol = np.zeros(MAXCUT)
    aviol = np.zeros(MAXCUT)
    xviol = np.zeros(MAXCUT)
    
    ccdf = np.zeros(ncut)
    ccdfo = np.zeros(ncut)
    ikout = np.zeros((nx,ny,ncut))
    
    x = df_extract[xcol].values
    y = df_extract[ycol].values
    v = df_extract[vcol].values
    
# The indicator data are constructed knowing the thresholds and the
# data value.
    
    if ivtype == 0:
        for icut in range(0,ncut): 
            vr[:,icut] = np.where((v <= thresh[icut] + 0.5) & (v > thresh[icut] - 0.5), '1', '0')
    else:
        for icut in range(0,ncut): 
            vr[:,icut] = np.where(v <= thresh[icut], '1', '0')
    vr[:,ncut] = v

# Make a KDTree for fast search of nearest neighbours   
    dp = list((y[i], x[i]) for i in range(0,MAXDAT))
    data_locs = np.column_stack((y,x))
    tree = sp.cKDTree(data_locs, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True)
    
# Summary statistics of the input data
    
    avg = vr[:,ncut].mean()
    stdev = vr[:,ncut].std()
    ss = stdev**2.0
    vrmin = vr[:,ncut].min()
    vrmax = vr[:,ncut].max()
    print('Data for IK3D: Variable column ' + str(vcol))
    print('  Number   = ' + str(MAXDAT))
    ndh = MAXDAT
    
    actloc = np.zeros(MAXDAT, dtype = int)
    for i in range(1,MAXDAT):
        actloc[i] = i
    
# Set up the rotation/anisotropy matrices that are needed for the
# variogram and search:

    print('Setting up rotation matrices for variogram and search')
    radsqd = radius * radius
    rotmat = []
    for ic in range(0,ncut):  
        rotmat_temp, maxcov[ic] = setup_rotmat(c0[ic],int(nst[ic]),it[:,ic],cc[:,ic],ang[:,ic],9999.9)
        rotmat.append(rotmat_temp)    
# Initialize accumulators:  # not setup yet
    nk = 0
    xk = 0.0
    vk = 0.0
    for icut in range (0,ncut):
        nviol[icut] =  0
        aviol[icut] =  0.0
        xviol[icut] = -1.0
    nxy   = nx*ny
    print('Working on the kriging')

# Report on progress from time to time:
    if koption == 0: 
        nxy   = nx*ny
        nloop = nxy
        irepo = max(1,min((nxy/10),10000))
    else:
        nloop = 10000000
        irepo = max(1,min((nd/10),10000))
    ddh = 0.0
    
# MAIN LOOP OVER ALL THE BLOCKS IN THE GRID:
    for index in range(0,nloop):
      
        if (int(index/irepo)*irepo) == index: print('   currently on estimate ' + str(index))
    
        if koption == 0:
            iy   = int((index)/nx) 
            ix   = index - (iy)*nx
            xloc = xmn + (ix)*xsiz
            yloc = ymn + (iy)*ysiz
        else:
            ddh = 0.0
            # TODO: pass the cross validation value

# Find the nearest samples within each octant: First initialize the counter arrays:
        na = -1   # accounting for 0 as first index
        dist.fill(1.0e+20)
        nums.fill(-1)
        current_node = (yloc,xloc)
        dist, close = tree.query(current_node,ndmax) # use kd tree for fast nearest data search
        # remove any data outside search radius
        close = close[dist<radius]
        dist = dist[dist<radius] 
        nclose = len(dist) 

# Is there enough samples?

        if nclose < ndmin:   # accounting for min index of 0
            for i in range(0,ncut):
                ccdfo[i] = UNEST
            print('UNEST at ' + str(ix) + ',' + str(iy))
        else:         

# Loop over all the thresholds/categories:
            for ic in range(0,ncut):
                krig = True
                if mik == 1 and ic >= 1: krig = False

# Identify the close data (there may be a different number of data at
# each threshold because of constraint intervals); however, if
# there are no constraint intervals then this step can be avoided.
                nca = -1
                for ia in range(0,nclose):
                    j  = int(close[ia]+0.5)
                    ii = actloc[j]
                    accept = True
                    if koption != 0 and (abs(x[j]-xloc) + abs(y[j]-yloc)).lt.EPSLON: accept = False
                    if accept:
                        nca = nca + 1
                        vra[nca] = vr[ii,ic]
                        xa[nca]  = x[j]
                        ya[nca]  = y[j]

# If there are no samples at this threshold then use the global cdf:
                if nca == -1:
                    if use_trend:
                        ccdf[ic] = trend[ny-iy-1,ix,ic]
                    else:
                        ccdf[ic] = gcdf[ic]
                else:
            
# Now, only load the variogram, build the matrix,... if kriging:
                    neq = nclose + ktype
                    na = nclose

# Set up kriging matrices:
                    iin=-1 # accounting for first index of 0
                    for j in range(0,na):
# Establish Left Hand Side Covariance Matrix:
                        for i in range(0,na):  # was j - want full matrix                    
                            iin = iin + 1
                            a[iin] = cova2(xa[i],ya[i],xa[j],ya[j],nst[ic],c0[ic],PMX,cc[:,ic],aa[:,ic],it[:,ic],ang[:,ic],anis[:,ic],rotmat[ic],maxcov[ic]) 
                        if ktype == 1:
                            iin = iin + 1
                            a[iin] = maxcov[ic]            
                        r[j] = cova2(xloc,yloc,xa[j],ya[j],nst[ic],c0[ic],PMX,cc[:,ic],aa[:,ic],it[:,ic],ang[:,ic],anis[:,ic],rotmat[ic],maxcov[ic]) 
    
# Set the unbiasedness constraint:
                    if ktype == 1:
                        for i in range(0,na):
                            iin = iin + 1
                            a[iin] = maxcov[ic]
                        iin      = iin + 1
                        a[iin]   = 0.0
                        r[neq-1]  = maxcov[ic]
                        rr[neq-1] = r[neq]
# Solve the system:
                    if neq == 1:
                        ising = 0.0
                        s[0]  = r[0] / a[0]
                    else:
                        s = ksol_numpy(neq,a,r)

# Finished kriging (if it was necessary):

# Compute Kriged estimate of cumulative probability:
                    sumwts   = 0.0
                    ccdf[ic] = 0.0
                    for i in range(0,nclose):
                        ccdf[ic] = ccdf[ic] + vra[i]*s[i]
                        sumwts   = sumwts   + s[i]
                    if ktype == 0: 
                        if use_trend == True:
                            ccdf[ic] = ccdf[ic] + (1.0-sumwts)*trend[ny-iy-1,ix,ic]
                        else:
                            ccdf[ic] = ccdf[ic] + (1.0-sumwts)*gcdf[ic]

# Keep looping until all the thresholds are estimated:
 
# Correct and write the distribution to the output file:
            nk = nk + 1
            ccdfo = ordrel(ivtype,ncut,ccdf)
        
# Write the IK CCDF for this grid node:
            if koption == 0:
                 ikout[ny-iy-1,ix,:] = ccdfo
            else:
                 print('TBD')
    return ikout  
