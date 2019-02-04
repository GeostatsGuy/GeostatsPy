# GSLIB's GAM program (Deutsch and Journel, 1998) converted from the original Fortran to Python 
# by Michael Pyrcz, the University of Texas at Austin (Jan, 2019)
def gam(array,tmin,tmax,xsiz,ysiz,ixd,iyd,nlag,isill): 
# Parameters - consistent with original GSLIB    
# array - 2D gridded data / model
# tmin, tmax - property trimming limits
# xsiz, ysiz - grid cell extents in x and y directions
# ixd, iyd - lag offset in grid cells
# nlag - number of lags to calculate
# isill - 1 for standardize sill
#
# Set constants
    if array.ndim == 2:
        ny = (array.shape[0])
        nx = (array.shape[1])
    elif array.ndim == 1:
        nx = len(array)     
        ny = 1

    nvarg = 1   # for mulitple variograms repeat the program
    nxy = nx*ny
    mxdlv = nlag

# Allocate the needed memory:   
    lag = np.zeros(mxdlv)
    vario = np.zeros(mxdlv)
    hm = np.zeros(mxdlv)
    tm = np.zeros(mxdlv)
    hv = np.zeros(mxdlv)
    npp = np.zeros(mxdlv)
    ivtail = np.zeros(nvarg + 2)
    ivhead = np.zeros(nvarg + 2)
    ivtype = np.zeros(nvarg + 2)
    ivtail[0] = 0; ivhead[0] = 0; ivtype[0] = 0;
    
# Summary statistics for the data after trimming
    inside = ((array > tmin) & (array < tmax))
    avg = array[(array > tmin) & (array < tmax)].mean()
    stdev = array[(array > tmin) & (array < tmax)].std()
    var = stdev**2.0
    vrmin = array[(array > tmin) & (array < tmax)].min()
    vrmax = array[(array > tmin) & (array < tmax)].max()
    num = ((array > tmin) & (array < tmax)).sum()
    
# For the fixed seed point, loop through all directions:
    for iy in range(0,ny):
        for ix in range(0,nx):
            if inside[iy,ix]:
                vrt = array[iy,ix]
                ixinc = ixd
                iyinc = iyd
                ix1   = ix
                iy1   = iy
                for il in range(0,nlag):
                    ix1 = ix1 + ixinc
                    if ix1 >= 0 and ix1 < nx:
                        iy1 = iy1 + iyinc
                        if iy1 >= 1 and iy1 < ny:
                            if inside[iy1,ix1]:
                                vrh = array[iy1,ix1]
                                npp[il] = npp[il] + 1
                                tm[il] = tm[il] + vrt
                                hm[il] = hm[il] + vrh
                                vario[il] = vario[il] + ((vrh-vrt)**2.0) 

# Get average values for gam, hm, tm, hv, and tv, then compute
# the correct "variogram" measure:
    for il in range(0,nlag):
        if npp[il] > 0:
            rnum   = npp[il]
            lag[il] = np.sqrt((ixd*xsiz*il)**2+(iyd*ysiz*il)**2)
            vario[il] = vario[il] / float(rnum)
            hm[il]  = hm[il]  / float(rnum)
            tm[il]  = tm[il]  / float(rnum)       

# Standardize by the sill

            if isill == 1:
                vario[il] = vario[il] / var
            
# Semivariogram

            vario[il] = 0.5 * vario[il]
    return lag, vario, npp
    