import math                       # for trig and constants
from numba import jit             # for precompile speed up of loops with NumPy ndarrays

# GSLIB's GAMV program (Deutsch and Journel, 1998) converted from the original Fortran to Python 
# by Michael Pyrcz, the University of Texas at Austin (Jan, 2019)
# Note simplified for 2D, semivariogram only and one direction at a time

def gamv(df,xcol,ycol,vcol,tmin,tmax,xlag,xltol,nlag,azm,atol,bandwh,isill): 
# Parameters - consistent with original GSLIB    
# df - DataFrame with the spatial data, xcol, ycol, vcol coordinates and property columns
# tmin, tmax - property trimming limits
# xlag, xltol - lag distance and lag distance tolerance
# nlag - number of lags to calculate
# azm, atol - azimuth and azimuth tolerance
# bandwh - horizontal bandwidth / maximum distance offset orthogonal to azimuth
# isill - 1 for standardize sill

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
    #print('Number of Data ' + str(nd) +', Average ' + str(avg) + ' Variance ' + str(sills))

# Define the distance tolerance if it isn't already:
    if xltol < 0.0: xltol = 0.5 * xlag

# Loop over combinatorial of data pairs to calculate the variogram
    dis, vario, npp = variogram_loop(x,y,vr,xlag,xltol,nlag,azm,atol,bandwh) 

# Standardize sill to one by dividing all variogram values by the variance
    for il in range(0,nlag+2):
        if isill == 1:
            vario[il] = vario[il] / sills

# Apply 1/2 factor to go from variogram to semivariogram            
        vario[il] = 0.5 * vario[il]
    
# END - return variogram model information
    return dis, vario, npp
    
@jit(nopython=True) # all NumPy array operations included in this function for precompile with NumBa
def variogram_loop(x,y,vr,xlag,xltol,nlag,azm,atol,bandwh):
    
# Allocate the needed memory: 
    nvarg = 1
    mxdlv = nlag + 2 # in gamv the npp etc. arrays go to nlag + 2
    dis = np.zeros(mxdlv)
    lag = np.zeros(mxdlv)
    vario = np.zeros(mxdlv)
    hm = np.zeros(mxdlv)
    tm = np.zeros(mxdlv)
    hv = np.zeros(mxdlv)
    npp = np.zeros(mxdlv)
    ivtail = np.zeros(nvarg + 2)
    ivhead = np.zeros(nvarg + 2)
    ivtype = np.ones(nvarg + 2)
    ivtail[0] = 0; ivhead[0] = 0; ivtype[0] = 0;
    
    EPSLON = 1.0e-20
    nd = len(x)
# The mathematical azimuth is measured counterclockwise from EW and
# not clockwise from NS as the conventional azimuth is:
    azmuth = (90.0-azm)*math.pi/180.0
    uvxazm = math.cos(azmuth)
    uvyazm = math.sin(azmuth)
    if atol <= 0.0:
        csatol = math.cos(45.0*math.pi/180.0)
    else:
        csatol = math.cos(atol*math.pi/180.0)

# Initialize the arrays for each direction, variogram, and lag:
    nsiz = nlag+2
    dismxs = ((float(nlag) + 0.5 - EPSLON) * xlag) ** 2  
    
# MAIN LOOP OVER ALL PAIRS:
    for i in range(0,nd):
        for j in range(0,nd):

# Definition of the lag corresponding to the current pair:
            dx  = x[j] - x[i]
            dy  = y[j] - y[i]
            dxs = dx*dx
            dys = dy*dy
            hs  = dxs + dys
            if hs <= dismxs:
                if hs < 0.0: 
                    hs = 0.0
                h = np.sqrt(hs)

# Determine which lag this is and skip if outside the defined distance
# tolerance:            
                if h <= EPSLON:
                    lagbeg = 0
                    lagend = 0
                else:
                    lagbeg = -1
                    lagend = -1
                    for ilag in range(1,nlag+1):
                        if h >= (xlag*float(ilag-1)-xltol) and h <= (xlag*float(ilag-1)+xltol): # reduced to -1
                            if lagbeg < 0: 
                                lagbeg = ilag 
                            lagend = ilag 
                if lagend >= 0: 

# Definition of the direction corresponding to the current pair. All
# directions are considered (overlapping of direction tolerance cones
# is allowed):

# Check for an acceptable azimuth angle:
                    dxy = np.sqrt(max((dxs+dys),0.0))
                    if dxy < EPSLON:
                        dcazm = 1.0
                    else:
                        dcazm = (dx*uvxazm+dy*uvyazm)/dxy

# Check the horizontal bandwidth criteria (maximum deviation 
# perpendicular to the specified direction azimuth):
                    band = uvxazm*dy - uvyazm*dx
                  
# Apply all the previous checks at once to avoid a lot of nested if statements
                    if (abs(dcazm) >= csatol) and (abs(band) <= bandwh):
# Check whether or not an omni-directional variogram is being computed:
                        omni = False
                        if atol >= 90.0: omni = True

# For this variogram, sort out which is the tail and the head value:
                        iv = 0  # hardcoded just one varioigram
                        it = ivtype[iv]
                        if dcazm >= 0.0:
                            vrh   = vr[i]
                            vrt   = vr[j]
                            if omni:
                                vrtpr = vr[i]
                                vrhpr = vr[j]
                        else:
                            vrh   = vr[j]
                            vrt   = vr[i]
                            if omni:
                                vrtpr = vr[j]
                                vrhpr = vr[i]

# Reject this pair on the basis of missing values:

# Data was trimmed at the beginning

# The Semivariogram (all other types of measures are removed for now)
                        for il in range(lagbeg,lagend+1):
                            npp[il] = npp[il] + 1
                            dis[il] = dis[il] + h
                            tm[il]  = tm[il]  + vrt
                            hm[il]  = hm[il]  + vrh
                            vario[il] = vario[il] + ((vrh-vrt)*(vrh-vrt))
                            if(omni):
                                npp[il]  = npp[il]  + 1.0
                                dis[il] = dis[il] + h
                                tm[il]  = tm[il]  + vrtpr
                                hm[il]  = hm[il]  + vrhpr
                                vario[il] = vario[il] + ((vrhpr-vrtpr)*(vrhpr-vrtpr))
                                
# Get average values for gam, hm, tm, hv, and tv, then compute
# the correct "variogram" measure:
    for il in range(0,nlag+2):
        i = il
        if npp[i] > 0:
            rnum   = npp[i]
            dis[i] = dis[i] / (rnum)
            vario[i] = vario[i] / (rnum)
            hm[i]  = hm[i]  / (rnum)
            tm[i]  = tm[i]  / (rnum)
    
    return dis, vario, npp