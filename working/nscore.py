import numpy as np
import pandas as pd

# GSLIB's NSCORE program (Deutsch and Journel, 1998) converted from the original Fortran to Python 
# by Michael Pyrcz, the University of Texas at Austin (Jan, 2019)
# First we have the necessary GSLIB subroutines translated 

def dlocate(xx,iis,iie,x):
    from bisect import bisect
    n = len(xx)
    if iie <= iis:
        iis = 0; ie = n-1
    array = xx[iis:iie-1]  # this is accounting for swith to 0,...,n-1 index
    j = bisect(array,x)
    return j

def dsortem(ib,ie,a,iperm,b=0,c=0,d=0,e=0,f=0,g=0,h=0):
    a = a[ib:ie]
    inds = a.argsort()
    a = np.copy(a[inds]) # deepcopy forces pass to outside scope
    if(iperm == 1):
        return a
    b_slice = b[ib:ie]
    b = b_slice[inds]    
    if iperm == 2:
        return a,b
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
        return a, b, c, d, e, f, h     
    h_slice = h[ib:ie]
    h = h_slice[inds]
    return a, b, c, d, e, f, h

def gauinv(p):
    lim = 1.0e-10; p0 = -0.322232431088; p1 = -1.0; p2 = -0.342242088547
    p3 = -0.0204231210245; p4 = -0.0000453642210148; q0 = 0.0993484626060
    q1 = 0.588581570495; q2 = 0.531103462366; q3 = 0.103537752850; q4 = 0.0038560700634

# Check for an error situation:
    if p < lim:
        xp = -1.0e10
        return xp
    if p > (1.0-lim):
        xp =  1.0e10
        return xp    

# Get k for an error situation:
    pp = p
    if p > 0.5: pp = 1 - pp
    xp   = 0.0
    if p == 0.5: 
        return xp

# Approximate the function:
    y  = np.sqrt(np.log(1.0/(pp*pp)))
    xp = float(y + ((((y*p4+p3)*y+p2)*y+p1)*y+p0) /
            ((((y*q4+q3)*y+q2)*y+q1)*y+q0) )
    if float(p) == float(pp): 
        xp = -xp
    return xp

def gcum(x):
    z = x
    if z < 0:  
        z = -z
    t= 1./(1.+ 0.2316419*z)
    gcum = t*(0.31938153   + t*(-0.356563782 + t*(1.781477937 +
           t*(-1.821255978 + t*1.330274429))))
    e2= 0.0
    
# standard deviations out gets treated as infinity:
    if z <= 6: 
        e2 = np.exp(-z*z/2.0)*0.3989422803
    gcum = 1.0- e2 * gcum
    if x >= 0.0: 
        return gcum
    gcum = 1.0 - gcum
    return gcum

def dpowint(xlow,xhigh,ylow,yhigh,xval,pwr):
    EPSLON = 1.0e-20
    if (xhigh-xlow) < EPSLON:
        dpowint = (yhigh+ylow)/2.0
    else:
        dpowint = ylow + (yhigh-ylow)*(((xval-xlow)/(xhigh-xlow))**pwr)
    return dpowint

# GSLIB's NSCORE program (Deutsch and Journel, 1998) converted from the original Fortran to Python 
# by Michael Pyrcz, the University of Texas at Austin (Jan, 2019)
def nscore(df,vcol,wcol=0,ismooth=0,dfsmooth=0,smcol=0,smwcol=0):
# Parameters - consistent with original GSLIB    
# df - Pandas DataFrame with the spatial data
# vcol - name of the variable column
# wcol (optional) - name of the weigth column, if not included assumes equal weighting
# ismooth - if 1 then use a reference distribution 
# dfsmooth - Pandas DataFrame required if reference distribution is used
# smcol, smwtcol - reference distribution property and weight required if reference distribution is used
#
# Set constants
    np.random.seed(73073)
    pwr = 1.0                         # interpolation power, hard coded to 1.0 in GSLIB
    EPSILON = 1.0e-20

# Decide which file to use for establishing the transformation table:
    if ismooth == 1: 
        nd = len(dfsmooth)
        vr = dfsmooth[smcol].values
        wt_ns = np.ones(nd)
        if smwcol != 0:
            wt_ns = dfsmooth[smwcol].values 
    else:
        nd = len(df)
        vr = df[vcol].values
        wt_ns = np.ones(nd)
        if wcol != 0:
            wt_ns = df[wcol].values
    twt = np.sum(wt_ns)

# Sort data by value:
    istart = 0
    iend   = nd
    vr, wt_ns = dsortem(istart,iend,vr,2,wt_ns)

# Compute the cumulative probabilities and write transformation table
    wtfac = 1.0/twt
    oldcp = 0.0
    cp    = 0.0
    for j in range(istart,iend):  
        w = wtfac*wt_ns[j]
        cp = cp + w
        wt_ns[j] = (cp + oldcp)/2.0
        vrrg = gauinv(wt_ns[j])
        vrg = float(vrrg)
        oldcp =  cp

# Now, reset the weight to the normal scores value:
        wt_ns[j] = vrg
    
# Normal Scores Transform:

    nd_trans = len(df)
    ns = np.zeros(nd_trans)
    val = df[vcol].values
    for i in range(0,nd_trans): 
        vrr = val[i] + np.random.rand() * EPSILON

# Now, get the normal scores value for "vrr" 
        j = dlocate(vr,1,nd,vrr)
        j   = min(max(1,j),(nd-1))
        ns[i] = dpowint(vr[j],vr[j+1],wt_ns[j],wt_ns[j+1],vrr,pwr)
        #print(vrr,ns[i])
    return ns, vr, wt_ns