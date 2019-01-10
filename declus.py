import numpy as np
import pandas as pd

# GSLIB's DECLUS program (Deutsch and Journel, 1998) converted from the original Fortran to Python 
# by Michael Pyrcz, the University of Texas at Austin (Jan, 2019)
# note this was simplified to 2D only
def declus(df,xcol,ycol,vcol,iminmax,noff,ncell,cmin,cmax):
# Parameters - consistent with original GSLIB    
# df - Pandas DataFrame with the spatial data
# xcol, ycol - name of the x and y coordinate columns
# vcol - name of the property column
# iminmax - 1 / True for use cell size with max decluster mean, 0 / False for declustered mean minimizing cell size
# noff - number of offsets
# ncell - number of cell sizes
# cmin, cmax - min and max cell size
#
# Load Data and Set Up Arrays
    nd = len(df)
    x = df[xcol].values
    y = df[ycol].values
    v = df[vcol].values
    wt = np.zeros(nd)
    wtopt = np.ones(nd)
    index = np.zeros(nd, np.int32)
    xcs_mat = np.zeros(ncell+2) # we use 1,...,n for this array
    vrcr_mat = np.zeros(ncell+2) # we use 1,...,n for this array
    anisy = 1.0   # hard code the cells to 2D isotropic
    roff = float(noff)
    
# Calculate extents    
    xmin = np.min(x); xmax = np.max(x)
    ymin = np.min(y); ymax = np.max(y)
      
# Calculate summary statistics
    vmean = np.mean(v)
    vstdev = np.std(v)
    vmin = np.min(v)
    vmax = np.max(v)
    xcs_mat[0] = 0.0; vrcr_mat[0] = vmean; vrop = vmean # include the naive case
    print('There are ' + str(nd) + ' data with:')
    print('   mean of      ' + str(vmean) + ' ')
    print('   min and max  ' + str(vmin) + ' and ' + str(vmax))
    print('   standard dev ' + str(vstdev) + ' ')
    
# define a "lower" origin to use for the cell sizes:
    xo1 = xmin - 0.01
    yo1 = ymin - 0.01

# define the increment for the cell size:
    xinc = (cmax-cmin) / ncell
    yinc = xinc

# loop over "ncell+1" cell sizes in the grid network:
    ncellx = int((xmax-(xo1-cmin))/cmin)+1
    ncelly = int((ymax-(yo1-cmin*anisy))/(cmin))+1
    ncellt = ncellx*ncelly 
    cellwt = np.zeros(ncellt)
    xcs =  cmin - xinc
    ycs = (cmin*anisy) - yinc

# MAIN LOOP over cell sizes:
    for lp in range(1,ncell+2):   # 0 index is the 0.0 cell, note n + 1 in Fortran
        xcs = xcs + xinc
        ycs = ycs + yinc
        
# initialize the weights to zero: 
        wt.fill(0.0)

# determine the maximum number of grid cells in the network:
        ncellx = int((xmax-(xo1-xcs))/xcs)+1
        ncelly = int((ymax-(yo1-ycs))/ycs)+1
        ncellt = float(ncellx*ncelly)

# loop over all the origin offsets selected:
        xfac = min((xcs/roff),(0.5*(xmax-xmin)))
        yfac = min((ycs/roff),(0.5*(ymax-ymin)))
        for kp in range(1,noff+1):
            xo = xo1 - (float(kp)-1.0)*xfac
            yo = yo1 - (float(kp)-1.0)*yfac

# initialize the cumulative weight indicators:
            cellwt.fill(0.0)
    
# determine which cell each datum is in:
            for i in range(0,nd):
                icellx = int((x[i] - xo)/xcs) + 1
                icelly = int((y[i] - yo)/ycs) + 1
                icell  = icellx + (icelly-1)*ncellx  
                index[i] = icell
                cellwt[icell] = cellwt[icell] + 1.0


# The weight assigned to each datum is inversely proportional to the
# number of data in the cell.  We first need to get the sum of weights
# so that we can normalize the weights to sum to one:
            sumw = 0.0
            for i in range(0,nd):
                ipoint = index[i]
                sumw   = sumw + (1.0 / cellwt[ipoint])
            sumw = 1.0 / sumw
                
# Accumulate the array of weights (that now sum to one):
            for i in range(0,nd):
                ipoint = index[i]
                wt[i] = wt[i] + (1.0/cellwt[ipoint])*sumw

# End loop over all offsets:

# compute the weighted average for this cell size:
        sumw  = 0.0
        sumwg = 0.0
        for i in range(0,nd):
            sumw  = sumw + wt[i]
            sumwg = sumwg + wt[i]*v[i]
        vrcr = sumwg / sumw
        vrcr_mat[lp] = vrcr
        xcs_mat[lp] = xcs

# see if this weighting is optimal:
        if iminmax and vrcr < vrop or not iminmax and vrcr > vrop or ncell == 1:
            best = xcs
            vrop = vrcr
            wtopt = wt.copy()   # deep copy

# END MAIN LOOP over all cell sizes:

# Get the optimal weights:
    sumw = 0.0
    for i in range(0,nd):
        sumw = sumw + wtopt[i]
    wtmin = np.min(wtopt)
    wtmax = np.max(wtopt)
    facto = float(nd) / sumw
    wtopt = wtopt * facto
    return wtopt,xcs_mat,vrcr_mat