"""
Some GeostatsPy functions - by Michael Pyrcz, maintained at
https://git.io/fhbRb.

A set of functions to provide reimplementations of GSLIB and access to GSLIB
functionalities through GSLIB .exe in Python. For the functions nscore, declus,
gam, gamv, vmodel, kb2d and sgsim you will need to have these executables:
nscore.exe, declus.exe, gam.exe, gamv.exe, vmodel.exe, kb2d.exe & sgsim.exe must
be in the working directory or in a directory included in the path environmental
variables.

The source for these are available from GSLIB.com. If you would like to get the
executables, ready to use without any need to compile them, go to GSLIB.com for
Windows and Linux. I failed to find any Mac OS X executables so my Ph.D. student
Wendi Liu compiled them for us (thank you Wendi!) and we have posted them here
https://github.com/GeostatsGuy/GSLIB_MacOS. If folks on Windows are encountering
missing DLL's, I could post static builds. Wendi provided instructions to help
Mac users with missing DLL issues at that same location above.
"""

import os  # for setting working directory and running fortran executables
import random as rand  # for random numbers

import matplotlib
import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for ndarrays
import pandas as pd  # for DataFrames
from scipy import signal

# Hard coded image file output
image_type = "tif"
dpi = 600

def ndarray2GSLIB_3D(array, data_file, col_name):
    """Convert 1D or 2D or 3D numpy ndarray to a GSLIB Geo-EAS file for use with
    GSLIB methods.

    :param array: input array
    :param data_file: file name
    :param col_name: column name
    :return: None
    """

    if array.ndim not in [1, 2,3]:
        raise ValueError("must use a 3D array")

    with open(data_file, "w") as f:
        f.write(data_file + "\n")
        f.write("1 \n")
        f.write(col_name + "\n")

        if array.ndim == 3:
            nz, ny, nx = array.shape

            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        f.write(str(array[nz - 1 - iz,ny - 1 - iy, ix]) + "\n")
        
        if array.ndim == 2:
            ny, nx = array.shape

            for iy in range(ny):
                for ix in range(nx):
                    f.write(str(array[ny - 1 - iy, ix]) + "\n")

        elif array.ndim == 1:
            nx = len(array)
            for ix in range(0, nx):
                f.write(str(array[ix]) + "\n")


def GSLIB2ndarray(data_file, kcol, nx, ny):
    """Convert GSLIB Geo-EAS file to a 1D or 2D numpy ndarray for use with
    Python methods

    :param data_file: file name
    :param kcol: TODO
    :param nx: shape along x dimension
    :param ny: shape along y dimension
    :return: ndarray, column name
    """
    if ny > 1:
        array = np.ndarray(shape=(ny, nx), dtype=float, order="F")
    else:
        array = np.zeros(nx)

    with open(data_file) as f:
        head = [next(f) for _ in range(2)]  # read first two lines
        line2 = head[1].split()
        ncol = int(line2[0])  # get the number of columns

        for icol in range(ncol):  # read over the column names
            head = next(f)
            if icol == kcol:
                col_name = head.split()[0]
        if ny > 1:
            for iy in range(ny):
                for ix in range(0, nx):
                    head = next(f)
                    array[ny - 1 - iy][ix] = head.split()[kcol]
        else:
            for ix in range(nx):
                head = next(f)
                array[ix] = head.split()[kcol]
    return array, col_name

def GSLIB2ndarray_3D(data_file, kcol,nreal, nx, ny, nz):
    """Convert GSLIB Geo-EAS file to a 1D or 2D numpy ndarray for use with
    Python methods

    :param data_file: file name
    :param kcol: name of column which contains property
    :param nreal: Number of realizations
    :param nx: shape along x dimension
    :param ny: shape along y dimension
    :param nz: shape along z dimension
    :return: ndarray, column name
    """
    if nz > 1 and ny > 1:
        array = np.ndarray(shape = (nreal, nz, ny, nx), dtype=float, order="F")
    elif ny > 1:
        array = np.ndarray(shape=(nreal, ny, nx), dtype=float, order="F")
    else:
        array = np.zeros(nreal, nx)

    with open(data_file) as f:
        head = [next(f) for _ in range(2)]  # read first two lines
        line2 = head[1].split()
        ncol = int(line2[0])  # get the number of columns

        for icol in range(ncol):  # read over the column names
            head = next(f)
            if icol == kcol:
                col_name = head.split()[0]
        for ineal in range(nreal):
            if nz > 1 and ny > 1:
                for iz in range(nz):
                    for iy in range(ny):
                        for ix in range(nx):
                            head = next(f)
                            array[ineal][nz - 1 - iz][ny - 1 - iy][ix] = head.split()[kcol]
            elif ny > 1:
                for iy in range(ny):
                    for ix in range(0, nx):
                        head = next(f)
                        array[ineal][ny - 1 - iy][ix] = head.split()[kcol]
            else:
                for ix in range(nx):
                    head = next(f)
                    array[ineal][ix] = head.split()[kcol]
    return array, col_name

def Dataframe2GSLIB(data_file, df):
    """Convert pandas DataFrame to a GSLIB Geo-EAS file for use with GSLIB
    methods.

    :param data_file: file name
    :param df: dataframe
    :return: None
    """
    ncol = len(df.columns)
    nrow = len(df.index)

    with open(data_file, "w") as f:
        f.write(data_file + "\n")
        f.write(str(ncol) + "\n")

        for icol in range(ncol):
            f.write(df.columns[icol] + "\n")
        for irow in range(nrow):
            for icol in range(ncol):
                f.write(str(df.iloc[irow, icol]) + " ")
            f.write("\n")


def GSLIB2Dataframe(data_file):
    """Convert GSLIB Geo-EAS files to a pandas DataFrame for use with Python
    methods.

    :param data_file: dataframe
    :return: None
    """

    columns = []
    with open(data_file) as f:
        head = [next(f) for _ in range(2)]  # read first two lines
        line2 = head[1].split()
        ncol = int(line2[0])  # get the number of columns

        for icol in range(ncol):  # read over the column names
            head = next(f)
            columns.append(head.split()[0])

        data = np.loadtxt(f, skiprows=0)
        df = pd.DataFrame(data)
        df.columns = columns
        return df


def hist(array, xmin, xmax, log, cumul, bins, weights, xlabel, title, fig_name):
    """Histogram, reimplemented in Python of GSLIB hist with Matplotlib methods,
    displayed and as image file.

    :param array: ndarray
    :param xmin: lower range of the bins for x axis
    :param xmax: upper range of the bins for x axis
    :param log: if True, the histogram axis will be set to a log scale
    :param cumul: If True, then a histogram is computed where each bin gives
                  the counts in that bin plus all bins for smaller values
    :param bins: bins
    :param weights: an array of weights, of the same shape as `array`
    :param xlabel: label for x axis
    :param title: title
    :param fig_name: figure name
    :return: None
    """
    plt.figure(figsize=(8, 6))
    density = False; edgecolor = 'black'
    if cumul == True: density = True; edgecolor = None
    plt.hist(
        array,
        alpha=0.9,
        color="darkorange",
        edgecolor=edgecolor,
        bins=bins,
        range=[xmin, xmax],
        weights=weights,
        log=log,
        cumulative=cumul,
        density=density
    )
    plt.title(title)
    plt.xlabel(xlabel)
    if cumul == False: 
        plt.ylabel("Frequency") 
    else:
        plt.ylabel("Cumulative Probability")
        plt.ylim([0.0,1.0])
    plt.xlim([xmin,xmax])
    plt.savefig(fig_name + "." + image_type, dpi=dpi)
    plt.show()


def hist_st(array, xmin, xmax, log, cumul, bins, weights, xlabel, title):
    """Histogram, reimplemented in Python of GSLIB hist with Matplotlib methods
    (version for subplots).

    :param array: ndarray
    :param xmin: lower range of the bins for x
    :param xmax: upper range of the bins for x
    :param log: if True, the histogram axis will be set to a log scale
    :param cumul: If True, then a histogram is computed where each bin gives
                  the counts in that bin plus all bins for smaller values
    :param bins: bins
    :param weights: an array of weights, of the same shape as `array`
    :param xlabel: label for x
    :param title: title
    :return: None
    """
    density = False; edgecolor = 'black'
    if cumul == True: density = True; edgecolor = None
    plt.hist(
        array,
        alpha=0.9,
        color="darkorange",
        edgecolor=edgecolor,
        bins=bins,
        range=[xmin, xmax],
        weights=weights,
        log=log,
        cumulative=cumul,
        density=density
    )
    plt.title(title)
    plt.xlabel(xlabel)
    if cumul == False: 
        plt.ylabel("Frequency") 
    else:
        plt.ylabel("Cumulative Probability")
        plt.ylim([0.0,1.0])
    plt.xlim([xmin,xmax])
    
def locmap(
    df,
    xcol,
    ycol,
    vcol,
    xmin,
    xmax,
    ymin,
    ymax,
    vmin,
    vmax,
    title,
    xlabel,
    ylabel,
    vlabel,
    cmap,
    fig_name,
):
    """Location map, reimplementation in Python of GSLIB locmap with Matplotlib
    methods.

    :param df: dataframe
    :param xcol: data for x axis
    :param ycol: data for y axis
    :param vcol: color, sequence, or sequence of color
    :param xmin: x axis minimum
    :param xmax: x axis maximum
    :param ymin: y axis minimum
    :param ymax: y axis maximum
    :param vmin: normalize luminance data
    :param vmax: normalize luminance data
    :param title: title
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param vlabel: TODO
    :param cmap: colormap
    :param fig_name: figure name
    :return: PathCollection
    """
    plt.figure(figsize=(8, 6))
    im = plt.scatter(
        df[xcol],
        df[ycol],
        s=None,
        c=df[vcol],
        marker=None,
        cmap=cmap,
        norm=None,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        linewidths=0.8,
        edgecolors="black",
    )
    plt.title(title)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar(
        im, orientation="vertical", ticks=np.linspace(vmin, vmax, 10)
    )
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    plt.savefig(fig_name + "." + image_type, dpi=dpi)
    plt.show()
    return im


def locmap_st(
    df,
    xcol,
    ycol,
    vcol,
    xmin,
    xmax,
    ymin,
    ymax,
    vmin,
    vmax,
    title,
    xlabel,
    ylabel,
    vlabel,
    cmap,
):
    """Location map, reimplementation in Python of GSLIB locmap with Matplotlib
    methods (version for subplots).

    :param df: dataframe
    :param xcol: data for x axis
    :param ycol: data for y axis
    :param vcol: color, sequence, or sequence of color
    :param xmin: x axis minimum
    :param xmax: x axis maximum
    :param ymin: y axis minimum
    :param ymax: y axis maximum
    :param vmin: normalize luminance data
    :param vmax: normalize luminance data
    :param title: title
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param vlabel: TODO
    :param cmap: colormap
    :return: PathCollection
    """
    im = plt.scatter(
        df[xcol],
        df[ycol],
        s=None,
        c=df[vcol],
        marker=None,
        cmap=cmap,
        norm=None,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        linewidths=0.8,
        edgecolors="black",
    )
    plt.title(title)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar(
        im, orientation="vertical", ticks=np.linspace(vmin, vmax, 10)
    )
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    return im


def pixelplt(
    array,
    xmin,
    xmax,
    ymin,
    ymax,
    step,
    vmin,
    vmax,
    title,
    xlabel,
    ylabel,
    vlabel,
    cmap,
    fig_name,
):
    """Pixel plot, reimplementation in Python of GSLIB pixelplt with Matplotlib
    methods.

    :param array: ndarray
    :param xmin: x axis minimum
    :param xmax: x axis maximum
    :param ymin: y axis minimum
    :param ymax: y axis maximum
    :param step: step
    :param vmin: TODO
    :param vmax: TODO
    :param title: title
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param vlabel: TODO
    :param cmap: colormap
    :param fig_name: figure name
    :return: QuadContourSet
    """
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step)
    )
    plt.figure(figsize=(8, 6))
    # im = plt.contourf(
    #    xx,
    #    yy,
    #    array,
    #    cmap=cmap,
    #    vmin=vmin,
    #    vmax=vmax,
    #    levels=np.linspace(vmin, vmax, 100),
    #)
    im = plt.imshow(array,interpolation = None,extent = [xmin,xmax,ymin,ymax], vmin = vmin, vmax = vmax,cmap = cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar(
        im, orientation="vertical", ticks=np.linspace(vmin, vmax, 10)
    )
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    plt.savefig(fig_name + "." + image_type, dpi=dpi)
    plt.show()
    return im


def pixelplt_st(
    array,
    xmin,
    xmax,
    ymin,
    ymax,
    step,
    vmin,
    vmax,
    title,
    xlabel,
    ylabel,
    vlabel,
    cmap,
):
    """Pixel plot, reimplementation in Python of GSLIB pixelplt with Matplotlib
    methods (version for subplots).

    :param array: ndarray
    :param xmin: x axis minimum
    :param xmax: x axis maximum
    :param ymin: y axis minimum
    :param ymax: y axis maximum
    :param step: step
    :param vmin: TODO
    :param vmax: TODO
    :param title: title
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param vlabel: TODO
    :param cmap: colormap
    :return: QuadContourSet
    """
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step)
    )

    # Use dummy since scatter plot controls legend min and max appropriately
    # and contour does not!
    x = []
    y = []
    v = []

    #cs = plt.contourf(
    #    xx,
    #    yy,
    #    array,
    #    cmap=cmap,
    #    vmin=vmin,
    #    vmax=vmax,
    #    levels=np.linspace(vmin, vmax, 100),
    #)
    cs = plt.imshow(array,interpolation = None,extent = [xmin,xmax,ymin,ymax], vmin = vmin, vmax = vmax,cmap = cmap)
    im = plt.scatter(
        x,
        y,
        s=None,
        c=v,
        marker=None,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        linewidths=0.8,
        edgecolors="black",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.clim(vmin, vmax)
    cbar = plt.colorbar(im, orientation="vertical")
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    return cs


def pixelplt_log_st(
    array,
    xmin,
    xmax,
    ymin,
    ymax,
    step,
    vmin,
    vmax,
    title,
    xlabel,
    ylabel,
    vlabel,
    cmap,
):
    """Pixel plot, reimplementation in Python of GSLIB pixelplt with Matplotlib
    methods (version for subplots, log scale).

    :param array: ndarray
    :param xmin: x axis minimum
    :param xmax: x axis maximum
    :param ymin: y axis minimum
    :param ymax: y axis maximum
    :param step: step
    :param vmin: TODO
    :param vmax: TODO
    :param title: title
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param vlabel: TODO
    :param cmap: colormap
    :return: QuadContourSet
    """
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step)
    )

    # Use dummy since scatter plot controls legend min and max appropriately
    # and contour does not!
    x = []
    y = []
    v = []

    color_int = np.r_[np.log(vmin): np.log(vmax): 0.5]
    color_int = np.exp(color_int)
    cs = plt.contourf(
        xx,
        yy,
        array,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        levels=color_int,
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    )
    im = plt.scatter(
        x,
        y,
        s=None,
        c=v,
        marker=None,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        linewidths=0.8,
        edgecolors="black",
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar(im, orientation="vertical")
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    return cs


def locpix(
    array,
    xmin,
    xmax,
    ymin,
    ymax,
    step,
    vmin,
    vmax,
    df,
    xcol,
    ycol,
    vcol,
    title,
    xlabel,
    ylabel,
    vlabel,
    cmap,
    fig_name,
):
    """Pixel plot and location map, reimplementation in Python of a GSLIB MOD
    with Matplotlib methods.

    :param array: ndarray
    :param xmin: x axis minimum
    :param xmax: x axis maximum
    :param ymin: y axis minimum
    :param ymax: y axis maximum
    :param step: step
    :param vmin: TODO
    :param vmax: TODO
    :param df: dataframe
    :param xcol: data for x axis
    :param ycol: data for y axis
    :param vcol: color, sequence, or sequence of color
    :param title: title
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param vlabel: TODO
    :param cmap: colormap
    :param fig_name: figure name
    :return: QuadContourSet
    """
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step)
    )

    plt.figure(figsize=(8, 6))
    #cs = plt.contourf(
    #    xx,
    #    yy,
    #    array,
    #    cmap=cmap,
    #    vmin=vmin,
    #    vmax=vmax,
    #    levels=np.linspace(vmin, vmax, 100),
    #)
    cs = plt.imshow(array,interpolation = None,extent = [xmin,xmax,ymin,ymax], vmin = vmin, vmax = vmax,cmap = cmap)
    plt.scatter(
        df[xcol],
        df[ycol],
        s=None,
        c=df[vcol],
        marker=None,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        linewidths=0.8,
        edgecolors="black",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cbar = plt.colorbar(orientation="vertical")
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    plt.savefig(fig_name + "." + image_type, dpi=dpi)
    plt.show()
    return cs


def locpix_st(
    array,
    xmin,
    xmax,
    ymin,
    ymax,
    step,
    vmin,
    vmax,
    df,
    xcol,
    ycol,
    vcol,
    title,
    xlabel,
    ylabel,
    vlabel,
    cmap,
):
    """Pixel plot and location map, reimplementation in Python of a GSLIB MOD
    with Matplotlib methods (version for subplots).

    :param array: ndarray
    :param xmin: x axis minimum
    :param xmax: x axis maximum
    :param ymin: y axis minimum
    :param ymax: y axis maximum
    :param step: step
    :param vmin: TODO
    :param vmax: TODO
    :param df: dataframe
    :param xcol: data for x axis
    :param ycol: data for y axis
    :param vcol: color, sequence, or sequence of color
    :param title: title
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param vlabel: TODO
    :param cmap: colormap
    :return: QuadContourSet
    """
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step)
    )

    #cs = plt.contourf(
    #    xx,
    #    yy,
    #    array,
    #    cmap=cmap,
    #    vmin=vmin,
    #    vmax=vmax,
    #    levels=np.linspace(vmin, vmax, 100),
    #)
    cs = plt.imshow(array,interpolation = None,extent = [xmin,xmax,ymin,ymax], vmin = vmin, vmax = vmax,cmap = cmap)
    plt.scatter(
        df[xcol],
        df[ycol],
        s=None,
        c=df[vcol],
        marker=None,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        linewidths=0.8,
        edgecolors="black",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cbar = plt.colorbar(orientation="vertical")
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    return cs


def locpix_log_st(
    array,
    xmin,
    xmax,
    ymin,
    ymax,
    step,
    vmin,
    vmax,
    df,
    xcol,
    ycol,
    vcol,
    title,
    xlabel,
    ylabel,
    vlabel,
    cmap,
):
    """Pixel plot and location map, reimplementation in Python of a GSLIB MOD
    with Matplotlib methods (version for subplots, log scale).

    :param array: ndarray
    :param xmin: x axis minimum
    :param xmax: x axis maximum
    :param ymin: y axis minimum
    :param ymax: y axis maximum
    :param step: step
    :param vmin: TODO
    :param vmax: TODO
    :param df: dataframe
    :param xcol: data for x axis
    :param ycol: data for y axis
    :param vcol: color, sequence, or sequence of color
    :param title: title
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param vlabel: TODO
    :param cmap: colormap
    :return: QuadContourSet
    """
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step)
    )

    color_int = np.r_[np.log(vmin): np.log(vmax): 0.5]
    color_int = np.exp(color_int)
    cs = plt.contourf(
        xx,
        yy,
        array,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        levels=color_int,
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    )
    plt.scatter(
        df[xcol],
        df[ycol],
        s=None,
        c=df[vcol],
        marker=None,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        linewidths=0.8,
        edgecolors="black",
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cbar = plt.colorbar(orientation="vertical")
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    return cs


def affine(array, tmean, tstdev):
    """Affine distribution correction reimplemented in Python with numpy
    methods.

    :param array: ndarray
    :param tmean: TODO
    :param tstdev: TODO
    :return: ndarray
    """
    mean = np.average(array)
    stdev = np.std(array)
    array = (tstdev / stdev) * (array - mean) + tmean
    return array


def nscore(x):
    """Normal score transform, wrapper for nscore from GSLIB (.exe must be
    available in PATH or working directory).

    :param x: ndarray
    :return: ndarray
    """
    nx = np.ma.size(x)
    ny = 1
    ndarray2GSLIB(x, "nscore.dat", "value")

    with open("nscore.par", "w") as f:
        f.write("                  Parameters for NSCORE                                    \n")
        f.write("                  *********************                                    \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETERS:                                                       \n")
        f.write("nscore.dat           -file with data                                       \n")
        f.write("1   0                    -  columns for variable and weight                \n")
        f.write("-1.0e21   1.0e21         -  trimming limits                                \n")
        f.write("0                        -1=transform according to specified ref. dist.    \n")
        f.write("../histsmth/histsmth.out -  file with reference dist.                      \n")
        f.write("1   2                    -  columns for variable and weight                \n")
        f.write("nscore.out               -file for output                                  \n")
        f.write("nscore.trn               -file for output transformation table             \n")

    os.system("nscore.exe nscore.par")
    y, name = GSLIB2ndarray("nscore.out", 1, nx, ny)
    return y


def make_variogram(
    nug,
    nst,
    it1,
    cc1,
    azi1,
    hmaj1,
    hmin1,
    it2=1,
    cc2=0,
    azi2=0,
    hmaj2=0,
    hmin2=0,
):
    """Make a dictionary of variogram parameters for application with spatial
    estimation and simulation.

    :param nug: TODO
    :param nst: TODO
    :param it1: TODO
    :param cc1: TODO
    :param azi1: TODO
    :param hmaj1: TODO
    :param hmin1: TODO
    :param it2: TODO
    :param cc2: TODO
    :param azi2: TODO
    :param hmaj2: TODO
    :param hmin2: TODO
    :return: TODO
    """
    if cc2 == 0:
        nst = 1
    var = dict(
        [
            ("nug", nug),
            ("nst", nst),
            ("it1", it1),
            ("cc1", cc1),
            ("azi1", azi1),
            ("hmaj1", hmaj1),
            ("hmin1", hmin1),
            ("it2", it2),
            ("cc2", cc2),
            ("azi2", azi2),
            ("hmaj2", hmaj2),
            ("hmin2", hmin2),
        ]
    )
    if nug + cc1 + cc2 != 1:
        print(
            "\x1b[0;30;41m make_variogram Warning: "
            "sill does not sum to 1.0, do not use in simulation \x1b[0m"
        )
    if (
        cc1 < 0
        or cc2 < 0
        or nug < 0
        or hmaj1 < 0
        or hmaj2 < 0
        or hmin1 < 0
        or hmin2 < 0
    ):
        print(
            "\x1b[0;30;41m make_variogram Warning: "
            "contributions and ranges must be all positive \x1b[0m"
        )
    if hmaj1 < hmin1 or hmaj2 < hmin2:
        print(
            "\x1b[0;30;41m make_variogram Warning: "
            "major range should be greater than minor range \x1b[0m"
        )
    return var


def gamv_2d(df, xcol, ycol, vcol, nlag, lagdist, azi, atol, bstand):
    """Irregularly sampled variogram, 2D wrapper for gam from GSLIB (.exe must
    be available in PATH or working directory).

    :param df: dataframe
    :param xcol: TODO
    :param ycol: TODO
    :param vcol: TODO
    :param nlag: TODO
    :param lagdist: TODO
    :param azi: TODO
    :param atol: TODO
    :param bstand: TODO
    :return: TODO
    """
    lag = []
    gamma = []
    npair = []

    df_ext = pd.DataFrame({"X": df[xcol], "Y": df[ycol], "Z": df[vcol]})
    Dataframe2GSLIB("gamv_out.dat", df_ext)

    with open("gamv.par", "w") as f:
        f.write("                  Parameters for GAMV                                      \n")
        f.write("                  *******************                                      \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETERS:                                                       \n")
        f.write("gamv_out.dat                    -file with data                            \n")
        f.write("1   2   0                         -   columns for X, Y, Z coordinates      \n")
        f.write("1   3   0                         -   number of variables,col numbers      \n")
        f.write("-1.0e21     1.0e21                -   trimming limits                      \n")
        f.write("gamv.out                          -file for variogram output               \n")
        f.write(str(nlag) + "                      -number of lags                          \n")
        f.write(str(lagdist) + "                       -lag separation distance                 \n")
        f.write(str(lagdist * 0.5) + "                   -lag tolerance                           \n")
        f.write("1                                 -number of directions                    \n")
        f.write(str(azi) + " " + str(atol) + " 99999.9 0.0  90.0  50.0  -azm,atol,bandh,dip,dtol,bandv \n")
        f.write(str(bstand) + "                    -standardize sills? (0=no, 1=yes)        \n")
        f.write("1                                 -number of variograms                    \n")
        f.write("1   1   1                         -tail var., head var., variogram type    \n")

    os.system("gamv.exe gamv.par")

    with open("gamv.out") as f:
        next(f)  # skip the first line

        for line in f:
            _, l, g, n, *_ = line.split()
            lag.append(float(l))
            gamma.append(float(g))
            npair.append(float(n))

    return lag, gamma, npair

def gamv_3d(df, xcol, ycol, zcol, vcol, nlag, lagdist,lag_tol, azi, atol, bandh, dip, dtol, bandv, isill):
    """Irregularly sampled variogram, 3D wrapper for gam from GSLIB (.exe must
    be available in PATH or working directory).

    :param df: dataframe
    :param xcol: TODO
    :param ycol: TODO
    :param zcol: TODO
    :param vcol: TODO
    :param nlag: TODO
    :param lagdist: TODO
    :param azi: TODO
    :param atol: TODO
    :param bstand: TODO
    :return: TODO
    """
    lag = []
    gamma = []
    npair = []

    df_ext = pd.DataFrame({"X": df[xcol], "Y": df[ycol], "Z": df[zcol],"Variable": df[vcol]})
    Dataframe2GSLIB("gamv_out.dat", df_ext)

    with open("gamv.par", "w") as f:
        f.write("                  Parameters for GAMV                                      \n")
        f.write("                  *******************                                      \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETERS:                                                       \n")
        f.write("gamv_out.dat                    -file with data                            \n")
        f.write("1   2   3                         -   columns for X, Y, Z coordinates      \n")
        f.write("1   4   0                         -   number of variables,col numbers      \n")
        f.write("-1.0e21     1.0e21                -   trimming limits                      \n")
        f.write("gamv.out                          -file for variogram output               \n")
        f.write(str(nlag) + "                      -number of lags                          \n")
        f.write(str(lagdist) + "                       -lag separation distance                 \n")
        f.write(str(lag_tol) + "                   -lag tolerance                           \n")
        f.write("1                                 -number of directions                    \n")
        f.write(str(azi) + " " + str(atol) + " " + str(bandh) + " " +str(dip) + " " + str(dtol) + " " + str(bandv)+ "  -azm,atol,bandh,dip,dtol,bandv \n")
        f.write(str(isill) + "                    -standardize sills? (0=no, 1=yes)        \n")
        f.write("1                                 -number of variograms                    \n")
        f.write("1   1   1                         -tail var., head var., variogram type    \n")

    os.system("gamv.exe gamv.par")

    with open("gamv.out") as f:
        next(f)  # skip the first line

        for line in f:
            _, l, g, n, *_ = line.split()
            lag.append(float(l))
            gamma.append(float(g))
            npair.append(float(n))

    return lag, gamma, npair

def varmapv_2d(
    df,
    xcol,
    ycol,
    vcol,
    nx,
    ny,
    lagdist,
    minpairs,
    vmax,
    bstand,
    title,
    vlabel,
    cmap,
    fig_name,
):
    """Irregular spaced data, 2D wrapper for varmap from GSLIB (.exe must be
    available in PATH or working directory).

    :param df: dataframe
    :param xcol: TODO
    :param ycol: TODO
    :param vcol: TODO
    :param nx: TODO
    :param ny: TODO
    :param lagdist: TODO
    :param minpairs: TODO
    :param vmax: TODO
    :param bstand: TODO
    :param title: TODO
    :param vlabel: TODO
    :param cmap: colormap
    :param fig_name: figure name
    :return: TODO
    """
    df_ext = pd.DataFrame(
        {"X": df[xcol], "Y": df[ycol], "Z": df[vcol]} # TODO unknown function
    )
    Dataframe2GSLIB("varmap_out.dat", df_ext)

    with open("varmap.par", "w") as f:
        f.write("              Parameters for VARMAP                                        \n")
        f.write("              *********************                                        \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETERS:                                                       \n")
        f.write("varmap_out.dat          -file with data                                    \n")
        f.write("1   3                        -   number of variables: column numbers       \n")
        f.write("-1.0e21     1.0e21           -   trimming limits                           \n")
        f.write("0                            -1=regular grid, 0=scattered values           \n")
        f.write(" 50   50    1                -if =1: nx,     ny,   nz                      \n")
        f.write("1.0  1.0  1.0                -       xsiz, ysiz, zsiz                      \n")
        f.write("1   2   0                    -if =0: columns for x,y, z coordinates        \n")
        f.write("varmap.out                   -file for variogram output                    \n")
        f.write(str(nx) + " " + str(ny) + " 0 " + "-nxlag, nylag, nzlag                     \n")
        f.write(str(lagdist) + " " + str(lagdist) + " 1.0              -dxlag, dylag, dzlag \n")
        f.write(str(minpairs) + "             -minimum number of pairs                      \n")
        f.write(str(bstand) + "               -standardize sill? (0=no, 1=yes)              \n")
        f.write("1                            -number of variograms                         \n")
        f.write("1   1   1                    -tail, head, variogram type                   \n")

    os.system("varmap.exe varmap.par")
    nnx = nx * 2 + 1
    nny = ny * 2 + 1
    varmap_, name = GSLIB2ndarray("varmap.out", 0, nnx, nny)

    xmax = (float(nx) + 0.5) * lagdist
    xmin = -1 * xmax
    ymax = (float(ny) + 0.5) * lagdist
    ymin = -1 * ymax
    pixelplt(
        varmap_,
        xmin,
        xmax,
        ymin,
        ymax,
        lagdist,
        0,
        vmax,
        title,
        "X",
        "Y",
        vlabel,
        cmap,
        fig_name
    )
    return varmap_


def varmap(
    array,
    nx,
    ny,
    hsiz,
    nlagx,
    nlagy,
    minpairs,
    vmax,
    bstand,
    title,
    vlabel,
    cmap,
    fig_name,
):
    """Regular spaced data, 2D wrapper for varmap from GSLIB (.exe must be
    available in PATH or working directory).

    :param array: ndarray
    :param nx: TODO
    :param ny: TODO
    :param hsiz: TODO
    :param nlagx: TODO
    :param nlagy: TODO
    :param minpairs: TODO
    :param vmax: TODO
    :param bstand: TODO
    :param title: title
    :param vlabel: TODO
    :param cmap: colormap
    :param fig_name: figure name
    :return: TODO
    """
    ndarray2GSLIB(array, "varmap_out.dat", "gam.dat")

    with open("varmap.par", "w") as f:
        f.write("              Parameters for VARMAP                                        \n")
        f.write("              *********************                                        \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETERS:                                                       \n")
        f.write("varmap_out.dat          -file with data                                    \n")
        f.write("1   1                        -   number of variables: column numbers       \n")
        f.write("-1.0e21     1.0e21           -   trimming limits                           \n")
        f.write("1                            -1=regular grid, 0=scattered values           \n")
        f.write(str(nx) + " " + str(ny) + " 1  -if =1: nx,     ny,   nz                     \n")
        f.write(str(hsiz) + " " + str(hsiz) + " 1.0  - xsiz, ysiz, zsiz                     \n")
        f.write("1   2   0                    -if =0: columns for x,y, z coordinates        \n")
        f.write("varmap.out                   -file for variogram output                    \n")
        f.write(str(nlagx) + " " + str(nlagy) + " 0 " + "-nxlag, nylag, nzlag               \n")
        f.write(str(hsiz) + " " + str(hsiz) + " 1.0              -dxlag, dylag, dzlag       \n")
        f.write(str(minpairs) + "             -minimum number of pairs                      \n")
        f.write(str(bstand) + "               -standardize sill? (0=no, 1=yes)              \n")
        f.write("1                            -number of variograms                         \n")
        f.write("1   1   1                    -tail, head, variogram type                   \n")

    os.system("varmap.exe varmap.par")
    nnx = nlagx * 2 + 1
    nny = nlagy * 2 + 1
    varmap_, name = GSLIB2ndarray("varmap.out", 0, nnx, nny)

    xmax = (float(nlagx) + 0.5) * hsiz
    xmin = -1 * xmax
    ymax = (float(nlagy) + 0.5) * hsiz
    ymin = -1 * ymax
    pixelplt(
        varmap_,
        xmin,
        xmax,
        ymin,
        ymax,
        hsiz,
        0,
        vmax,
        title,
        "X",
        "Y",
        vlabel,
        cmap,
        fig_name
    )
    return varmap_


def vmodel(
    nlag,
    step,
    azi,
    nug,
    nst,
    tstr1,
    c1,
    azi1,
    rmaj1,
    rmin1,
    tstr2=1,
    c2=0,
    azi2=0,
    rmaj2=0,
    rmin2=0,
):
    """Variogram model, 2D wrapper for vmodel from GSLIB (.exe must be
    available in PATH or working directory).

    :param nlag: TODO
    :param step: TODO
    :param azi: TODO
    :param nug: TODO
    :param nst: TODO
    :param tstr1: TODO
    :param c1: TODO
    :param azi1: TODO
    :param rmaj1: TODO
    :param rmin1: TODO
    :param tstr2: TODO
    :param c2: TODO
    :param azi2: TODO
    :param rmaj2: TODO
    :param rmin2: TODO
    :return: TODO
    """
    lag = []
    gamma = []

    with open("vmodel.par", "w") as f:
        f.write("                                                                           \n")
        f.write("                  Parameters for VMODEL                                    \n")
        f.write("                  *********************                                    \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETERS:                                                       \n")
        f.write("vmodel.var                   -file for variogram output                    \n")
        f.write("1 " + str(nlag) + "          -number of directions and lags                \n")
        f.write(str(azi) + " 0.0 " + str(step) + " -azm, dip, lag distance                  \n")
        f.write(str(nst) + " " + str(nug) + " -nst, nugget effect                           \n")
        f.write(str(tstr1) + " " + str(c1) + " " + str(azi1) + " 0.0   0.0   0.0 -it,cc,ang1,ang2,ang3 \n")
        f.write(str(rmaj1) + " " + str(rmin1) + " 0.0 -a_hmax, a_hmin, a_vert               \n")
        f.write(str(tstr2) + " " + str(c2) + " " + str(azi2) + " 0.0   0.0   0.0 -it,cc,ang1,ang2,ang3 \n")
        f.write(str(rmaj2) + " " + str(rmin2) + " 0.0 -a_hmax, a_hmin, a_vert               \n")

    os.system("vmodel.exe vmodel.par")

    with open("vmodel.var") as f:
        next(f)  # skip the first line

        for line in f:
            _, l, g, *_ = line.split()
            lag.append(float(l))
            gamma.append(float(g))

    return lag, gamma


def declus(df, xcol, ycol, vcol, cmin, cmax, cnum, bmin):
    """Cell-based declustering, 2D wrapper for declus from GSLIB (.exe must be
    available in PATH or working directory).

    :param df: dataframe
    :param xcol: TODO
    :param ycol: TODO
    :param vcol: TODO
    :param cmin: TODO
    :param cmax: TODO
    :param cnum: TODO
    :param bmin: TODO
    :return: TODO
    """
    nrow = len(df)
    weights = []

    with open("declus_out.dat", "w") as f:
        f.write("declus_out.dat" + "\n")
        f.write("3" + "\n")
        f.write("x" + "\n")
        f.write("y" + "\n")
        f.write("value" + "\n")

        for irow in range(0, nrow):
            f.write(
                str(df.iloc[irow][xcol]) + " " +
                str(df.iloc[irow][ycol]) + " " +
                str(df.iloc[irow][vcol]) + " \n"
            )

    with open("declus.par", "w") as f:
        f.write("                  Parameters for DECLUS                                    \n")
        f.write("                  *********************                                    \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETERS:                                                       \n")
        f.write("declus_out.dat           -file with data                                   \n")
        f.write("1   2   0   3               -  columns for X, Y, Z, and variable           \n")
        f.write("-1.0e21     1.0e21          -  trimming limits                             \n")
        f.write("declus.sum                  -file for summary output                       \n")
        f.write("declus.out                  -file for output with data & weights           \n")
        f.write("1.0   1.0                   -Y and Z cell anisotropy (Ysize=size*Yanis)    \n")
        f.write(str(bmin) + "                -0=look for minimum declustered mean (1=max)   \n")
        f.write(str(cnum) + " " + str(cmin) + " " + str(cmax) + " -number of cell sizes, min size, max size      \n")
        f.write("5                           -number of origin offsets                      \n")

    os.system("declus.exe declus.par")
    df = GSLIB2Dataframe("declus.out")
    for irow in range(nrow):
        weights.append(df.iloc[irow, 3])

    return weights


def sgsim_uncond(nreal, nx, ny, hsiz, seed, var, output_file):
    """Sequential Gaussian simulation, 2D unconditional wrapper for sgsim from
    GSLIB (.exe must be available in PATH or working directory).

    :param nreal: TODO
    :param nx: TODO
    :param ny: TODO
    :param hsiz: TODO
    :param seed: TODO
    :param var: TODO
    :param output_file: output file
    :return: TODO
    """
    nug = var["nug"]
    nst = var["nst"]
    it1 = var["it1"]
    cc1 = var["cc1"]
    azi1 = var["azi1"]
    hmaj1 = var["hmaj1"]
    hmin1 = var["hmin1"]
    it2 = var["it2"]
    cc2 = var["cc2"]
    azi2 = var["azi2"]
    hmaj2 = var["hmaj2"]
    hmin2 = var["hmin2"]
    max_range = max(hmaj1, hmaj2)
    hmn = hsiz * 0.5
    hctab = int(max_range / hsiz) * 2 + 1

    with open("sgsim.par", "w") as f:
        f.write("              Parameters for SGSIM                                         \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("none                          -file with data                              \n")
        f.write("1  2  0  3  5  0              -  columns for X,Y,Z,vr,wt,sec.var.          \n")
        f.write("-1.0e21 1.0e21                -  trimming limits                           \n")
        f.write("0                             -transform the data (0=no, 1=yes)            \n")
        f.write("none.trn                      -  file for output trans table               \n")
        f.write("1                             -  consider ref. dist (0=no, 1=yes)          \n")
        f.write("none.dat                      -  file with ref. dist distribution          \n")
        f.write("1  0                          -  columns for vr and wt                     \n")
        f.write("-4.0    4.0                   -  zmin,zmax(tail extrapolation)             \n")
        f.write("1      -4.0                   -  lower tail option, parameter              \n")
        f.write("1       4.0                   -  upper tail option, parameter              \n")
        f.write("0                             -debugging level: 0,1,2,3                    \n")
        f.write("nonw.dbg                      -file for debugging output                   \n")
        f.write(str(output_file) + "           -file for simulation output                  \n")
        f.write(str(nreal) + "                 -number of realizations to generate          \n")
        f.write(str(nx) + " " + str(hmn) + " " + str(hsiz) + "                              \n")
        f.write(str(ny) + " " + str(hmn) + " " + str(hsiz) + "                              \n")
        f.write("1 0.0 1.0                     - nz zmn zsiz                                \n")
        f.write(str(seed) + "                  -random number seed                          \n")
        f.write("0     8                       -min and max original data for sim           \n")
        f.write("12                            -number of simulated nodes to use            \n")
        f.write("0                             -assign data to nodes (0=no, 1=yes)          \n")
        f.write("1     3                       -multiple grid search (0=no, 1=yes),num      \n")
        f.write("0                             -maximum data per octant (0=not used)        \n")
        f.write(str(max_range) + " " + str(max_range) + " 1.0 -maximum search  (hmax,hmin,vert) \n")
        f.write(str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n")
        f.write(str(hctab) + " " + str(hctab) + " 1 -size of covariance lookup table        \n")
        f.write("0     0.60   1.0              -ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC        \n")
        f.write("none.dat                      -  file with LVM, EXDR, or COLC variable     \n")
        f.write("4                             -  column for secondary variable             \n")
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(" " + str(hmaj1) + " " + str(hmin1) + " 1.0 - a_hmax, a_hmin, a_vert        \n")
        f.write(str(it2) + " " + str(cc2) + " " + str(azi2) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(" " + str(hmaj2) + " " + str(hmin2) + " 1.0 - a_hmax, a_hmin, a_vert        \n")

    os.system("sgsim.exe sgsim.par")
    sim_array = GSLIB2ndarray(output_file, 0, nx, ny)
    return sim_array[0]


def kb2d(df, xcol, ycol, vcol, nx, ny, hsiz, var, output_file):
    """Kriging estimation, 2D wrapper for kb2d from GSLIB (.exe must be
    available in PATH or working directory).

    :param df: dataframe
    :param xcol: TODO
    :param ycol: TODO
    :param vcol: TODO
    :param nx: TODO
    :param ny: TODO
    :param hsiz: TODO
    :param var: TODO
    :param output_file: output file
    :return: TODO
    """
    df_temp = pd.DataFrame({"X": df[xcol], "Y": df[ycol], "Var": df[vcol]})
    Dataframe2GSLIB("data_temp.dat", df_temp)

    nug = var["nug"]
    nst = var["nst"]
    it1 = var["it1"]
    cc1 = var["cc1"]
    azi1 = var["azi1"]
    hmaj1 = var["hmaj1"]
    hmin1 = var["hmin1"]
    it2 = var["it2"]
    cc2 = var["cc2"]
    azi2 = var["azi2"]
    hmaj2 = var["hmaj2"]
    hmin2 = var["hmin2"]
    max_range = max(hmaj1, hmaj2)
    hmn = hsiz * 0.5

    with open("kb2d.par", "w") as f:
        f.write("              Parameters for KB2D                                          \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("data_temp.dat                         -file with data                      \n")
        f.write("1  2  3                               -  columns for X,Y,vr                \n")
        f.write("-1.0e21   1.0e21                      -   trimming limits                  \n")
        f.write("0                                     -debugging level: 0,1,2,3            \n")
        f.write("none.dbg                              -file for debugging output           \n")
        f.write(str(output_file) + "                   -file for kriged output              \n")
        f.write(str(nx) + " " + str(hmn) + " " + str(hsiz) + "                              \n")
        f.write(str(ny) + " " + str(hmn) + " " + str(hsiz) + "                              \n")
        f.write("1    1                                -x and y block discretization        \n")
        f.write("1    30                               -min and max data for kriging        \n")
        f.write(str(max_range) + "                     -maximum search radius               \n")
        f.write("1    -9999.9                          -0=SK, 1=OK,  (mean if SK)           \n")
        f.write(str(nst) + " " + str(nug) + "          -nst, nugget effect                  \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + " " + str(hmaj1) + " " + str(hmin1) + " -it, c ,azm ,a_max ,a_min \n")
        f.write(str(it2) + " " + str(cc2) + " " + str(azi2) + " " + str(hmaj2) + " " + str(hmin2) + " -it, c ,azm ,a_max ,a_min \n")

    os.system("kb2d.exe kb2d.par")
    est_array = GSLIB2ndarray(output_file, 0, nx, ny)
    var_array = GSLIB2ndarray(output_file, 1, nx, ny)
    return est_array[0], var_array[0]


def sgsim(nreal, df, xcol, ycol, vcol, nx, ny, hsiz, seed, var, output_file):
    """Sequential Gaussian simulation, 2D wrapper for sgsim from GSLIB (.exe
    must be available in PATH or working directory).

    :param nreal: TODO
    :param df: dataframe
    :param xcol: TODO
    :param ycol: TODO
    :param vcol: TODO
    :param nx: TODO
    :param ny: TODO
    :param hsiz: TODO
    :param seed: TODO
    :param var: TODO
    :param output_file: output file
    :return: TODO
    """
    x = df[xcol]
    y = df[ycol]
    v = df[vcol]
    var_min = v.values.min()
    var_max = v.values.max()
    df_temp = pd.DataFrame({"X": x, "Y": y, "Var": v})
    Dataframe2GSLIB("data_temp.dat", df_temp)

    nug = var["nug"]
    nst = var["nst"]
    it1 = var["it1"]
    cc1 = var["cc1"]
    azi1 = var["azi1"]
    hmaj1 = var["hmaj1"]
    hmin1 = var["hmin1"]
    it2 = var["it2"]
    cc2 = var["cc2"]
    azi2 = var["azi2"]
    hmaj2 = var["hmaj2"]
    hmin2 = var["hmin2"]
    max_range = max(hmaj1, hmaj2)
    hmn = hsiz * 0.5
    hctab = int(max_range / hsiz) * 2 + 1

    with open("sgsim.par", "w") as f:
        f.write("              Parameters for SGSIM                                         \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("data_temp.dat                 -file with data                              \n")
        f.write("1  2  0  3  0  0              -  columns for X,Y,Z,vr,wt,sec.var.          \n")
        f.write("-1.0e21 1.0e21                -  trimming limits                           \n")
        f.write("1                             -transform the data (0=no, 1=yes)            \n")
        f.write("none.trn                      -  file for output trans table               \n")
        f.write("0                             -  consider ref. dist (0=no, 1=yes)          \n")
        f.write("none.dat                      -  file with ref. dist distribution          \n")
        f.write("1  0                          -  columns for vr and wt                     \n")
        f.write(str(var_min) + " " + str(var_max) + "   zmin,zmax(tail extrapolation)       \n")
        f.write("1   " + str(var_min) + "      -  lower tail option, parameter              \n")
        f.write("1   " + str(var_max) + "      -  upper tail option, parameter              \n")
        f.write("0                             -debugging level: 0,1,2,3                    \n")
        f.write("nonw.dbg                      -file for debugging output                   \n")
        f.write(str(output_file) + "           -file for simulation output                  \n")
        f.write(str(nreal) + "                 -number of realizations to generate          \n")
        f.write(str(nx) + " " + str(hmn) + " " + str(hsiz) + "                              \n")
        f.write(str(ny) + " " + str(hmn) + " " + str(hsiz) + "                              \n")
        f.write("1 0.0 1.0                     - nz zmn zsiz                                \n")
        f.write(str(seed) + "                  -random number seed                          \n")
        f.write("0     8                       -min and max original data for sim           \n")
        f.write("12                            -number of simulated nodes to use            \n")
        f.write("0                             -assign data to nodes (0=no, 1=yes)          \n")
        f.write("1     3                       -multiple grid search (0=no, 1=yes),num      \n")
        f.write("0                             -maximum data per octant (0=not used)        \n")
        f.write(str(max_range) + " " + str(max_range) + " 1.0 -maximum search  (hmax,hmin,vert) \n")
        f.write(str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n")
        f.write(str(hctab) + " " + str(hctab) + " 1 -size of covariance lookup table        \n")
        f.write("0     0.60   1.0              -ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC        \n")
        f.write("none.dat                      -  file with LVM, EXDR, or COLC variable     \n")
        f.write("4                             -  column for secondary variable             \n")
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(" " + str(hmaj1) + " " + str(hmin1) + " 1.0 - a_hmax, a_hmin, a_vert        \n")
        f.write(str(it2) + " " + str(cc2) + " " + str(azi2) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(" " + str(hmaj2) + " " + str(hmin2) + " 1.0 - a_hmax, a_hmin, a_vert        \n")

    os.system("sgsim.exe sgsim.par")
    sim_array = GSLIB2ndarray(output_file, 0, nx, ny)
    return sim_array[0]


def cosgsim_uncond(nreal, nx, ny, hsiz, seed, var, sec, correl, output_file):
    """Sequential Gaussian simulation, 2D unconditional wrapper for sgsim from
    GSLIB (.exe must be available in PATH or working directory).

    :param nreal: TODO
    :param nx: TODO
    :param ny: TODO
    :param hsiz: TODO
    :param seed: TODO
    :param var: TODO
    :param sec: TODO
    :param correl: TODO
    :param output_file: output file
    :return: TODO
    """
    nug = var["nug"]
    nst = var["nst"]
    it1 = var["it1"]
    cc1 = var["cc1"]
    azi1 = var["azi1"]
    hmaj1 = var["hmaj1"]
    hmin1 = var["hmin1"]
    it2 = var["it2"]
    cc2 = var["cc2"]
    azi2 = var["azi2"]
    hmaj2 = var["hmaj2"]
    hmin2 = var["hmin2"]

    max_range = max(hmaj1, hmaj2)
    hmn = hsiz * 0.5
    hctab = int(max_range / hsiz) * 2 + 1

    ndarray2GSLIB(sec, "sec.dat", "sec_dat")

    with open("sgsim.par", "w") as f:
        f.write("              Parameters for SGSIM                                         \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("none                          -file with data                              \n")
        f.write("1  2  0  3  5  0              -  columns for X,Y,Z,vr,wt,sec.var.          \n")
        f.write("-1.0e21 1.0e21                -  trimming limits                           \n")
        f.write("0                             -transform the data (0=no, 1=yes)            \n")
        f.write("none.trn                      -  file for output trans table               \n")
        f.write("0                             -  consider ref. dist (0=no, 1=yes)          \n")
        f.write("none.dat                      -  file with ref. dist distribution          \n")
        f.write("1  0                          -  columns for vr and wt                     \n")
        f.write("-4.0    4.0                   -  zmin,zmax(tail extrapolation)             \n")
        f.write("1      -4.0                   -  lower tail option, parameter              \n")
        f.write("1       4.0                   -  upper tail option, parameter              \n")
        f.write("0                             -debugging level: 0,1,2,3                    \n")
        f.write("nonw.dbg                      -file for debugging output                   \n")
        f.write(str(output_file) + "           -file for simulation output                  \n")
        f.write(str(nreal) + "                 -number of realizations to generate          \n")
        f.write(str(nx) + " " + str(hmn) + " " + str(hsiz) + "                              \n")
        f.write(str(ny) + " " + str(hmn) + " " + str(hsiz) + "                              \n")
        f.write("1 0.0 1.0                     - nz zmn zsiz                                \n")
        f.write(str(seed) + "                  -random number seed                          \n")
        f.write("0     8                       -min and max original data for sim           \n")
        f.write("12                            -number of simulated nodes to use            \n")
        f.write("0                             -assign data to nodes (0=no, 1=yes)          \n")
        f.write("1     3                       -multiple grid search (0=no, 1=yes),num      \n")
        f.write("0                             -maximum data per octant (0=not used)        \n")
        f.write(str(max_range) + " " + str(max_range) + " 1.0 -maximum search  (hmax,hmin,vert) \n")
        f.write(str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n")
        f.write(str(hctab) + " " + str(hctab) + " 1 -size of covariance lookup table        \n")
        f.write("4 " + str(correl) + " 1.0     -ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC        \n")
        f.write("sec.dat                       -  file with LVM, EXDR, or COLC variable     \n")
        f.write("1                             -  column for secondary variable             \n")
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + " 0.0 0.0 -it,cc,ang1,ang2,ang3 \n")
        f.write(" " + str(hmaj1) + " " + str(hmin1) + " 1.0 - a_hmax, a_hmin, a_vert        \n")
        f.write(str(it2) + " " + str(cc2) + " " + str(azi2) + " 0.0 0.0 -it,cc,ang1,ang2,ang3 \n")
        f.write(" " + str(hmaj2) + " " + str(hmin2) + " 1.0 - a_hmax, a_hmin, a_vert        \n")

    os.system("sgsim.exe sgsim.par")
    sim_array = GSLIB2ndarray(output_file, 0, nx, ny)
    return sim_array[0]


def sample(array, xmin, ymin, step, name, df, xcol, ycol):
    """Sample 2D model with provided X and Y and append to DataFrame.

    :param array: ndarray
    :param xmin: TODO
    :param ymin: TODO
    :param step: TODO
    :param name: TODO
    :param df: dataframe
    :param xcol: TODO
    :param ycol: TODO
    :return: dataframe
    """
    if array.ndim != 2:
        raise ValueError("Array must be 2D")

    ny, nx = array.shape

    v = []
    nsamp = len(df)
    for isamp in range(nsamp):
        x = df.iloc[isamp][xcol]
        y = df.iloc[isamp][ycol]
        iy = min(ny - int((y - ymin) / step) - 1, ny - 1)
        ix = min(int((x - xmin) / step), nx - 1)
        v.append(array[iy, ix])
    df[name] = v
    return df


def gkern(kernlen=21, std=3):
    """Return a 2D Gaussian kernel array.

    Stack Overflow solution from Teddy Hartano.
    """
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def regular_sample(array, xmin, xmax, ymin, ymax, step, mx, my, nx, ny, name):
    """Extract regular spaced samples from a 2D spatial model.

    :param array: ndarray
    :param xmin: TODO
    :param xmax: TODO
    :param ymin: TODO
    :param ymax: TODO
    :param step: TODO
    :param mx: TODO
    :param my: TODO
    :param nx: TODO
    :param ny: TODO
    :param name: TODO
    :return: dataframe
    """
    x = []
    y = []
    v = []

    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step)
    )
    iiy = 0
    for iy in range(ny):
        if iiy >= my:
            iix = 0
            for ix in range(nx):
                if iix >= mx:
                    x.append(xx[ix, iy])
                    y.append(yy[ix, iy])
                    v.append(array[ix, iy])
                    iix = 0
                    iiy = 0
                iix = iix + 1
        iiy = iiy + 1
    df = pd.DataFrame(np.c_[x, y, v], columns=["X", "Y", name])
    return df


def random_sample(array, xmin, xmax, ymin, ymax, step, nsamp, name):
    """Extract random samples from a 2D spatial model.

    :param array: ndarray
    :param xmin: TODO
    :param xmax: TODO
    :param ymin: TODO
    :param ymax: TODO
    :param step: TODO
    :param nsamp: TODO
    :param name: TODO
    :return: dataframe
    """
    x = []
    y = []
    v = []

    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, step), np.arange(ymax - 1, ymin - 1, -1 * step)
    )
    ny, nx = xx.shape

    sample_index = rand.sample(range(nx * ny), nsamp)
    for isamp in range(nsamp):
        iy = int(sample_index[isamp] / ny)
        ix = sample_index[isamp] - iy * nx
        x.append(xx[iy, ix])
        y.append(yy[iy, ix])
        v.append(array[iy, ix])
    df = pd.DataFrame(np.c_[x, y, v], columns=["X", "Y", name])
    return df


def DataFrame2ndarray(df, xcol, ycol, vcol, xmin, xmax, ymin, ymax, step):
    """Take spatial data from a DataFrame and make a sparse ndarray (NaN where
    no data in cell).

    :param df: dataframe
    :param xcol: TODO
    :param ycol: TODO
    :param vcol: TODO
    :param xmin: TODO
    :param xmax: TODO
    :param ymin: TODO
    :param ymax: TODO
    :param step: TODO
    :return: ndarray
    """
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, step), np.arange(ymax - 1, ymin - 1, -1 * step)
    )
    ny, nx = xx.shape

    array = np.full((ny, nx), np.nan)
    nsamp = len(df)

    for isamp in range(0, nsamp):
        iy = min(ny - 1, ny - int((df.iloc[isamp][ycol] - ymin) / step) - 1)
        ix = min(nx - 1, int((df.iloc[isamp][xcol] - xmin) / step))
        array[iy, ix] = df.iloc[isamp][vcol]
    return array


def make_variogram_3D(
    nug,
    nst,
    it1,
    cc1,
    azi1,
    dip1,
    hmax1,
    hmed1,
    hmin1,
    it2=1,
    cc2=0,
    azi2=0,
    dip2=0,
    hmax2=0,
    hmed2=0,
    hmin2=0,
):
    """Make a dictionary of variogram parameters for application with spatial
    estimation and simulation.

    :param nug: Nugget constant (isotropic)
    :param nst: Number of structures (up to 2)
    :param it1: Structure of 1st variogram (1: Gaussian, 2: Exponential, 3: Spherical)
    :param cc1: Contribution of 2nd variogram
    :param azi1: Azimuth of 1st variogram
    :param dip1: Dip of 1st variogram
    :param hmax1: Range in major direction (Horizontal)
    :param hmed1: Range in minor direction (Horizontal)
    :param hmin1: Range in vertical direction
    :param it2: Structure of 2nd variogram (1: Gaussian, 2: Exponential, 3: Spherical)
    :param cc2: Contribution of 2nd variogram
    :param azi2: Azimuth of 2nd variogram
    :param dip1: Dip of 2nd variogram
    :param hmax2: Range in major direction (Horizontal)
    :param hmed2: Range in minor direction (Horizontal)
    :param hmin2: Range in vertical direction
    :return: TODO
    """
    if cc2 == 0:
        nst = 1
    var = dict(
        [
            ("nug", nug),
            ("nst", nst),
            ("it1", it1),
            ("cc1", cc1),
            ("azi1", azi1),
            ("dip1", dip1),
            ("hmax1", hmax1),
            ("hmed1", hmed1),
            ("hmin1", hmin1),
            ("it2", it2),
            ("cc2", cc2),
            ("azi2", azi2),
            ("dip2", dip2),
            ("hmax2", hmax2),
            ("hmed2", hmed2),
            ("hmin2", hmin2),
        ]
    )
    if nug + cc1 + cc2 != 1:
        print(
            "\x1b[0;30;41m make_variogram Warning: "
            "sill does not sum to 1.0, do not use in simulation \x1b[0m"
        )
    if (
        cc1 < 0
        or cc2 < 0
        or nug < 0
        or hmax1 < 0
        or hmax2 < 0
        or hmin1 < 0
        or hmin2 < 0
    ):
        print(
            "\x1b[0;30;41m make_variogram Warning: "
            "contributions and ranges must be all positive \x1b[0m"
        )
    if hmax1 < hmed1 or hmax2 < hmed2:
        print(
            "\x1b[0;30;41m make_variogram Warning: "
            "major range should be greater than minor range \x1b[0m"
        )
    return var


def sgsim_3D(nreal, df, xcol, ycol, zcol, vcol, nx, ny, nz, hsiz, vsiz, seed, var, output_file):
    """Sequential Gaussian simulation, 2D wrapper for sgsim from GSLIB (.exe
    must be available in PATH or working directory).

    :param nreal: TODO
    :param df: dataframe
    :param xcol: TODO
    :param ycol: TODO
    :param vcol: TODO
    :param nx: TODO
    :param ny: TODO
    :param hsiz: TODO
    :param seed: TODO
    :param var: TODO
    :param output_file: output file
    :return: TODO
    """
    x = df[xcol]
    y = df[ycol]
    z = df[zcol]
    v = df[vcol]
    var_min = v.values.min()
    var_max = v.values.max()
    df_temp = pd.DataFrame({"X": x, "Y": y, "Z": z, "Var": v})
    Dataframe2GSLIB("data_temp.dat", df_temp)

    nug = var["nug"]
    nst = var["nst"]
    it1 = var["it"][0]
    cc1 = var["cc"][0]
    azi1 = var["azi"][0]
    dip1 = var["dip"][0] 
    hmax1 = var["hmaj"][0]
    hmin1 = var["hmin"][0]
    hvert1 = var["hvert"][0]
    it2 = var["it"][1]
    cc2 = var["cc"][1]
    azi2 = var["azi"][1]
    dip2 = var["dip"][1] 
    hmax2 = var["hmaj"][1]
    hmin2 = var["hmin"][1]
    hvert2 = var["hvert"][1]
    max_range = max(hmax1, hmax2)
    max_range_v = max(hvert1, hvert2)
    hmn = hsiz * 0.5
    zmn = vsiz * 0.5
    hctab_h = int(max_range / hsiz) * 2 + 1
    hctab_v = int(max_range_v / vsiz) * 2 + 1

    with open("sgsim.par", "w") as f:
        f.write("              Parameters for SGSIM                                         \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("data_temp.dat                 -file with data                              \n")
        f.write("1  2  3  4  0  0              -  columns for X,Y,Z,vr,wt,sec.var.          \n")
        f.write("-1.0e21 1.0e21                -  trimming limits                           \n")
        f.write("1                             -transform the data (0=no, 1=yes)            \n")
        f.write("none.trn                      -  file for output trans table               \n")
        f.write("0                             -  consider ref. dist (0=no, 1=yes)          \n")
        f.write("none.dat                      -  file with ref. dist distribution          \n")
        f.write("1  0                          -  columns for vr and wt                     \n")
        f.write(str(var_min) + " " + str(var_max) + "   zmin,zmax(tail extrapolation)       \n")
        f.write("1   " + str(var_min) + "      -  lower tail option, parameter              \n")
        f.write("1   " + str(var_max) + "      -  upper tail option, parameter              \n")
        f.write("0                             -debugging level: 0,1,2,3                    \n")
        f.write("nonw.dbg                      -file for debugging output                   \n")
        f.write(str(output_file) + "           -file for simulation output                  \n")
        f.write(str(nreal) + "                 -number of realizations to generate          \n")
        f.write(str(nx) + " " + str(hmn) + " " + str(hsiz) + "                              \n")
        f.write(str(ny) + " " + str(hmn) + " " + str(hsiz) + "                              \n")
        f.write(str(nz) + " " + str(zmn) + " " + str(vsiz) + "                              \n")
        f.write(str(seed) + "                  -random number seed                          \n")
        f.write("0     8                       -min and max original data for sim           \n")
        f.write("12                            -number of simulated nodes to use            \n")
        f.write("1                             -assign data to nodes (0=no, 1=yes)          \n")
        f.write("1     3                       -multiple grid search (0=no, 1=yes),num      \n")
        f.write("0                             -maximum data per octant (0=not used)        \n")
        f.write(str(max_range) + " " + str(max_range) +" "+ str(max_range_v) + " -maximum search  (hmax,hmin,vert) \n")
        f.write(str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n")
        f.write(str(hctab_h) + " " + str(hctab_h) + " " + str(hctab_v) + " -size of covariance lookup table        \n")
        f.write("1     0.60   1.0              - ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC        \n")
        f.write("none.dat                      -  file with LVM, EXDR, or COLC variable     \n")
        f.write("4                             -  column for secondary variable             \n")
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + "  " + str(dip1) +" 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(" " + str(hmax1) + " 	" + str(hmed1) +  "		" + str(hmin1) + "  - a_hmax, a_hmin, a_vert        \n")
        f.write(str(it2) + " " + str(cc2) + " 	" + str(azi2) + "		" + str(dip2) +" 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(" " + str(hmax2) + " " + str(hmed2) +  " " +str(hmin2) + " - a_hmax, a_hmin, a_vert        \n")

    os.system("sgsim.exe sgsim.par")
    sim_array = GSLIB2ndarray_3D(output_file, 0, nreal, nx, ny, nz)
    return sim_array[0]
