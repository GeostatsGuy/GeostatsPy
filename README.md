  
<p>
    <img src="https://github.com/GeostatsGuy/GeostatsPy/blob/master/geostatspy_logo.png?raw=true" width="200" height="200" />
</p>

[![Documentation Status](https://readthedocs.org/projects/geostatspy/badge/?version=latest)](https://geostatspy.readthedocs.io/en/latest/?badge=latest)

# Cite As:

Pyrcz, M.J., Jo. H., Kupenko, A., Liu, W., Gigliotti, A.E., Salomaki, T., and Santos, J., 2021, GeostatsPy Python Package, PyPI, Python Package Index, https://pypi.org/project/geostatspy/.

# GeostatsPy Package 

The GeostatsPy Package brings GSLIB: Geostatistical Library (Deutsch and Journel, 1998) functions to Python. GSLIB is a practical and extremely robust set of code for building spatial modeling workflows. 

I created the GeostatsPy Package to support my students in my **Data Analytics**, **Geostatistics** and **Machine Learning** courses. I find my students benefit from hands-on opportunities, in fact it is hard to imagine teaching these topics without providing the opportunity to handle the numerical methods and build workflows. Last year, I tried to have them use the original FORTRAN executables and even with support and worked out examples, it was an uphill battle. In addition, all my students and I are now working in Python for our research. Thus, having access to geostatistical methods in Python directly impacts and facilitates the research of my group. 

Finally, I like to code. I have over 25 years of experience in FORTRAN, C++ and Visual Basic programing. This includes frontend (Qt interfaces in C++) and backend development with small and at times very large engineering and geoscience projects. 

### What's New with Version 0.33

Professor Honggeun Jo's (Inha University, South Korea) lead the implimentation of 3D methods to GeostatsPy. This include:

* 3D variogram calculation (gam_3D), modeling (make_variogram_3D) and visualization (vmodel_3D)
* kriging (kb3d)
* note, Prof. Jo has already completed a 3D indicator kriging algorithm that I will test and add shortly.

Note, GeostatsPy follows the NumPy standard and assumes 3D arrays indexed as my_array[nz,ny,nx] with both y and z reversed. I will add a few well-documented demonstrations to my GitHub shortly.

Michael

### What's New with Version 0.27

Finally got to those bugs in sequential Gaussian simulation! We now have improved reproduction of the variogram and a big simplication of the inputs.

### What's Included

This package contains 2 parts:

1. **geostatspy.geostats** includes GSLIB functions rewritten in Python. This currently includes all the variogram, distribution transformations, and spatial estimation and simulation (SGSIM soon) methods. I will continue adding functions to support modeling operations for practical subsurface model cosntruction. 

2. **geostatspy.GSLIB** includes reimplimentation of the GSLIB visualizations and low tech wrappers of the numerical methods  (note: the low-tech wrapper require access to GSLIB executables).

<p>
    <img src="https://github.com/GeostatsGuy/GeostatsPy/blob/master/TCG_color_logo.png" width="220" height="200" />
</p>

### The Authors

This package is being developed at the University of Texas in the Texas Center for Geostatistics.

* **Michael J. Pyrcz, Ph.D., P.Eng.** - associate professor with The University of Texas at Austin. Primary author of the package.

* **Honggeun Jo** - Ph.D. student working with Michael Pyrcz at The University of Texas at Austin. Author of 3D subroutines, 3D variogram calculation and modeling and wrapper for sgsim for 3D modeling.  Thank you Honggeun!

* **Anton Kupenko** - bug fixes, added docstrings, code refractory for PEP8, removed duplicated functions and variables.  Thank you Anton!

* **Wendi Liu** - Ph.D. student working with Michael Pyrcz at The University of Texas at Austin. Author of 3D subroutines and gammabar method.  Also, GSLIB compiles in Mac OSX, and 3D variogram calculation wrapper.  Thank you Wendi!

* **Alex E. Gigliotti** - undergraduate student working with Michael Pyrcz at The University of Texas at Austin. Established unit testing.  Thank you Alex!

* **Travis Salomaki** - Ph.D. student working with Michael Pyrcz at The University of Texas at Austin. Improving package docs.  Thank you Travis!

* **Javier Santos** - one of Michael Pyrcz's graduate students contributed the post processing algorithm for summarizing over multiple realizations.  Thank you Javier!

## Package Inventory

Here's a list and some details on each of the functions available.

### geostatspy.GSLIB Functions

Utilities to support moving between Python DataFrames and ndarrays, Data Tables, Gridded Data and Models in Geo-EAS file format (standard to GSLIB):

1. **ndarray2GSLIB** - utility to convert 1D or 2D numpy ndarray to a GSLIB Geo-EAS file for use with GSLIB methods 
2. **GSLIB2ndarray** - utility to convert GSLIB Geo-EAS files to a 1D or 2D numpy ndarray for use with Python methods
3. **Dataframe2GSLIB(data_file,df)** - utility to convert pandas DataFrame to a GSLIB Geo-EAS file for use with GSLIB methods
4. **GSLIB2Dataframe** - utility to convert GSLIB Geo-EAS files to a pandas DataFrame for use with Python methods
5. **DataFrame2ndarray** - take spatial data from a DataFrame and make a sparse 2D ndarray (NaN where no data in cell)

Visualization functions with the same parameterization as GSLIB using matplotlib:

6. **pixelplt** - reimplemention in Python of GSLIB pixelplt with matplotlib methods
7. **pixelplt_st** - reimplemention in Python of GSLIB pixelplt with matplotlib methods with support for sub plots
8. **pixelplt_log_st** - reimplemention in Python of GSLIB pixelplt with matplotlib methods with support for sub plots and log color bar
9. **locpix** - pixel plot and location map, reimplementation in Python of a GSLIB MOD with MatPlotLib methods
10. **locpix_st** - pixel plot and location map, reimplementation in Python of a GSLIB MOD with MatPlotLib methods with support for sub plots
11. **locpix_log_st** - pixel plot and location map, reimplementation in Python of a GSLIB MOD with MatPlotLib methods with support for sub plots and log color bar
12. **hist** - histograms reimplemented in Python of GSLIB hist with MatPlotLib methods
13. **hist_st** - histograms reimplemented in Python of GSLIB hist with MatPlotLib methods with support for sub plots

Data transformations

14. **affine** - affine distribution transformation to correct feature mean and standard deviation
15. **nscore** - normal score transform, wrapper for nscore from GSLIB (GSLIB's nscore.exe must be in working directory)
16. **declus** - cell-based declustering, 2D wrapper for declus from GSLIB (GSLIB's declus.exe must be in working directory)

Spatial Continuity

17. **make_variogram** - make a dictionary of variogram parameters to for application with spatial estimation and simulation 
18. **gamv** - irregularly sampled variogram, 2D wrapper for gam from GSLIB (.exe must be in working directory)
19. **varmap** - regular spaced data, 2D wrapper for varmap from GSLIB (.exe must be in working directory)
20. **varmapv** - irregular spaced data, 2D wrapper for varmap from GSLIB (.exe must be in working directory)
21. **vmodel** - variogram model, 2D wrapper for vmodel from GSLIB (.exe must be in working directory)

Spatial Modeling

22. **kb2d** - kriging estimation, 2D wrapper for kb2d from GSLIB (GSLIB's kb2d.exe must be in working directory)
23. **sgsim_uncond** - sequential Gaussian simulation, 2D unconditional wrapper for sgsim from GSLIB (GSLIB's sgsim.exe must be in working directory)
24. **sgsim** - sequential Gaussian simulation, 2D and 3D wrapper for sgsim from GSLIB (GSLIB's sgsim.exe must be in working directory)
25. **cosgsim_uncond** - sequential Gaussian simulation, 2D unconditional wrapper for sgsim from GSLIB (GSLIB's sgsim.exe must be in working directory)

Spatial Model Resampling

26. **sample** - sample 2D model with provided X and Y and append to DataFrame
27. **gkern** - make a Gaussian kernel for convolution, moving window averaging (from Teddy Hartano, Stack Overflow)
28. **regular_sample** - extract regular spaced samples from a 2D spatial model 
29. **random_sample** - extract random samples from a 2D spatial model  
30. **DataFrame2ndarray** - convent spatial point data in a DataFrame to a sparse ndarray grid

### geostatspy.geostats Functions

Numerical methods in GSLIB (Deutsch and Journel, 1998) translated to Python:

31. **correct_trend** - correct the order relations of an indicator-based trend model
32. **backtr** - GSLIB's backtr function  to transform a distribution
33. **declus** - GSLIB's DECLUS program reimplimented for cell-based declustering in 2D
34. **gam** - GSLIB's GAM program reimplimented for variogram calculation with regular data in 2D
35. **gamv** - GSLIB's GAMV program reimplimented for variogram calculation with iregular data in 2D 
36. **varmapv** - GSLIB's VARMAP program reimplimented for irregularly spaced spatial data in 2D 
37. **vmodel** - GSLIB's VMODEL program reimplimented for visualization of nested variogram models in 2D
38. **nscore** - GSLIB's NSCORE program reimplimented for normal score distribution transformation
39. **kb2d** - GSLIB's KB2D program reimplimented for 2D kriging-based spatial estimation
40. **ik2d** - GSLIB's IK3D program reimplimented for 2D indicator-based kriging estimation
41. **kb3d** - GSLIB's kt3d program reimplimented for 3D kriging-based spatial kriging estimation
42. **sgsim** - GSLIB's sgsim program reimplimented for 2D spatial simulation
43. **postsim** - GSLIB's postsim program reimplimented for summarizing over multiple realizations

More functionality will be added soon.

### Package Dependencies

The functions rely on the following packages:

1. **numpy** - for ndarrays
2. **pandas** - for DataFrames
3. **numpy.linalg** - for linear algebra
4. **numba** - for numerical speed up
5. **scipy** - for fast nearest neighbor search
6. **matplotlib.pyplot** - for plotting
7. **tqdm** - for progress bar
8. **statsmodels** - for weighted (debiased) statistics                

These packages should be available with any modern Python distribution (e.g. https://www.anaconda.com/download/).

If you get a package import error, you may have to first install some of these packages. This can usually be accomplished by opening up a command window on Windows and then typing 'python -m pip install [package-name]'. More assistance is available with the respective package docs.  

### Explanation of GeostatsPy

GeostatsPy includes functions that run 2D workflows from GSLIB in Python (i.e. low tech wrappers), Python translations and reimplementations of GSLIB methods, along with utilities to move between GSLIB's Geo-EAS data sets and Pandas DataFrames, and grids and 2D NumPy ndarrays respectively and other useful operations such as resampling from regular datasets and rescaling distributions.  

The reimplementations as of now include NSCORE, GAM, GAMV, VMODEL, DECLUS, KB2D, IK2D and SGSIM etc. and most of the visualizations using the standard GSLIB parametric inputs and matplotlib back end. The low tech wrappers simply write the GSLIB parameters, run the GSLIB executables and then read in the GSLIB output. This allows for construction of Python workflows with the very robust GSLIB programs.

#### Why make this package? 

I wanted a set of functions to utilize the very robust and numerically efficient GSLIB: Geostatistical Library (Deutsch and Journel, 1998) in Python. While there are other current solutions in Python, I found that these solutions are either proprietary (not open source), not maintained or missing vital functionality; therefore, I have not been able to use these other solutions to teach modeling workflows to students with little or no programming experience. Imagine getting 55 undergraduate students to resort back to a previous version of Python because a single dependency of an available package is not available in a current Python version. Image a student about to submit an assignment, and the code won't run immediately before submission because of an update to a dependency. I need methods for my students that just work, are reliable and do not require students to complete a more complicated environment setup.

Deutsch and Journel (1998) gave the community GSLIB, an extremely robust and flexible set of tools to build spatial modeling workflows. I have spent almost 20 years working with GSLIB along with a wide variety of subsurface modeling software. The powerful flexibility of GSLIB may be lost in methods that attempt to 'can' the inputs and parameters into complicated and hard to work with objects or attempt to combine the programs into a single program. I love open source for teaching the theory because students must see under the hood! The concept of basic building blocks and simple, common inputs is essential to GSLIB. I tried to preserve this by putting together functions with the same conventions as GSLIB, the result is a set of functions that (1) are practical for my students to use and (2) will move the GSLIB veterans into Python workflow construction. Honestly, I did nothing original, but that was my intention.  

I'm a very busy new professor, I'll keep adding more functionality as I have time.

#### More on GSLIB

The GSLIB source is available from GSLIB.com. If you would like to get the executables, ready to use without any need to compile them, go to GSLIB.com for Windows and Linux. I failed to find any Mac OS X executables so my Ph.D. student Wendi Liu compiled them for us (thank you Wendi!) and we have posted them here https://github.com/GeostatsGuy/GSLIB_MacOS. If folks on Windows are encountering missing DLL's, I could post static builds. Wendi provided instructions to help Mac users with missing DLL issues at that same location above.

#### Making Images 

The graphical / visualization methods have 2 variants. '_ST' specifies that function is suitable for building subplots (stacked / combined images) while those without '_ST' are for stand-alone images (an image is returned with the function call and an image file is saved in the working directory). The resolution and image file type are specified at the top of the GeostatPy.GSLIB functions.   

#### Assistance Welcome

Found an issue (I'm sure there are issues!)? Got a new idea? Want to work on this? Let me know, submit revisions, I'm happy to collaborate.

### Package Examples

There are many example workflow examples available on my GitHub account at https://github.com/GeostatsGuy/, specifically the GeostatsPy https://github.com/GeostatsGuy/GeostatsPy and PythonNumericalDemos  https://github.com/GeostatsGuy/PythonNumericalDemos repositories. Most of these examples simply placed the code for the required functions directly in the Jupyter notebook. These were made before this Package was made as I was developing all the functions individually. To use these examples just make these modifications:

1. install geostatspy with the command *pip install geostatspy*. I used the terminal Anaconda Navigator under the environments tab to make sure the package was accessible to Jupyter Notebooks.  
2. add *import geostatspy.GSLIB as GSLIB* and *import geostatspy.geostats as geostats* to the top of the workflow 
3. add *GSLIB.* or *geostats.* as a 'prefex to the GeostatsPy functions based on which set they belong to.

Over the next month I will update all workflows to use the geostatspy package instead of pasting code into the workflows. 

Here's a simple exaple of declustering with the geostatspy package. It looks long because we include making a synthetic dataset, dropping samples to impose a sampling bias, declustering and all the visualization and diagnostics.

```python
import geostatspy.GSLIB as GSLIB                          # GSLIB utilities, viz and wrapped functions
import geostatspy.geostats as geostats                    # GSLIB converted to Python
import matplotlib.pyplot as plt                           # plotting
import scipy.stats                                        # summary stats of ndarrays

# Make a 2d simulation
nx = 100; ny = 100; cell_size = 10                        # grid number of cells and cell size
xmin = 0.0; ymin = 0.0;                                   # grid origin
xmax = xmin + nx * cell_size; ymax = ymin + ny * cell_size# calculate the extent of model
seed = 74073                                              # random number seed  for stochastic simulation    
range_max = 1800; range_min = 500; azimuth = 65           # Porosity variogram ranges and azimuth
vario = GSLIB.make_variogram(0.0,nst=1,it1=1,cc1=1.0,azi1=65,hmaj1=1800,hmin1=500) # assume variogram model
mean = 10.0; stdev = 2.0                                  # Porosity mean and standard deviation
vmin = 4; vmax = 16; cmap = plt.cm.plasma                 # color min and max and using the plasma color map

# calculate a stochastic realization with standard normal distribution
sim = GSLIB.sgsim_uncond(1,nx,ny,cell_size,seed,vario,"simulation") # 2d unconditional simulation
sim = GSLIB.affine(sim,mean,stdev)                        # correct the distribution to a target mean and standard deviation

# extract samples from the 2D realization 
sampling_ncell = 10  # sample every 10th node from the model
samples = GSLIB.regular_sample(sim,xmin,xmax,ymin,ymax,sampling_ncell,10,10,nx,ny,'Realization')

# remove samples to create a sample bias (preferentially removed low values to bias high)
samples_cluster = samples.drop([80,79,78,73,72,71,70,65,64,63,61,57,56,54,53,47,45,42]) # this removes specific rows (samples)
samples_cluster = samples_cluster.reset_index(drop=True)  # we reset and remove the index (it is not sequential anymore)
GSLIB.locpix(sim,xmin,xmax,ymin,ymax,cell_size,vmin,vmax,samples_cluster,'X','Y','Realization','Porosity Realization and Regular Samples','X(m)','Y(m)','Porosity (%)',cmap,"Por_Samples")

# apply the declus program convert to Python
wts,cell_sizes,averages = geostats.declus(samples_cluster,'X','Y','Realization',iminmax=1,noff=5,ncell=100,cmin=1,cmax=2000)
samples_cluster['wts'] = wts            # add the weights to the sample data
samples_cluster.head()

# plot the results and diagnostics for the declustering
plt.subplot(321)
GSLIB.locmap_st(samples_cluster,'X','Y','wts',xmin,xmax,ymin,ymax,0.0,2.0,'Declustering Weights','X (m)','Y (m)','Weights',cmap)

plt.subplot(322)
GSLIB.hist_st(samples_cluster['wts'],0.0,2.0,log=False,cumul=False,bins=20,weights=None,xlabel="Weights",title="Declustering Weights")
plt.ylim(0.0,20)

plt.subplot(323)
GSLIB.hist_st(samples_cluster['Realization'],0.0,20.0,log=False,cumul=False,bins=20,weights=None,xlabel="Porosity",title="Naive Porosity")
plt.ylim(0.0,20)

plt.subplot(324)
GSLIB.hist_st(samples_cluster['Realization'],0.0,20.0,log=False,cumul=False,bins=20,weights=samples_cluster['wts'],xlabel="Porosity",title="Naive Porosity")
plt.ylim(0.0,20)

# Plot the declustered mean vs. cell size to check the cell size selection
plt.subplot(325)
plt.scatter(cell_sizes,averages, c = "black", marker='o', alpha = 0.2, edgecolors = "none")
plt.xlabel('Cell Size (m)')
plt.ylabel('Porosity Average (%)')
plt.title('Porosity Average vs. Cell Size')
plt.ylim(8,12)
plt.xlim(0,2000)

print(scipy.stats.describe(wts))

plt.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=3.5, wspace=0.2, hspace=0.2)
plt.show()

```
I also have various lecture notes, demonstrations and example workflows on GitHub (see link below), and some tutorials on my YouTube channel (see link below). I will continue to add more.  I'm just getting started,

*Michael*

Michael Pyrcz, Ph.D., P.Eng. Professor, Cockrell School of Engineering, The Jackson School of Geosciences, The University of Texas at Austin

#### More Resources Available at: [Twitter](https://twitter.com/geostatsguy) | [GitHub](https://github.com/GeostatsGuy) | [Website](http://michaelpyrcz.com) | [GoogleScholar](https://scholar.google.com/citations?user=QVZ20eQAAAAJ&hl=en&oi=ao) | [Book](https://www.amazon.com/Geostatistical-Reservoir-Modeling-Michael-Pyrcz/dp/0199731446) | [YouTube](https://www.youtube.com/channel/UCLqEr-xV-ceHdXXXrTId5ig)  | [LinkedIn](https://www.linkedin.com/in/michael-pyrcz-61a648a1)


