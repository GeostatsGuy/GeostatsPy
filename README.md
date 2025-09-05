<h1 align="center"<p>
    <img src="https://github.com/GeostatsGuy/GeostatsPy/blob/master/geostatspy_logo.png?raw=true" width="200" height="200" />
</p></h1>

[![Documentation Status](https://readthedocs.org/projects/geostatspy/badge/?version=latest)](https://geostatspy.readthedocs.io/en/latest/?badge=latest)

<h1 align="center">GeostatsPy Python Package: Open-source Spatial Data Analytics and Geostatistics</h1>

<h3 align="center"> Now in 3D! </h3>

<h3 align="center"></h3>

[![Documentation Status](https://readthedocs.org/projects/geostatspy/badge/?version=latest)](https://geostatspy.readthedocs.io/en/latest/?badge=latest)

#### Cite As:

Pyrcz, M.J., Jo. H., Kupenko, A., Liu, W., Gigliotti, A.E., Salomaki, T., and Santos, J., 2021, GeostatsPy Python Package: Open-source Spatial Data Analytics and Geostatistics, DOI: /doi/10.5281/zenodo.13835444.

[![DOI](https://zenodo.org/badge/139197285.svg)](https://zenodo.org/doi/10.5281/zenodo.13835444)

___

#### NEW: 3D Sequential Indicator Simulation (SISIM) and 3D Sequential Gaussian Simulation (SGSIM)

I have a confession: many of the GSLIB functions in the GeostatsPy Python package were ported or written from scratch by me—often late at night and into the early morning hours—just in time for the next day’s lecture.

Back in 2018, I desperately needed a usable, open-source geostatistics Python package to teach my courses. I found one, but on the first day of the semester, it broke due to an update in a dependent package. With no alternatives, I decided to build my own. My goal was simple: implement the minimum necessary tools to teach geostatistics in practice, not just in theory.

Because:

* you can’t teach geostatistics without algorithms, and…

* students weren’t going to edit parameter files and run old FORTRAN GSLIB executables. (I tried that once—many students dropped the course in the first week!)

* my [Excel demonstrations](https://github.com/GeostatsGuy/ExcelNumericalDemos) are great for theory and math, but they don’t support building full models.

So, in those early days, I made a key decision, everything would be in 2D only. Why?

* it made testing and debugging much easier.

* it was perfect for live demos and slides: easy to explain, visualize, and diagnose.

Fast forward to now... I’m lucky to have an amazing graduate student team—typically 12–15 students, mostly PhDs—and many of them want to build 3D geostatistical models. So, I couldn’t help myself. I went back and updated everything—search algorithms, kriging, distance metrics with geometric anisotropy, and more... and finally,

* sgsim_3D - a 3D implimentation of sequential Gaussian simulation

* sisim_3D - a 3D implimentation of sequetial indicator simulation

##### Notes on this First Version:

Tested:

1. simple and ordinary kriging

2. stationary mean or proportion realizations

3. data conditioning

Not yet tested:

1. locally variable mean / proportion

2. collocated cokriging

I’ll be asking one of my PhD students to test these soon!

##### Improvements Along the Way

I made some improvements over the current 2D implimentations,

1. continued to replace legacy FORTRAN-style loops from GSLIB with NumPy broadcasting, making the code more readable, robust, and concise.

2. fixed the reference distribution option in sgsim_3D, allowing users to transform/back-transform with limited conditioning data using an external Gaussian table.

I did make a couple of improvements,

* I'm always looking to replace the FORTRAN loops of GSLIB with broadcast methods from NumPy for more robust, readable and concise codes.

* I have fixed the reference distribution option for SGSIM, so you can use few data and another file to provide the Gaussian transformation and back-transformation table.

##### Special Thanks

Huge thanks to [Professor Honggeun Jo](https://www.linkedin.com/in/honggeun-jo/?originalSubdomain=kr), whose prior work on 3D variograms, covariance functions, and more made this possible. Your contributions to GeostatsPy have been incredible.

___

#### NEW: Check out the new e-book, Applied Geostatistics in Python: A Hands-on Guide with GeostatsPy, by Michael J. Pyrcz

The goal of this e-book is to teach the application of geostatistics in Python, for those new to geostatistics I provide theory and links to my course content, and for those experienced practitioners I provide example workflows and plots that you can implement.

e-book citation and link:

Pyrcz, M.J., 2024, Applied Geostatistics in Python: A Hands-on Guide with GeostatsPy, https://geostatsguy.github.io/GeostatsPyDemos_Book.

___

#### GeostatsPy Python Package 

The GeostatsPy Package brings GSLIB: Geostatistical Library (Deutsch and Journel, 1998) functions to Python. GSLIB is a practical and extremely robust set of code for building spatial modeling workflows. 

I created the GeostatsPy Package to support my students in my **Data Analytics**, **Geostatistics** and **Machine Learning** courses. I find my students benefit from hands-on opportunities, in fact it is hard to imagine teaching these topics without providing the opportunity to handle the numerical methods and build workflows. Last year, I tried to have them use the original FORTRAN executables and even with support and worked out examples, it was an uphill battle. In addition, all my students and I are now working in Python for our research. Thus, having access to geostatistical methods in Python directly impacts and facilitates the research of my group. This package retains the spirit of GSLIB:

* **modularity** - a collection of standalone functions that may be applied in sequence for maximum flexibility for building workflows
* **minimalistic** - the simplest possible code to support the "look at the code" approach to learning
* **fundamental** - based on the well-established geostatistical theory by avoiding ad hoc methods and assumptions

This package contains 2 parts:

1. **geostatspy.geostats** includes GSLIB functions rewritten in Python. This currently includes all the variogram, distribution transformations, and spatial estimation and simulation methods. I will continue adding functions to support modeling operations for practical subsurface model cosntruction. 

2. **geostatspy.GSLIB** includes reimplimentation of the GSLIB visualizations and low tech wrappers of the numerical methods (note: the low-tech wrapper require access to GSLIB executables).

#### Getting Started

I have built out many well-documented workflow in Jupyter Notebooks using GeostatsPy functions to complete common workflows in spatial data analytics and geostatistics. They are available in my [GeostatsPy_Demos Repository](https://github.com/GeostatsGuy/GeostatsPy_Demos). I hope these are helpful!

#### Setup

A minimum environment includes:

* Python 3.7.10 - due to the depdendency of GeostatsPy on the Numba package for code acceleration

***

#### Installing GeostatsPy

GeostatsPy is available on the Python Package Index (PyPI) [GeostatsPy PyPI](https://pypi.org/project/geostatspy/).

To install GeostatsPy, use pip

```console
pip install geostatspy
```

***

#### GeostatsPy Package Dependencies

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

***

#### Recent Updates

Here's some highlights from recent updates:

##### What's New with Version 0.33

Professor Honggeun Jo's (Inha University, South Korea) lead the implimentation of 3D methods to GeostatsPy. This include:

* 3D variogram calculation (gam_3D), modeling (make_variogram_3D) and visualization (vmodel_3D)
* kriging (kb3d)
* note, Prof. Jo has already completed a 3D indicator kriging algorithm that I will test and add shortly.

Note, GeostatsPy follows the NumPy standard and assumes 3D arrays indexed as my_array[nz,ny,nx] with both y and z reversed. I will add a few well-documented demonstrations to my GitHub shortly.

##### What's New with Version 0.27

Finally got to those bugs in sequential Gaussian simulation! We now have improved reproduction of the variogram and a big simplication of the inputs.

***

#### The Authors

This package is being developed at The University of Texas in the Texas Center for Geostatistics.

* **Professor Michael J. Pyrcz, Ph.D., P.Eng.** - professor with The University of Texas at Austin. Primary author of the package.

* **Professor Honggeun Jo, Ph.D.** - assistant professor with Inha University, South Korea. Author of 3D subroutines, 3D variogram calculation and modeling and wrapper for sgsim for 3D modeling and more! Thank you, Professor Jo!

* **Anton Kupenko** - bug fixes, added docstrings, code refractory for PEP8, removed duplicated functions and variables. Thank you, Anton!

* **Wendi Liu, Ph.D.** - while a Ph.D. student working with Michael Pyrcz at The University of Texas at Austin. Author of 3D subroutines and gammabar method. Also, GSLIB compiles in Mac OSX, and 3D variogram calculation wrapper. Thank you, Dr. Wendi Liu!

* **Alex E. Gigliotti** - undergraduate student working with Michael Pyrcz at The University of Texas at Austin. Established unit testing.  Thank you Alex!

* **Travis Salomaki** - as an undergraduate student research project with Michael Pyrcz at The University of Texas at Austin. Improving package docs. Thank you, Travis!

* **Javier Santos, Ph.D.** - while a Ph.D. student working with Michael Pyrcz at The University of Texas at Austin. Author of the post processing algorithm for summarizing over multiple realizations. Thank you, Javier!

#### Package Inventory

Here's a list and some details on each of the functions available.

##### geostatspy.GSLIB Functions

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

##### geostatspy.geostats Functions

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

More functionality will continue to be added.

#### Explanation of GeostatsPy

GeostatsPy includes functions that run 2D workflows from GSLIB in Python (i.e. low tech wrappers), Python translations and reimplementations of GSLIB methods, along with utilities to move between GSLIB's Geo-EAS data sets and Pandas DataFrames, and grids and 2D NumPy ndarrays respectively and other useful operations such as resampling from regular datasets and rescaling distributions.  

The reimplementations as of now include NSCORE, GAM, GAMV, VMODEL, DECLUS, KB2D, IK2D and SGSIM etc. and most of the visualizations using the standard GSLIB parametric inputs and matplotlib back end. The low tech wrappers simply write the GSLIB parameters, run the GSLIB executables and then read in the GSLIB output. This allows for construction of Python workflows with the very robust GSLIB programs.

##### Why make this package? 

I wanted a set of functions to utilize the very robust and numerically efficient GSLIB: Geostatistical Library (Deutsch and Journel, 1998) in Python. While there are other current solutions in Python, I found that these solutions are either proprietary (not open source), not maintained or missing vital functionality; therefore, I have not been able to use these other solutions to teach modeling workflows to students with little or no programming experience. Imagine getting 55 undergraduate students to resort back to a previous version of Python because a single dependency of an available package is not available in a current Python version. Image a student about to submit an assignment, and the code won't run immediately before submission because of an update to a dependency. I need methods for my students that just work, are reliable and do not require students to complete a more complicated environment setup.

Deutsch and Journel (1998) gave the community GSLIB, an extremely robust and flexible set of tools to build spatial modeling workflows. I have spent almost 20 years working with GSLIB along with a wide variety of subsurface modeling software. The powerful flexibility of GSLIB may be lost in methods that attempt to 'can' the inputs and parameters into complicated and hard to work with objects or attempt to combine the programs into a single program. I love open source for teaching the theory because students must see under the hood! The concept of basic building blocks and simple, common inputs is essential to GSLIB. I tried to preserve this by putting together functions with the same conventions as GSLIB, the result is a set of functions that (1) are practical for my students to use and (2) will move the GSLIB veterans into Python workflow construction. Honestly, I did nothing original, but that was my intention.  

I'm a very busy new professor, I'll keep adding more functionality as I have time.

#### More on GSLIB

The GSLIB source is available from GSLIB.com. If you would like to get the executables, ready to use without any need to compile them, go to GSLIB.com for Windows and Linux. I failed to find any Mac OS X executables so my Ph.D. student Wendi Liu compiled them for us (thank you Wendi!) and we have posted them here https://github.com/GeostatsGuy/GSLIB_MacOS. If folks on Windows are encountering missing DLL's, I could post static builds. Wendi provided instructions to help Mac users with missing DLL issues at that same location above.

#### Making Images 

The graphical / visualization methods have 2 variants. '_ST' specifies that function is suitable for building subplots (stacked / combined images) while those without '_ST' are for stand-alone images (an image is returned with the function call and an image file is saved in the working directory). The resolution and image file type are specified at the top of the GeostatPy.GSLIB functions.   

#### Assistance Welcome

Found an issue (I'm sure there are issues!)? Got a new idea? Want to work on this? Let me know, submit revisions, I'm happy to collaborate.

***

#### Package Examples

I built out over 40 well-documented demonstration workflows that apply GeostatsPy to accomplish common spatial modeling tasks to support my students in my **Data Analytics and Geostatistics**, **Spatial Data Analytics** and **Machine Learning** courses and anyone else learning data analytics and machine learning. These are all available in the [GeostatsPyDemos Repository](https://github.com/GeostatsGuy/GeostatsPyDemos).
 
Here's a simple example of declustering with the GeostatsPy package. It looks a bit long because we include making a synthetic dataset, dropping samples to impose a sampling bias, declustering and all the visualization and diagnostics.

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

I also have various lecture notes, demonstrations and example workflows on GitHub (see link below), and lectures and tutorials on my YouTube channel (see link below). I hope that this is helpful,

*Michael*

Michael Pyrcz, Ph.D., P.Eng. Professor, Cockrell School of Engineering, The Jackson School of Geosciences, The University of Texas at Austin

#### More Resources Available at: [Twitter](https://twitter.com/geostatsguy) | [GitHub](https://github.com/GeostatsGuy) | [Website](http://michaelpyrcz.com) | [GoogleScholar](https://scholar.google.com/citations?user=QVZ20eQAAAAJ&hl=en&oi=ao) | [Book](https://www.amazon.com/Geostatistical-Reservoir-Modeling-Michael-Pyrcz/dp/0199731446) | [YouTube](https://www.youtube.com/channel/UCLqEr-xV-ceHdXXXrTId5ig)  | [LinkedIn](https://www.linkedin.com/in/michael-pyrcz-61a648a1)


