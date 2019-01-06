
## GeostatsPy: a Set of Functions for GSLIB: Geostatistical Library into Python Workflows

### Michael Pyrcz, Associate Professor, University of Texas at Austin 

#### [Twitter/@GeostatsGuy](https://twitter.com/geostatsguy) | [GitHub/GeostatsGuy](https://github.com/GeostatsGuy) | [Website](http://michaelpyrcz.com) | [GoogleScholar](https://scholar.google.com/citations?user=QVZ20eQAAAAJ&hl=en&oi=ao) | [Book](https://www.amazon.com/Geostatistical-Reservoir-Modeling-Michael-Pyrcz/dp/0199731446) | [YouTube Lectures](https://www.youtube.com/channel/UCLqEr-xV-ceHdXXXrTId5ig)

If you get a package import error, you may have to first install some of these packages. This can usually be accomplished by opening up a command window on Windows and then typing 'python -m pip install [package-name]'. More assistance is available with the respective package docs.  

#### Explanation of GeostatsPy

GeostatsPy includes functions that run 2D workflows in GSLIB from Python (i.e. low tech wrappers) and in some cases reimplementations of GSLIB methods, along with utilities to move between GSLIB's Geo-EAS data sets and DataFrames, and grids and 2D Numpy arrays respectively and other useful operations such as resampling from regular datasets and rescaling distributions.  Here's a sumary list of functions avaible.

0. import and export from GSLIB's Geo-EAS format
1. histograms
2. location maps and combined location map and pixel plot
3. affine and nscore distribution transformations
4. variogram calculation and modeling
5. sgsim unconditional and conditional
6. cosgsim unconditonal
7. regular, random and user-specified resampling from a grid 

The reimplimentations are mainly of the visualizations using the standard GSLIB parametric inputs and Matplotlib back end. The low tech wrappers simply write the GSLIB parameters, run the GSLIB executables and then read in the GSLIB output. 

Why do it this way? I wanted a set of functions for working with the very robust and numerically efficient GSLIB Geostatistical Library (Deutsch and Journel, 1998) from Python.  There are other current solutions in Python.  I found that these solutions are either proprietary (not open source), not maintained or missing vital functionality; therefore, I have not been able to use these other solutions to teach modeling workflows to students with little or no programming experience.  Imagine getting 55 undergraduate students to role back ot a previous version on Python because a single dependency of an available package is not available in a current Python version.  Image a student about to submit an assignment that won't work immdeiately before submission because of an update. I need methods for my students that just work, are reliable and do not require students to complete a more complicated environment setup.

Deutsch and Journel (1998) gave the community GSLIB, an extremely robust and flexible set of tools to build spatial modeling workflows.  I have spent almost 20 years working with GSLIB along with a wide variety of subsurface modeling software. The powerful flexibility of GSLIB may be lost in methods that attempt to 'can' the inputs and parameters into complicated and hard to work with objects or attempt to combine the programs into a single program.  I love open source for teaching the theory, students must see under the hood!  The concept of basic building blocks and simple, common inputs is essential to GSLIB.  I tried to preserve this by putting together functions with the same conventions as GSLIB, the result is a set of functions that (1) are practical for my students to use and (2) will move the GSLIB veterans into Python workflow construction. Honestly, I did nothing original, but that was my intention.  

Of course, I could have properly wrapped GSLIB, built a proper package and maintained it as open source free to all.    I address this one day or maybe someone else will do that. That would be good for our community.  I'm just a new professor keeping barely keeping up with providing new quality classes and supporting materials to students, supporting my graduate students and finding funding.  Providing tools to support the community is something I believe in, I learned that as a PhD student with the great opportunity to be supervised by Clayton Deutsch.  I'll do what I can.

#### The GeostatsPy Functions

The next block includes all the current functions.  To use them save this as a .py file and import it at the beginning of your Python workflow. You could also just include them in the workflow as I do below.  The executables associated with the functions should be in the current working directory.  These include GSLIB programs.

1. nscore.exe
2. declus.exe
3. gam.exe
4. gamv.exe
5. vmodel.exe
6. kb2d.exe
7. sgsim.exe

The source for these are available from GSLIB.com.  If you would like to get the executables, ready to use without any need to compile them, go to GSLIB.com for Windows and Linux.  I failed to find any Mac OS X executables so my Ph.D. student Wendi Liu compiled them for us (thank you Wendi!) and we have posted them here https://github.com/GeostatsGuy/GSLIB_MacOS.  If folks on Windows are encountering missing DLL's, I could post static builds.  Wendi provided instructions to help Mac users with missing DLL issues at that same location above.

The functions are followed by a set of demonstrations.  There is no attempt to build a comprehensive workflow nor demonstrate best practice, but the demonstrations should provide examples usecases to help you build your own workflows and benchmarks for testing.

#### Required Libraries

GeostatPy functions require some packages including:

1. NumPy
2. Pandas
3. Matplotlib

along with miscellaneous operating system interfaces (os) and pseudo-random numbers (rand).  By design these are all standard and should be available with standard Python installs (e.g. https://www.anaconda.com/download/).

#### Images 

The graphical methods have 2 variants.  '_ST' specifies that function is suitable for building subplots (stacked / combined images) while those without '_ST' are for stand-alone images (output in the output block and as an image file). The resolution and image file type are specified at the top of the GeostatPy functions.   

#### Assistance Welcome

Found an issue (I'm sure there are issues!)? Got a new idea? Want to work on this? Let me know, submit revisions, I'm happy to collaborate.
