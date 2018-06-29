## GeostatsPy - GSLIB (Geostatistical Library) Reimplemented in Python 
### Michael Pyrcz, Associate Professor, University of Texas at Austin 

#### Contacts: [Twitter/@GeostatsGuy](https://twitter.com/geostatsguy) | [GitHub/GeostatsGuy](https://github.com/GeostatsGuy) | [www.michaelpyrcz.com](http://michaelpyrcz.com) | [GoogleScholar](https://scholar.google.com/citations?user=QVZ20eQAAAAJ&hl=en&oi=ao) | [Book](https://www.amazon.com/Geostatistical-Reservoir-Modeling-Michael-Pyrcz/dp/0199731446)

#### Project Goal

Make the robust, well-known Geostatistical Library, GSLIB, available in Python.

#### Available Functions

The current state of GeostatsPy. These are the functions we have included here:

1. GSLIB2Dataframe - load GSLIB Geo-EAS data tables to Pandas DataFrame
2. DataFrame2GSLIB - save Pandas DataFrame to GSLIB Geo-EAS data tables
3. GSLIB2ndarray - load GSLIB Geo-EAS format regular grid data 1D or 2D to NumPy ndarray
4. ndarray2GSLIB - write NumPy array to GSLIB Geo-EAS format regular grid data 1D or 2D
5. hist - histograms plots reimplemented with GSLIB parameters using python methods
6. pixelplt - plot 2D NumPy arrays with same parameters as GSLIB's pixelplt 
7. locmap - location maps reimplemented with GSLIB parameters using python methods
8. vargplt - variogram visualizer
9. gam_2d -regularly sampled data variograms
10. gamv_2d - irregularly sampled data variograms
11. vmodel_2d - variogram map calculator from gridded data
12. vmodelv_2d - variogram map calculator from iregularly sampled data
13. locpix - my modification of GSLIB to superimpose a location map on a pixel plot reimplemented with GSLIB parameters using Python methods
14. affine - affine correction adjust the mean and standard deviation of a feature reimplemented with GSLIB parameters using Python methods
15. nscore - normal score transform (data transformation to Gaussian with a mean of zero and a standard deviation of one)
16. declus - cell-ased declustering
17.sgsim - sequantial Gaussian simulation limited to 2D and unconditional

#### Requirements

GSLIB executables must be in current directory (or a directory in path environmental variable).  Executables are available at http://www.statios.com/Quick/gslib.html.


Warning, there has been no attempt to make these functions robust in the precense of bad inputs. If you get a crazy error check the inputs. Are the arrays the correct dimension? Is the parameter order mixed up? Make sure the inputs are consistent with the descriptions in this document.
