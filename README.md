## Package for Optimizing QET based Phonon sensitive Detectors.


The code structure follows that of a real detector.

A Detector object is composed of an `Absorber` object and a `QET` object. A `QET` is composed of at `TES` object. `Absorber`, `QET`, and `TES` objects have material properties that are passed from `darkopt.materials`.

See `Examples/example.ipynb` for usage of the package

### Installation


To install the most recent (stable) development version of DarkOpt, clone this repo, then from the top-level directory of the repo, type the following line into your command line

`pip install .`

You may need to add the `--user` flag if using a shared Python installation.

This package requires python 3.6 or greater. A current version of Anaconda3 should be sufficient, however a conda environment file as well as a list of dependencies is provided (condaenv.yml and requirements.txt)

#### TODO
* Add Tc dependent plots
* 
* 



####	Work needed:
 
- Currently assumes bare crystal surface is elastic process, no down conversion
	- Think it is < 1%
	- Number of bounces before phonons go subgap ?
	- Need Down Conversion but now bellow 1 meV  
- Lowering W Tc and Fin Tc (are the set values correct?)  

This is based on erlier work from Matt Pyle's matlab optimization code.  
