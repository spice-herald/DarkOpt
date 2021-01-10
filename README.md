### Package for Optimizing QET based Phonon sensitive Detectors.



The code structure follows that of a real detector.

A Detector object is composed of an `Absorber` object and a `QET` object. A `QET` is composed of at `TES` object. `Absorber`, `QET`, and `TES` objects have material properties that are passed from `darkopt.materials`.

See `Examples/example.ipynb` for usage of the package




####	Current issues:

- Total load resistances/temperatures in electronics 
- Expoential overflow in simulated_noise line 131.
- Correct placement of squares etc. in formulae in simulated noise? Index manipulations maybe wrong also?

####	Work needed:
 
- Currently assumes bare crystal surface is elastic process, no down conversion
	- Think it is < 1%
	- Number of bounces before phonons go subgap ?
	- Need Down Conversion but now bellow 1 meV  
- Lowering W Tc and Fin Tc (are the set values correct?)  

This is based on erlier work from Matt Pyle's matlab optimization code.  
