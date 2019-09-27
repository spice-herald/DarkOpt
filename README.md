## Phonon Detector Design

###For Optimizing Future Dark Matter Detectors. 

####	Current issues:

	* Total load resistances/temperatures in electronics 
	* Expoential overflow in simulated_noise line 131.
	* Correct placement of squares etc. in formulae in simulated noise? Index manipulations maybe wrong
	also?

####	Work needed:
 
	* Currently assumes bare crystal surface is elastic process, no down conversion
		* Think it is < 1%
		* # bounces before phonons go subgap ?
		* Need Down Conversion but now bellow 1 meV  
	* Lowering W Tc and Fin Tc (are the set values correct?)  

This is a python translation and reorganization of much of Matt Pyle's matlab optimization code.  
