## PD2 Simulation 

A simulation for PD2 to obtain energy and position resolution. 

Current issues:

* Functional form of F_TFN
* Get Irwin Parameters to calculate w_eff (Get alpha to calculate L (eq 3.4 in thesis), heat capacity C)
* ~~What is T_o? Is it TES temperature??~~ How to get TES eq temp? 
* What is happening in electronics file lines 24-32
* Get combined iZIP efficiency (line 122 in detector.py)

 Resolved Issues: 
* ~~Get thermal conductance~~ Eq 3.3 in thesis  
* ~~Get Phonon ballistic propagation time from matlab sim~~ 
* ~~Get combined crystal + W/Al interface efficiency from matlab sim~~ 
* ~~Find out implementation of QET.ePQP~~ 
* ~~Understand lines 193-197 about diffusive transport to get QP transport efficiency QET.eQPabsb~~
* ~~Understand structure of electronics and fridge class. Get total inductance from electronics class.~~ 