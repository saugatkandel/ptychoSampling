# Near-field propagation and phase retrieval with TensorFlow

The experimental forward model for the nearfield propagation is adapted from that in the paper by [Clare et al](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-15-19728) with the following key changes:
* the full-field probe is scaled down from 2048 pixels to 512 pixels (factor of 4).
* the detector pixel pitch is scaled from 75 nm to 300 nm.
* the wavelength is changed from 14.96 keV to 8.7 keV (or 0.14 nm).
* The step size is changed from 20 $\mu$m to $13.2$ $\mu$m (44 pixels).


**Corrected parameters (vs the [published parameters](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-27-13-18653))**:
* Detector pixel pitch is 300 nm and not 600 nm.
* Step size is 13.2 $\mu$m and not 10 $\mu$m.
* The gaussian used to simulate the probe has a standard deviation of 150 pixels (45 $\mu$m) and not 50 pixels (19 $\mu$m).
