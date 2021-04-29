# ptychoSampling

Contains demo code for the paper https://arxiv.org/abs/2103.01767


**Contains**:

1. Tutorials for simple ptychography reconstruction applications with tensorflow (contained in
[tensorflow_tutorials](https://github.com/saugatkandel/ptychoSampling/tree/master/tensorflow_tutorials)). It is safe to ignore the python package setup procedure.
2. A generic, modular tensorflow-based simulation and reconstruction framework in [ptychoSampling](https://github.com/saugatkandel/ptychoSampling/tree/master/ptychoSampling). The documentation, however, is quit
e sparse and sometimes
 unchanged from that for older versions of the code.
 3. Simulation and reconstruction examples for far-field, near-field, and bragg ptychography cases.


**Notes**:
1. For ease of application, the forward model simulation code uses numpy. Only the reconstruction code uses
 Tensorflow.

 2. Uses Tensorflow 1.14 for now. In the future, I am planning on switching away from the static computational graphs
  to a dynamic framework (Tensorflow 2.0, Pytorch, Autograd, Jax, etc) for ease of usage.
