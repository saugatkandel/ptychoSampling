# ptychoSampling

**Contains**:
1. Contains demo code for the papers:
- [Using automatic differentiation as a general framework for ptychographic reconstruction](https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-13-18653&id=414640)

and

- [Efficient ptychographic phase retrieval via a matrix-free Levenberg-Marquardt algorithm](https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-15-23019&id=453099)

The optimization part of second-order optimization approach described in the latter paper is contained in https://github.com/saugatkandel/sopt as more general opitmization code, whereas this repository contains the wrappers for the ptycohgraphy applications.


2. Tutorials for simple ptychography reconstruction applications with tensorflow (contained in
[tensorflow_tutorials](https://github.com/saugatkandel/ptychoSampling/tree/master/tensorflow_tutorials)). It is safe to ignore the python package setup procedure.
3. A generic, modular tensorflow-based simulation and reconstruction framework in [ptychoSampling](https://github.com/saugatkandel/ptychoSampling/tree/master/ptychoSampling). The documentation, however, is quit
e sparse and sometimes
 unchanged from that for older versions of the code.
4. Simulation and reconstruction examples for far-field cases.

**Warning**:

The documentation is completely out-of-date. 


**Notes**:
1. For ease of application, the forward model simulation code uses numpy. Only the reconstruction code uses
 Tensorflow.

 2. Uses Tensorflow 1.14 for now. In the future, I am planning on switching away from the static computational graphs
  to a dynamic framework (Tensorflow 2.0, Pytorch, Autograd, Jax, etc) for ease of usage.

