## Simulation notes:##

**Adding object border for near-field and far-field ptychography experiments:** 
1. Problematically, in blind ptychography experiments we encounter the *affine phase ambiguity*, i.e. a phase gradient (or phase ramp) in both the probe and the object phases. To avoid this ambiguity, we can add a *known* border to the object (a tight support), and fill the border with ones (i.e,. assume the border is transparent). This is the *bright-field boundary condition* that Albert Fannjiang talks about in:
  * [Blind ptychography by Douglas-Rachford splitting](https://arxiv.org/pdf/1809.00962.pdf)
  
  Also refer to other works by Fannjiang:
  * [Blind ptychography: uniqueness and ambiguities](https://arxiv.org/abs/1806.02674)
  * [Raster grid pathology and the cure](https://arxiv.org/abs/1806.02674)
2. Thus, to avoid the affine phase ambiguity, we assume that the object support is *priorly known*, i.e. we impose a *tight support*.
3. If we do not remove the affine phase ambiguity this way, then we have to remove the phase ramp post-reconstruction, which is more difficult and error-prone. 
4. For the far-field ptychography condition, by adding this border, we additionally ensure overlapping probe coverage through the entirety of the object (including the edges).
5. We cannot use this approach for the Bragg ptychography case---while we get transmission through empty space, we do not get diffraction (at the Bragg angle) through empty space. As such, the bright field boundary condition does not apply.

**Implementation notes**:

1. As a default, the simulation code assumes that we can use Fraunhofer propagation for far-field propagation, and the 
Fresnel Transfer Function method for the near-field propagation. type, this is experimental and is not used anywhere else.

**Other notes**:
1. The documentation is rather sparse and sometimes incorrect.
2. The examples need more explanation.

**Todo**:
1. Clean up and add code for the test cases, the timing benchmarks, and for the results reported  in the paper.
2. Clean up documentation and verify everything works as intended.