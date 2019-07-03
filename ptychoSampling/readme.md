**Todo**:
1. Actually clean up the farfield_poisson code
2. Generalize the code so that it is adaptable for nearfield, farfield, and bragg cases. (This might be too difficult)

**Notes**:
1. The timings reported for the farfield ptychography code in the paper are from an earlier inefficient version of the code. The timings reported here are from a newer, much faster, version.

## Simulation notes:##

**Nearfield ptychography experiment**:
1. Uses a full-field probe with (smaller) object translations.
2. As such, the probe edges are not sampled sufficiently; the reconstructed probe edges are inaccurate.

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