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

***

### Implementation notes:
The reconstruction algorithm implemented here allows for the separation of ptychographic data into separate training and validation buckets. The validation data can be used as a criterion for early stopping of the reconstruction procedure. However, our [published results](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-27-13-18653) use an earlier version of the reconstruction framework without this feature. 

***

### Notes about computational efficiency:
**Caveat**: This applies only for the case when we have a full field probe with object translations.

There are different approaches we can use to calculate the exit waves for each minibatch forward model and derivative calculation. For a minibatch size $b$, object size $192\times 192$ pixels, and probe size $512 \times 512$ pixels, we can choose to:

1. *__Stacking the views__*: Explicitly calculate the object placement at each scan position. An efficient way to accomplish this is by calculating how far away the (smaller) respective object edges are from the corresponding edges of the (larger) full-field probe array---this is different at each scan position. Using this information, we can pad the object array with ones, thus emulating an object surrounded by transparent free space. For $b>1$, we can then *stack* these arrays to calculate a minibatch of *object views*. 


2. *__Scattering into an existing tensor__*: For each scan position, we priorly calculate the target positional index for each object pixel. Now, if we take an existing tensor filled with ones (representing free space) of size $b\times 512 \times 512$, we can use this positional index to *scatter* each object pixel into the appropriate position in the larger tensor (i.e. replace the prior value ($=1.0$) with the appropiate value from the object function). 
    
    Code example: (`tf_view_indices` contains the position index information, `batch_views` is target tensor.)
    
    ```
    batch_obj_view_indices = tf.gather(tf_view_indices, batch_indices)
    obj_repeated = tf.tile(tf.reshape(tf_obj_1, [-1]), [batch_size])
    batch_obj_views = tf.tensor_scatter_update(batch_views, tf.reshape(batch_obj_view_indices, [-1,2]), obj_repeated)
    batch_obj_views = tf.reshape(batch_obj_views, [batch_size, *probe_init.shape])
    ```
    
    
    *Additional note*: I cannot use map_fn here (instead of tiling) because map_fn creates a new tensor, instead of just replacing values in an existing one.
    
    
3. *__Scatter_nd (complex-valued)__*: We use the calculated positional index to scatter the object values into a new $b\times 512 \times 512$ tensor filled with zeros. The new tensor is calculated on the fly. To accomplish this, we use the following steps:
    * Subtract 1 from the object values: $O_1 = O - 1$.
    * Scatter $O_1$ to a $b\times 512 \times 512$ tensor initially filled with zeros: $O^{full}_1 = \textrm{Scatter}[O_1]$.
    * Add $1$ to get the correct transparency everywhere. $O^{full} = O^{full}_1 + 1$.
    
    Code example:
    
    ```
    batch_view_indices = tf.gather(tf_view_indices, batch_indices)
    tf_obj_2 = tf_obj_1 - 1
    gen_view_fn = lambda view_indices: tf.scatter_nd(tf.reshape(view_indices, [-1,1]), 
                                                 tf.reshape(tf_obj_2, [-1]),
                                                 [probe_init.size])
    batch_obj_views = tf.map_fn(gen_view_fn, batch_view_indices, dtype=tf.complex64)
    batch_obj_views = tf.reshape(batch_obj_views, [batch_size, *probe_init.shape]) + 1
    ```
    
    
    *Additional note*: Using tiling instead of the map_fn approach results in slower code.
    
    
4. *__Scatter_nd (real-valued)__*: We use the calculated positional index to separately scatter the real and imaginary parts of the object values to two new $b\times 512 \times 512$ tensors filled with zeros. Steps:
    * Separate real and imaginary parts of the object: $O\longrightarrow (\textrm{Re}[O], \textrm{Im}[O])$.
    * Set $\textrm{Re}[O]_1 = \textrm{Re}[O] - 1$.
    * Scatter $\textrm{Re}[O]_1$ and $\textrm{Im}[O])$ into two $b\times 512 \times 512$ tensors $\textrm{Re}[O]_1^{full}$ and $\textrm{Im}[O]^{full}$.
    * Get new complex-valued tensor: $O^{full} = \textrm{Re}[O]_1^{full} + 1j\cdot\textrm{Im}[O]^{full} + 1$.
    
    Code example:
    
    ```
    batch_view_indices = tf.gather(tf_view_indices, batch_indices)
    gen_view_real_fn = lambda view_indices: tf.scatter_nd(tf.reshape(view_indices, [-1,1]), 
                                                 tf.reshape(tf_obj_real_pad -1, [-1]),
                                                 [probe_init.size]) + 1
    gen_view_imag_fn = lambda view_indices: tf.scatter_nd(tf.reshape(view_indices, [-1,1]), 
                                                 tf.reshape(tf_obj_imag_pad, [-1]),
                                                 [probe_init.size])
    batch_obj_real_views = tf.map_fn(gen_view_real_fn, batch_view_indices, dtype=tf.float32)
    batch_obj_imag_views = tf.map_fn(gen_view_imag_fn, batch_view_indices, dtype=tf.float32)
    batch_obj_views = tf.complex(batch_obj_real_views, batch_obj_imag_views)
    batch_obj_views = tf.reshape(batch_obj_views, [batch_size, *probe_init.shape])
    ```
    
    
    *Additional note*: Using tiling instead of the map_fn approach results in slower code.


#### Time costs:

The time costs for methods 1 (M1), 2 (M2), 3 (M3) and 4 (M4) are (all GPU):

| Batch size    | M1    | M2    |  M3   |   M4  |
| ----------    | ----- | ----- | ----- | ----- |
| b=5           | 3.7 s | 11 s  | 3.2 s | N/A   |
| b=10          | 5.6 s | 19 s  | 6.2 s | 8.2 s |

**Note**: Method 3 fails silently on the GPU---the minibatch updates don't work. Works only on the cpu. 

**Conclusion**: We use M1.