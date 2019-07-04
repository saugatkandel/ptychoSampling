#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from skimage.feature import register_translation
import time
from scipy.stats import circmean
from helper import NearFieldPtychographySimulation, tfNearFieldPtychoRecons, tfNearFieldPtychoReconsFromSimulation
import dill



get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



def tensor_clip(t: tf.Tensor, 
                  max_abs: float=1.,
                  min_abs: float=0.):
    
    absval = tf.abs(t)
    abs_clipped = tf.clip_by_value(absval, min_abs, max_abs)
    multiplier = tf.cast(abs_clipped / (absval + 1e-30), 'complex64')
    return t * multiplier



nfsim = NearFieldPtychographySimulation()



with open('nfsim.pkl', 'wb') as f:
    dill.dump(nfsim, f)



with open('nfsim.pkl', 'rb') as f:
    nfsim = dill.load(f)



pos_centers = nfsim.scan_params.scan_area_buffer_npix + nfsim.positions + nfsim.obj_params.obj_w_border_npix // 2

plt.figure(figsize=[12, 3])
plot_items = [np.abs(nfsim.obj_true), 
              np.angle(nfsim.obj_true),
              np.abs(nfsim.probe_true),
              np.angle(nfsim.probe_true)]
plot_titles = ['Object Mod', 
               'Object Phase',
               'Probe Mod', 
               'Probe Phase']
for indx, item in enumerate(plot_items):
    plt.subplot(1, 4, indx + 1)
    plt.pcolormesh(item, cmap='gray')
    plt.colorbar()
    plt.title(plot_titles[indx])

plt.subplot(1,4,3)
plt.scatter(pos_centers.T[0], pos_centers.T[1])
plt.tight_layout()
plt.show()



nfrecons = tfNearFieldPtychoRecons(diffraction_mods=nfsim.diffraction_mods,
                                   positions=nfsim.positions,
                                   wavelength=nfsim.probe_params.wavelength,
                                   obj_detector_dist=nfsim.det_params.obj_dist,
                                   detector_pixel_pitch=nfsim.det_params.pixel_pitch,
                                   obj_npix=nfsim.obj_params.obj_npix,
                                   probe_npix=nfsim.probe_params.npix,
                                   probe_recons=True, 
                                   batch_size=5,
                                   n_validation_diffs=2)



nfrecons.setLossAndOptimizers(probe_learning_rate=1e1, obj_learning_rate=1e-1)



nfrecons.initSession()



nfrecons.run(patience=5000, max_iters=25000, debug_output_epoch_frequency=100)



recons_shape = np.array(nfsim.obj_true.shape)
probe_init = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(np.fft.fftshift(np.mean(nfsim.diffraction_mods, axis=0))) / H))

obj_init = (np.random.random(recons_shape) 
            * np.exp(1j * np.random.random(recons_shape) 
                     * np.pi))

view_indices = genViewIndices(positions, [obj_variable_npix, obj_variable_npix], probe_true.shape)indices = np.zeros(view_indices.shape) + np.arange(view_indices.shape[0])[:,None,None]
view_expanded = np.reshape(np.stack((indices, view_indices), axis=-1).astype('int32'), (*view_indices.shape, 2))

tf.reset_default_graph()

batch_size = 5

with tf.device('/gpu:0'):
    tf_obj_real = tf.Variable(np.real(obj_init), dtype='float32')
    tf_obj_imag = tf.Variable(np.imag(obj_init), dtype='float32')
    tf_obj = tf.complex(tf_obj_real, tf_obj_imag)
    
    tf_probe_real = tf.Variable(np.real(probe_init), dtype='float32')
    tf_probe_imag = tf.Variable(np.imag(probe_init), dtype='float32')
    tf_probe = tf.complex(tf_probe_real, tf_probe_imag)
    
    tf_mods = tf.constant(np.fft.fftshift(nfsim.diffraction_mods, axes=(1,2)), 
                           dtype='float32')
    
    
    #tf_view_indices = tf.constant(view_indices, dtype='int32')
    
    #tf_view_indices_expanded = tf.constant(view_expanded, dtype='int32')
    
    #batch_views = tf.ones((batch_size, probe_init.size), dtype='complex64')
    tf_kernel = tf.constant(H, dtype='complex64')

clipped_obj = tensor_clip(tf_obj, max_abs=1.0)
clipped_probe = tensor_clip(tf_probe, max_abs=1000.0)
#obj_clip_op = tf_obj.assign(clipped_obj)
#probe_clip_op = tf_probe.assign(clipped_probe)
obj_clip_op = [tf_obj_real.assign(tf.real(clipped_obj)),
               tf_obj_imag.assign(tf.imag(clipped_obj))]

probe_clip_op = [tf_probe_real.assign(tf.real(clipped_probe)),
               tf_probe_imag.assign(tf.imag(clipped_probe))]


tf_obj_1 = tf.pad(tf_obj, [[nfsim.obj_params.border_npix, nfsim.obj_params.border_npix],
                           [nfsim.obj_params.border_npix, nfsim.obj_params.border_npix]], constant_values=1)

tf_obj_real_pad = tf.pad(tf_obj_real, [[buffer_npix,buffer_npix],[buffer_npix,buffer_npix]], constant_values=1)
tf_obj_imag_pad = tf.pad(tf_obj_imag, [[buffer_npix,buffer_npix],[buffer_npix,buffer_npix]], constant_values=0.)

dataset_indices = tf.data.Dataset.range(diffs.shape[0])
dataset_indices = dataset_indices.shuffle(buffer_size=diffs.shape[0])
dataset_indices = dataset_indices.repeat()
dataset_batch = dataset_indices.batch(batch_size, drop_remainder=True)
dataset_batch = dataset_batch.apply(tf.data.experimental.prefetch_to_device('/gpu:0', buffer_size=5))



iterator = dataset_batch.make_one_shot_iterator()

batchi = iterator.get_next()
batch_indices = tf.Variable(tf.zeros(batch_size, dtype=tf.int64), trainable=False)
assign_op = batch_indices.assign(batchi)

batch_mods = tf.gather(tf_mods, batch_indices)



tf_obj_real_pads = []
tf_obj_imag_pads = []
n1 = nfsim.obj_params.obj_w_border_npix
n2 = nfsim.probe_params.npix
for p in nfsim.positions:
    
    padded_real = tf.pad(tf.real(tf_obj_1), [[p[0], n2 - (n1+p[0])],
                                           [p[1], n2 - (n1+p[1])]],
                        constant_values=1)
    padded_imag = tf.pad(tf.imag(tf_obj_1), [[p[0], n2 - (n1+p[0])],
                                           [p[1], n2 - (n1+p[1])]],
                        constant_values=0)
    tf_obj_real_pads.append(padded_real)
    tf_obj_imag_pads.append(padded_imag)

tf_obj_real_pads = tf.stack(tf_obj_real_pads)
tf_obj_imag_pads = tf.stack(tf_obj_imag_pads)
tf_obj_pads = tf.complex(tf_obj_real_pads, tf_obj_imag_pads)
batch_obj_views = tf.gather(tf_obj_pads, batch_indices)

batch_view_indices = tf.gather(tf_view_indices, batch_indices)
gen_view_fn = lambda view: tf.tensor_scatter_update(tensor=view[0], 
                                                    indices=tf.reshape(view[1], [-1,1]), 
                                                    updates=tf.reshape(tf_obj_1, [-1]))
batch_obj_views = tf.map_fn(fn=gen_view_fn, elems=[batch_views, batch_view_indices], dtype='complex64')
batch_obj_views = tf.reshape(batch_obj_views, [batch_size, *probe_init.shape])
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
batch_obj_views = tf.reshape(batch_obj_views, [batch_size, *probe_init.shape])batch_view_indices = tf.gather(tf_view_indices, batch_indices)
tf_obj_2 = tf_obj_1 - 1
gen_view_fn = lambda view_indices: tf.scatter_nd(tf.reshape(view_indices, [-1,1]), 
                                                 tf.reshape(tf_obj_2, [-1]),
                                                 [probe_init.size])
batch_obj_views = tf.map_fn(gen_view_fn, batch_view_indices, dtype=tf.complex64)
batch_obj_views = tf.reshape(batch_obj_views, [batch_size, *probe_init.shape]) + 1batch_obj_view_indices = tf.gather(tf_view_indices_expanded, batch_indices)
obj_repeated = tf.tile(tf.reshape(tf_obj_1, [-1]), [batch_size])
batch_obj_views = tf.tensor_scatter_update(batch_views, tf.reshape(batch_obj_view_indices, [-1,2]), obj_repeated)
batch_obj_views = tf.reshape(batch_obj_views, [batch_size, *probe_init.shape])

batch_out_wavefronts = (tf.ifft2d(tf.fft2d(batch_obj_views * 
                                          tf_probe) 
                                 / nfsim.probe_params.npix * tf_kernel) 
                        * nfsim.probe_params.npix)



loss = 0.5 * tf.reduce_sum((tf.abs(batch_out_wavefronts) -  batch_mods)**2)



lr_probe_placeholder = tf.placeholder(dtype=tf.float32)
lr_obj_placeholder = tf.placeholder(dtype=tf.float32)

probe_opt = tf.train.AdamOptimizer(learning_rate=lr_probe_placeholder)
probe_min = probe_opt.minimize(loss, var_list=[tf_probe_real, tf_probe_imag])
obj_opt = tf.train.AdamOptimizer(learning_rate=lr_obj_placeholder)
obj_min = obj_opt.minimize(loss, var_list=[tf_obj_real, tf_obj_imag])



init = tf.global_variables_initializer()
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True    
#sess_config.allow_soft_placement=True
session = tf.Session(config=sess_config)
session.run(init)



lossvals = []



pr_obj = session.run(tf_obj)
pr_probe = session.run(tf_probe)



_, __, lossval = session.run([obj_min, probe_min, loss], 
                             feed_dict={lr_obj_placeholder: 1e-2,
                                        lr_probe_placeholder: 1e1})



get_ipython().run_cell_magic('time', '', 'for i in range(100):\n    _, __, lossval = session.run([obj_min, probe_min, loss], \n                             feed_dict={lr_obj_placeholder: 1e-2,\n                                        lr_probe_placeholder: 1e1})\n    ')



# tensor-Scatter approach
# bs 5: 11 s
# bs 10: 19s

# Stacking the views
# bs 10: 5.57s
# bs 5: 3.7 s

# Scatter nd approach (this is buggy - the gpu version does not handle complex numbers correctly)
# bs 5: 3.2s
# bs 10: 6.16s

# Scatter nd approach for separate real and imaginary parts
# bs 10: 8.23s



get_ipython().run_cell_magic('time', '', 'for i_in in range(50000):\n    session.run(assign_op)\n    _, __, lossval = session.run([obj_min, probe_min, loss],\n                                 feed_dict={lr_obj_placeholder: 1e-1,\n                                            lr_probe_placeholder: 1e1})\n    lossvals.append(lossval)\n    \n    #if i_in % 100 == 0: session.run([obj_clip_op])#, probe_clip_op])\n    if not i_in % 500 == 0: continue\n\n    pr_obj, pr_probe = session.run([tf_obj, tf_probe])\n    probe_errs = register_translation(nfsim.probe_true, \n                                   pr_probe,\n                                   upsample_factor=10)\n    obj_errs = register_translation(nfsim.obj_true, pr_obj, upsample_factor=10)\n    print(i_in, probe_errs[1], obj_errs[1], lossval)')



recons_obj, recons_probe = session.run([tf_obj, tf_probe])
recons_obj = recons_obj * np.exp(-1j * circmean(np.angle(recons_obj)))
#recons_probe = recons_probe * np.exp(-1j * circmean(np.angle(recons_probe)))



np.angle(recons_obj).max(), np.angle(recons_obj).min()



circmean(np.angle(recons_obj))



plt.plot(np.log(lossvals))
plt.show()



recons_obj = np.load('vanilla_adam_recons_obj.npy')
recons_probe = np.load('vanilla_adam_recons_probe.npy')
probe_true = np.load('probe_true.npy')



test_shape = np.array(probe_true.shape) // 2 - np.array(probe_true.shape) // 4
s0 = test_shape[0] 

p1 = s0
p2 = -s0

obj_errs_2 = register_translation(obj_true, recons_obj * np.exp(1j * 0), upsample_factor=10)

probe_errs = register_translation(probe_true[p1:p2, p1:p2], recons_probe[p1:p2, p1:p2] * np.exp(1j * 0.95) , upsample_factor=10)
probe_errs_2 = register_translation(probe_true, recons_probe * np.exp(1j * 0.95), upsample_factor=10)
print('Obj')
print(obj_errs_2)
print('Probe')
probe_errs, probe_errs_2



test_obj = obj_true

test_probe = probe_true

plt.figure(figsize=[12, 6])
plt.subplot(2,4,1)
plt.pcolormesh(np.abs(test_obj), cmap='gray')
plt.title('True obj. ampl.', fontsize=17)
plt.subplot(2,4,2)
plt.pcolormesh(np.angle(test_obj), cmap='gray')
plt.title('True obj. phase', fontsize=17)
plt.colorbar()
plt.subplot(2,4,3)
plt.pcolormesh(np.abs(test_probe))
plt.title('True probe ampl', fontsize=17)
plt.colorbar()
plt.subplot(2,4,4)
plt.pcolormesh(np.angle(test_probe))
plt.colorbar()
plt.title('True probe phase', fontsize=17)
plt.subplot(2,4,5)
plt.pcolormesh(np.abs(recons_obj), cmap='gray')
plt.title('Reconstr. obj. ampl.', fontsize=17)
plt.colorbar()
plt.subplot(2,4,6)
plt.pcolormesh(np.angle(recons_obj), cmap='gray')
plt.colorbar()
plt.title('Reconstr. obj. phase', fontsize=17)
plt.subplot(2,4,7)
plt.pcolormesh(np.abs(recons_probe))
plt.colorbar()
plt.title('Reconst. probe. ampl.', fontsize=17)
plt.subplot(2,4,8)
plt.pcolormesh(np.angle(recons_probe))
plt.vlines([s0, probe_npix-s0], s0, probe_npix-s0, colors='red', linewidth=3)
plt.hlines([s0, probe_npix-s0], s0, probe_npix-s0, colors='red', linewidth=3)
plt.title('Reconst. probe. angle.', fontsize=17)
plt.colorbar()
plt.tight_layout()
plt.show()





