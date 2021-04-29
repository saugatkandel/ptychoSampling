import tensorflow as tf
import numpy as np
import abc
from typing import Tuple, Union, Callable
from ptychoSampling.obj import Obj
from ptychoSampling.probe import Probe, Probe3D
from ptychoSampling.reconstruction.wavefront_t import propFF_t, fftshift_t, propTF_t
from ptychoSampling.grid import ScanGrid, BraggPtychoGrid
from ptychoSampling.logger import logger

__all__ = ["ForwardModelT",
           "FarfieldForwardModelT",
           "NearfieldForwardModelT",
           "BraggPtychoForwardModelT"]

class ForwardModelT(abc.ABC):

    def __init__(self,
                 obj: Obj,
                 probe: Probe,
                 scan_grid: ScanGrid,
                 obj_abs_proj: bool = False,
                 obj_abs_max: float = 1.0,
                 probe_abs_proj: bool = False,
                 probe_abs_max: float = 1.0,
                 upsampling_factor: int = 1,
                 setup_second_order: bool = False,
                 dtype: str='float32'):

        self._obj = obj
        self._probe = probe
        self._scan_grid = scan_grid
        self._dtype = dtype

        self.model_vars = {}

        with tf.device("/gpu:0"):
            obj_constraint_fn = None
            if obj_abs_proj:
                obj_constraint_fn = lambda t: self._complexProjectionOp(t, obj_abs_max)
            self.obj_v, self.obj_cmplx_t = self._addComplexVariable(obj.array,
                                                                    constraint_fn=obj_constraint_fn,
                                                                    name="obj",
                                                                    dtype=self._dtype)
            self.obj_w_border_t = tf.pad(self.obj_cmplx_t, obj.border_shape, constant_values=obj.border_const)


            probe_constraint_fn = None
            if probe_abs_proj:
                probe_constraint_fn = lambda t: self._complexProjectionOp(t, probe_abs_max)
            self.probe_v, self.probe_cmplx_t = self._addComplexVariable(probe.wavefront,
                                                                        name="probe",
                                                                        constraint_fn=probe_constraint_fn,
                                                                        dtype=self._dtype)

            self.upsampling_factor = upsampling_factor

            if not setup_second_order:
                self._obj_views_all_t = self._getPtychoObjViewStack(self.obj_w_border_t, self.probe_cmplx_t)

        #self.scan_positions_pix = scan_positions_pix

    def _addRealVariable(self, init: Union[float, np.ndarray],
                         name: str,
                         constraint_fn: Callable = None,
                         dtype: str='float32') -> Tuple[tf.Variable, tf.Tensor]:
        var = tf.Variable(init.flat,
                          constraint=constraint_fn,
                          dtype=dtype,
                          name=name)
        output = tf.reshape(var, init.shape)
        self.model_vars[name] = {"variable": var,
                                 "output": output}
        return var, output

    def _addComplexVariable(self, init: Union[complex, np.ndarray],
                            name: str,
                            constraint_fn: Callable = None,
                            dtype: str = "float32" ) -> Tuple[tf.Variable, tf.Tensor]:
        init_reals = np.concatenate((np.real(init).flat, np.imag(init).flat))
        var = tf.Variable(init_reals, constraint=constraint_fn, dtype=dtype, name=name)
        output = self._reshapeComplexVariable(var, init.shape)
        self.model_vars[name] = {"variable": var,
                                 "output": output}
        return var, output

    @staticmethod
    def _reshapeComplexVariable(var: tf.Variable, shape: tuple) -> tf.Tensor:
        var_reshaped = tf.reshape(var, [2, *shape])
        output = tf.complex(var_reshaped[0], var_reshaped[1])
        return output

    @staticmethod
    def _complexProjectionOp(var: tf.Variable, max_abs: float = 1.0) -> tf.Tensor:
        """Projection operation for a complex-valued variable expressed in terms of its real and imaginary parts."""
        var_reshaped_t = tf.reshape(var, [2, -1])
        var_clipped_t = tf.clip_by_norm(var_reshaped_t, max_abs, axes=[0])
        var_clipped_reshaped_t = tf.reshape(var_clipped_t, [-1])
        return var_clipped_reshaped_t


    def _getPtychoObjViewStack(self, obj_w_border_t: tf.Tensor,
                               probe_cmplx_t: tf.Tensor,
                               position_indices_t: tf.Tensor=None) -> tf.Tensor:
        """Precalculate the object positioning for each scan position.

        Assumes a small object that is translated within the dimensions of a full-field probe. For each scan
        position, we translate the object, then pad the object array to the size of the probe beam. For the padding,
        we assume free-space (transparent) propagation and use 1.0.

        In Tensorflow, performing the pad-and-stack procedure in the GPU for complex -valued arrays seems to be
        buggy. As a workaround, we separately pad-and-stack the real and imaginary parts of the object with 1.0 and
        0 respectively.

        Returns
        ----------
        obj_views : tensor(complex)
            Stack of tensors that correspond to the padded object at each object translation.
        """
        if position_indices_t is None:
            position_indices_t = tf.range(self._scan_grid.positions_pix.shape[0])

        logger.info("Creating obj views for the scan positions.")
        if not self._scan_grid.full_field_probe:
            if not hasattr(self, "_obj_view_indices_t"):
                obj_view_indices = self._genViewIndices(self._scan_grid.positions_pix)
                self._obj_view_indices_t = tf.constant(obj_view_indices)

            batch_obj_view_indices_t = tf.gather(self._obj_view_indices_t, position_indices_t)
            batch_obj_views_t = tf.gather(tf.reshape(obj_w_border_t, [-1]), batch_obj_view_indices_t)
        else:
            batch_obj_views_t = self._getFullFieldProbeObjViews(obj_w_border_t,
                                                                probe_cmplx_t,
                                                                position_indices_t)

        return batch_obj_views_t


    def _genViewIndices(self, scan_positions_pix: np.ndarray) -> np.ndarray:
        """ Generate the indices...

        Parameters
        ----------
        scan_positions_pix

        Returns
        -------

        """
        ny, nx = self.probe_cmplx_t.get_shape().as_list()
        views_indices_all = []
        ony, onx = self.obj_w_border_t.get_shape().as_list()
        for py, px in scan_positions_pix:
            Y, X = np.ogrid[py:ny + py, px:nx + px]
            view_single = (Y % ony) * onx + (X % onx)
            views_indices_all.append(view_single)

        return np.array(views_indices_all)

    def _getFullFieldProbeObjViews(self,
                                   obj_w_border_t: tf.Tensor,
                                   probe_cmplx_t: tf.Tensor,
                                   position_indices_t: tf.Tensor) -> tf.Tensor:
        """Precalculate the object positioning for each scan position.

        Assumes a small object that is translated within the dimensions of a full-field probe. For each scan
        position, we translate the object, then pad the object array to the size of the probe beam. For the padding,
        we assume free-space (transparent) propagation and use 1.0.

        In Tensorflow, performing the pad-and-stack procedure in the GPU for complex -valued arrays seems to be
        buggy. As a workaround, we separately pad-and-stack the real and imaginary parts of the object with 1.0 and
        0 respectively.

        Returns
        ----------
        obj_views : tensor(complex)
            Stack of tensors that correspond to the padded object at each object translation.
        """
        obj_real_pads = []
        obj_imag_pads = []

        ony, onx = obj_w_border_t.get_shape().as_list()
        pny, pnx = probe_cmplx_t.get_shape().as_list()

        padfn = lambda p, t, c: tf.pad(t, [[p[0],  pny - (ony + p[0])], [p[1], pnx - (onx + p[1])]],
                                 constant_values=c)

        for i in range(position_indices_t.get_shape().as_list()[0]):
            p = self._scan_grid.positions_pix[i]
            padded_real = padfn(p, tf.real(obj_w_border_t), 1.0)
            padded_imag = padfn(p, tf.imag(obj_w_border_t), 0.)
            obj_real_pads.append(padded_real)
            obj_imag_pads.append(padded_imag)

        obj_real_pads_t = tf.stack(obj_real_pads)
        obj_imag_pads_t = tf.stack(obj_imag_pads)

        obj_views_t = tf.complex(obj_real_pads_t, obj_imag_pads_t)
        return obj_views_t

    def _downsample(self, amplitudes_t: tf.Tensor) -> tf.Tensor:
        diffs_t = amplitudes_t ** 2
        px = self._probe.shape[-1][-1]
        py = self._probe.shape[-2]
        u = self.upsampling_factor
        px_new = px // u
        py_new = py // u
        diffs_t = tf.reshape(diffs_t, [-1, px_new, u, py_new, u])
        diffs_t = tf.reduce_sum(diffs_t, axis=(-1, -3))
        return diffs_t ** 0.5


    @abc.abstractmethod
    def predict(self, position_indices_t: tf.Tensor, scope_name: str="") -> tf.Tensor:
        pass

    @abc.abstractmethod
    def predict_second_order(self, obj_v: tf.Variable,
                             probe_v: tf.Variable,
                             position_indices_t: tf.Tensor,
                             scope_name: str="") -> tf.Tensor:
        pass

class FarfieldForwardModelT(ForwardModelT):

    def predict(self, position_indices_t: tf.Tensor, scope_name:str="") -> tf.Tensor:

        scope_name = scope_name + "_predict" if scope_name else "predict"
        ndiffs = position_indices_t.get_shape().as_list()[0]

        with tf.name_scope(scope_name) as scope:
            if ndiffs == 0:
                return tf.zeros(shape=[], dtype=self._dtype, name=scope)

            batch_obj_views_t = tf.gather(self._obj_views_all_t, position_indices_t, name="test_gather")
            batch_obj_views_t = fftshift_t(batch_obj_views_t)
            exit_waves_t = batch_obj_views_t *  self.probe_cmplx_t
            out_wavefronts_t = propFF_t(exit_waves_t)
            amplitudes_t = tf.abs(out_wavefronts_t)
            if self.upsampling_factor > 1:
                amplitudes_t = self._downsample(amplitudes_t)
            return tf.reshape(amplitudes_t, [-1], name=scope)

    def predict_second_order(self, obj_v: tf.Variable,
                             probe_v: tf.Variable,
                             position_indices_t: tf.Tensor,
                             scope_name:str="") -> tf.Tensor:

        scope_name = scope_name + "_predict" if scope_name else "predict"
        ndiffs = position_indices_t.get_shape().as_list()[0]

        with tf.name_scope(scope_name) as scope:
            if ndiffs == 0:
                return tf.zeros(shape=[], dtype=self._dtype, name="scope")

            obj_cmplx_t = self._reshapeComplexVariable(obj_v, self._obj.shape)
            obj_w_border_t = tf.pad(obj_cmplx_t, self._obj.border_shape, constant_values=self._obj.border_const)

            probe_cmplx_t = self._reshapeComplexVariable(probe_v, self._probe.shape)

            batch_obj_views_t = self._getPtychoObjViewStack(obj_w_border_t, probe_cmplx_t, position_indices_t)
            batch_obj_views_t = fftshift_t(batch_obj_views_t)
            exit_waves_t = batch_obj_views_t * probe_cmplx_t
            out_wavefronts_t = propFF_t(exit_waves_t)
            amplitudes_t = tf.abs(out_wavefronts_t)
            if self.upsampling_factor > 1:
                amplitudes_t = self._downsample(amplitudes_t)
            return tf.reshape(amplitudes_t, [-1])


class NearfieldForwardModelT(ForwardModelT):
    def __init__(self, obj: Obj,
                 probe: Probe,
                 scan_grid: ScanGrid,
                 propagation_dist: float,
                 upsampling_factor: int = 1,
                 setup_second_order: bool = False,
                 dtype: str = "float32"):
        super().__init__(obj, probe, scan_grid, upsampling_factor, setup_second_order, dtype)
        with tf.device("/gpu:0"):
            _, self._transfer_function = propTF_t(self.probe_cmplx_t,
                                                  wavelength=probe.wavelength,
                                                  pixel_size=probe.pixel_size,
                                                  prop_dist=propagation_dist,
                                                  return_transfer_function=True)

    def predict(self, position_indices_t: tf.Tensor, scope_name: str="") -> tf.Tensor:

        ndiffs = position_indices_t.get_shape().as_list()[0]
        scope_name = scope_name + "_predict" if scope_name else "predict"

        with tf.name_scope(scope_name) as scope:
            if ndiffs == 0:
                return tf.zeros(shape=[], dtype=self._dtype, name=scope)

            batch_obj_views_t = tf.gather(self._obj_views_all_t, position_indices_t)
            batch_obj_views_t = fftshift_t(batch_obj_views_t)
            exit_waves_t = batch_obj_views_t * self.probe_cmplx_t

            out_wavefronts_t = propTF_t(exit_waves_t,
                                        reuse_transfer_function=True,
                                        transfer_function=self._transfer_function)
            amplitudes_t = tf.abs(out_wavefronts_t)
            if self.upsampling_factor > 1:
                amplitudes_t = self._downsample(amplitudes_t)
            return tf.reshape(amplitudes_t, [-1], name=scope)

    def predict_second_order(self, obj_v: tf.Variable,
                             probe_v: tf.Variable,
                             position_indices_t: tf.Tensor,
                             scope_name: str="") -> tf.Tensor:
        ndiffs = position_indices_t.get_shape().as_list()[0]
        scope_name = scope_name + "_predict" if scope_name else "predict"

        with tf.name_scope(scope_name) as scope:
            if ndiffs == 0:
                return tf.zeros(shape=[], dtype=self._dtype, name=scope)

            obj_cmplx_t = self._reshapeComplexVariable(obj_v, self._obj.shape)
            obj_w_border_t = tf.pad(obj_cmplx_t, self._obj.border_shape, constant_values=self._obj.border_const)

            probe_cmplx_t = self._reshapeComplexVariable(probe_v, self._probe.shape)

            batch_obj_views_t = self._getPtychoObjViewStack(obj_w_border_t, probe_cmplx_t, position_indices_t)
            batch_obj_views_t = fftshift_t(batch_obj_views_t)
            exit_waves_t = batch_obj_views_t * probe_cmplx_t

            out_wavefronts_t = propTF_t(exit_waves_t,
                                        reuse_transfer_function=True,
                                        transfer_function=self._transfer_function)
            amplitudes_t = tf.abs(out_wavefronts_t)
            if self.upsampling_factor > 1:
                amplitudes_t = self._downsample(amplitudes_t)
            return tf.reshape(amplitudes_t, [-1], name=scope)


class BraggPtychoForwardModelT(ForwardModelT):
    def __init__(self, obj: Obj,
                 probe: Probe3D,
                 scan_grid: BraggPtychoGrid,
                 exit_wave_axis: str = "y",
                 upsampling_factor: int = 1,
                 setup_second_order: bool = False,
                 dtype: str = "float32"):
        if scan_grid.full_field_probe:
            e = ValueError("Full field probe not supported for Bragg ptychography.")
            logger.error(e)
            raise e
        if scan_grid.grid2d_axes != ("y", "z"):
            e = ValueError("Only supports the case where the ptychographic scan is on the yz-plane.")
            logger.error(e)
            raise e
        if exit_wave_axis != 'y':
            e = ValueError("Only supports the case where the exit waves are output along the y-direction.")
            logger.error(e)
            raise e

        super().__init__(obj, probe, scan_grid, upsampling_factor, setup_second_order, dtype)
        logger.info("Creating the phase modulations for the scan angles.")

        with tf.device("/gpu:0"):
            self._probe_phase_modulations_all_t = tf.constant(self._getProbePhaseModulationsStack(),
                                                              dtype='complex64')
            self._full_rc_positions_indices_t = tf.constant(scan_grid.full_rc_positions_indices, dtype='int64')


    def predict(self, position_indices_t: tf.Tensor,
                scope_name: str="") -> tf.Tensor:
        ndiffs = position_indices_t.get_shape().as_list()[0]
        scope_name = scope_name + "_predict" if scope_name else "predict"

        with tf.name_scope(scope_name) as scope:
            if ndiffs == 0:
                return tf.zeros(shape=[], dtype=self._dtype, name=scope)

            batch_rc_positions_indices = tf.gather(self._full_rc_positions_indices_t, position_indices_t)
            batch_obj_views_t = tf.gather(self._obj_views_all_t, batch_rc_positions_indices[:, 1])
            batch_phase_modulations_t = tf.gather(self._probe_phase_modulations_all_t, batch_rc_positions_indices[:, 0])

            batch_obj_views_t = batch_obj_views_t
            exit_waves_t = batch_obj_views_t * self.probe_cmplx_t * batch_phase_modulations_t
            exit_waves_proj_t = fftshift_t(tf.reduce_sum(exit_waves_t, axis=-3))

            out_wavefronts_t = propFF_t(exit_waves_proj_t)
            amplitudes_t = tf.abs(out_wavefronts_t)
            if self.upsampling_factor > 1:
                amplitudes_t = self._downsample(amplitudes_t)
            return tf.reshape(amplitudes_t, [-1], name=scope)

    def predict_second_order(self, obj_v: tf.Variable,
                             probe_v: tf.Variable,
                             position_indices_t: tf.Tensor,
                             scope_name: str="") -> tf.Tensor:
        ndiffs = position_indices_t.get_shape().as_list()[0]
        scope_name = scope_name + "_predict" if scope_name else "predict"

        with tf.name_scope(scope_name) as scope:
            if ndiffs == 0:
                return tf.zeros(shape=[], dtype=self._dtype, name=scope)
            obj_cmplx_t = self._reshapeComplexVariable(obj_v, self._obj.shape)
            obj_w_border_t = tf.pad(obj_cmplx_t, self._obj.border_shape, constant_values=self._obj.border_const)

            probe_cmplx_t = self._reshapeComplexVariable(probe_v, self._probe.shape)


            batch_rc_positions_indices = tf.gather(self._full_rc_positions_indices_t, position_indices_t)
            batch_obj_views_t = self._getPtychoObjViewStack(obj_w_border_t, probe_cmplx_t,
                                                            batch_rc_positions_indices[:,1])#position_indices_t)


            batch_phase_modulations_t = tf.gather(self._probe_phase_modulations_all_t, batch_rc_positions_indices[:, 0])

            batch_obj_views_t = batch_obj_views_t
            exit_waves_t = batch_obj_views_t * self.probe_cmplx_t * batch_phase_modulations_t
            exit_waves_proj_t = fftshift_t(tf.reduce_sum(exit_waves_t, axis=-3))

            out_wavefronts_t = propFF_t(exit_waves_proj_t)
            amplitudes_t = tf.abs(out_wavefronts_t)
            if self.upsampling_factor > 1:
                amplitudes_t = self._downsample(amplitudes_t)
            return tf.reshape(amplitudes_t, [-1], name=scope)

    def _genViewIndices(self, scan_positions_pix: np.ndarray) -> np.ndarray:
        """ Generate the indices...

        Parameters
        ----------
        obj
        probe
        scan_positions_pix

        Returns
        -------

        """
        ny, nx, nz = self.probe_cmplx_t.get_shape().as_list()
        ony, onx, onz = self.obj_w_border_t.get_shape().as_list()

        views_indices_all = []
        for py, pz in scan_positions_pix:
            Y, X, Z = np.ogrid[py: py + ny, 0: nx , pz: pz + nz]
            view_single = ((Y % ony) * onx + (X % onx)) * onz + (Z % onz)
            views_indices_all.append(view_single)
        return np.array(views_indices_all)

    def _getPtychoObjViewStack(self,  obj_w_border_t: tf.Tensor,
                               probe_cmplx_t: tf.Tensor,
                               position_indices_t: tf.Tensor=None) ->tf.Tensor:
        """Precalculate the object positioning for each scan position.

        Assumes a small object that is translated within the dimensions of a full-field probe. For each scan
        position, we translate the object, then pad the object array to the size of the probe beam. For the padding,
        we assume free-space (transparent) propagation and use 1.0.

        In Tensorflow, performing the pad-and-stack procedure in the GPU for complex -valued arrays seems to be
        buggy. As a workaround, we separately pad-and-stack the real and imaginary parts of the object with 1.0 and
        0 respectively.

        Returns
        ----------
        obj_views : tensor(complex)
            Stack of tensors that correspond to the padded object at each object translation.
        """
        if position_indices_t is None:
            position_indices_t = tf.range(self._scan_grid.positions_pix.shape[0])

        logger.info("Creating obj views for the scan positions.")
        if not hasattr(self, "_obj_view_indices_t"):
            obj_view_indices = self._genViewIndices(self._scan_grid.positions_pix)
            self._obj_view_indices_t = tf.constant(obj_view_indices)

        batch_obj_view_indices_t = tf.gather(self._obj_view_indices_t, position_indices_t)
        batch_obj_views_t = tf.gather(tf.reshape(obj_w_border_t, [-1]), batch_obj_view_indices_t)

        #obj_view_indices = self._genViewIndices(scan_grid.positions_pix)
        #obj_view_indices_t = tf.constant(obj_view_indices, dtype='int64')
        #obj_views_t = tf.gather(tf.reshape(self.obj_w_border_t, [-1]), obj_view_indices_t)
        #obj_views_t = tf.reshape(obj_views_t,
        #                         (obj_view_indices.shape[0],
        #                          *(self.probe_cmplx_t.get_shape().as_list())))
        return batch_obj_views_t

    def _getProbePhaseModulationsStack(self) -> np.ndarray:
        ttheta = self._scan_grid.two_theta
        domega = self._scan_grid.del_omega

        ki = 2 * np.pi / self._probe.wavelength * np.array([np.cos(ttheta), np.sin(ttheta), 0])
        kf = 2 * np.pi / self._probe.wavelength * np.array([1, 0, 0])
        q = (kf - ki)[:, None]

        ki_new = 2 * np.pi / self._probe.wavelength * np.array([np.cos(ttheta + self._scan_grid.rc_angles),
                                                         np.sin(ttheta + self._scan_grid.rc_angles),
                                                         0 * self._scan_grid.rc_angles])
        kf_new = 2 * np.pi / self._probe.wavelength * np.array([np.cos(self._scan_grid.rc_angles),
                                                         np.sin(self._scan_grid.rc_angles),
                                                         0 * self._scan_grid.rc_angles])
        q_new = kf_new - ki_new
        delta_q = q_new - q
        # Probe dimensions in real space (assumes even shape)
        position_grids = [np.arange(-s // 2, s // 2) * ds for (s, ds) in zip(self._probe.shape,
                                                                             self._probe.pixel_size)]
        Ry, Rx, Rz = np.meshgrid(*position_grids, indexing='ij')
        phase_modulations_all = np.exp(1j * np.array([delta_q[0, i] * Ry
                                                      + delta_q[1, i] * Rx
                                                      + delta_q[2, i] * Rz
                                                      for i in range(self._scan_grid.n_rc_angles)]))
        return phase_modulations_all


class JointForwardModelT(ForwardModelT):

    def __init__(self,
                 obj: Obj,
                 probe: Probe,
                 scan_grid: ScanGrid,
                 obj_abs_proj: bool = False,
                 obj_abs_max: float = 1.0,
                 probe_abs_max: float = None,
                 upsampling_factor: int = 1,
                 setup_second_order: bool = False,
                 dtype:str = "float32"):

        self._obj = obj
        self._probe = probe
        self._scan_grid = scan_grid
        self._dtype = dtype

        self.model_vars = {}

        joint_var_array = np.concatenate((obj.array.flat,
                                                probe.wavefront.flat))
        self._joint_array_shape = joint_var_array.shape

        with tf.device("/gpu:0"):
            joint_constraint_fn = None
            if obj_abs_proj:
                joint_constraint_fn = lambda t: self._complexProjectionOp(t,
                                                                          obj.array.size,
                                                                          obj_abs_max,
                                                                          probe_abs_max)
            self.joint_v, self.joint_cmplx_t = self._addComplexVariable(joint_var_array,
                                                                        constraint_fn=joint_constraint_fn,
                                                                        name='joint_var',
                                                                        dtype=self._dtype)
            obj_t_flat = self.joint_cmplx_t[:obj.array.size]
            self.obj_cmplx_t = tf.reshape(obj_t_flat, obj.shape)

            probe_t_flat = self.joint_cmplx_t[obj.array.size:]
            self.probe_cmplx_t = tf.reshape(probe_t_flat, probe.shape)

            self.obj_w_border_t = tf.pad(self.obj_cmplx_t, obj.border_shape, constant_values=obj.border_const)

            self.upsampling_factor = upsampling_factor

            if not setup_second_order:
                self._obj_views_all_t = self._getPtychoObjViewStack(self.obj_w_border_t, self.probe_cmplx_t)

        #self.scan_positions_pix = scan_positions_pix

    @staticmethod
    def _complexProjectionOp(joint_v: tf.Variable,
                             obj_cmplx_size: int,
                             obj_max_abs: float = 1.0,
                             probe_max_abs: float = None) -> tf.Tensor:
        """Projection operation for a complex-valued variable expressed in terms of its real and imaginary parts."""

        reshaped_joint_t = tf.reshape(joint_v, [2, -1])

        obj_reals_t = reshaped_joint_t[:, :obj_cmplx_size]
        obj_clipped_t = tf.clip_by_norm(obj_reals_t, obj_max_abs, axes=[0])
        probe_reals_t = reshaped_joint_t[:, obj_cmplx_size:]
        if probe_max_abs is not None:
            probe_clipped_t = tf.clip_by_norm(probe_reals_t, probe_max_abs, axes=[0])
        else:
            probe_clipped_t = probe_reals_t

        clipped_joint_t = tf.concat((obj_clipped_t, probe_clipped_t), axis=1)
        return tf.reshape(clipped_joint_t, [-1])



    @abc.abstractmethod
    def predict_second_order(self, obj_v: tf.Variable,
                             probe_v: tf.Variable,
                             position_indices_t: tf.Tensor,
                             scope_name: str="") -> tf.Tensor:
        pass

class JointFarfieldForwardModelT(JointForwardModelT):

    def predict_second_order(self, joint_v: tf.Variable,
                             position_indices_t: tf.Tensor,
                             scope_name:str="") -> tf.Tensor:

        scope_name = scope_name + "_predict" if scope_name else "predict"
        ndiffs = position_indices_t.get_shape().as_list()[0]

        with tf.name_scope(scope_name) as scope:
            if ndiffs == 0:
                return tf.zeros(shape=[], dtype=self._dtype, name="scope")

            joint_t = self._reshapeComplexVariable(joint_v, self._joint_array_shape)

            obj_t_flat = joint_t[:self._obj.array.size]
            probe_t_flat = joint_t[self._obj.array.size:]
            obj_cmplx_t = tf.reshape(obj_t_flat, self._obj.shape)
            probe_cmplx_t = tf.reshape(probe_t_flat, self._probe.shape)

            obj_w_border_t = tf.pad(obj_cmplx_t, self._obj.border_shape, constant_values=self._obj.border_const)

            batch_obj_views_t = self._getPtychoObjViewStack(obj_w_border_t, probe_cmplx_t, position_indices_t)
            batch_obj_views_t = fftshift_t(batch_obj_views_t)
            exit_waves_t = batch_obj_views_t * probe_cmplx_t
            out_wavefronts_t = propFF_t(exit_waves_t)
            amplitudes_t = tf.abs(out_wavefronts_t)
            if self.upsampling_factor > 1:
                amplitudes_t = self._downsample(amplitudes_t)
            return tf.reshape(amplitudes_t, [-1])


    def predict(self, position_indices_t: tf.Tensor, scope_name:str="") -> tf.Tensor:

        scope_name = scope_name + "_predict" if scope_name else "predict"
        ndiffs = position_indices_t.get_shape().as_list()[0]

        with tf.name_scope(scope_name) as scope:
            if ndiffs == 0:
                return tf.zeros(shape=[], dtype=self._dtype, name=scope)

            batch_obj_views_t = tf.gather(self._obj_views_all_t, position_indices_t, name="test_gather")
            batch_obj_views_t = fftshift_t(batch_obj_views_t)
            exit_waves_t = batch_obj_views_t *  self.probe_cmplx_t
            out_wavefronts_t = propFF_t(exit_waves_t)
            amplitudes_t = tf.abs(out_wavefronts_t)
            if self.upsampling_factor > 1:
                amplitudes_t = self._downsample(amplitudes_t)
            return tf.reshape(amplitudes_t, [-1], name=scope)
