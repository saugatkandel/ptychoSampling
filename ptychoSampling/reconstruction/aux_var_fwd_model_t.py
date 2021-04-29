import tensorflow as tf
import numpy as np
import abc
from typing import Tuple, Union, Callable
from ptychoSampling.obj import Obj
from ptychoSampling.probe import Probe, Probe3D
from ptychoSampling.reconstruction.wavefront_t import propFF_t, fftshift_t, propTF_t
from ptychoSampling.grid import ScanGrid, BraggPtychoGrid
from ptychoSampling.logger import logger
from ptychoSampling.reconstruction.forwardmodel_t import ForwardModelT, FarfieldForwardModelT

class AuxVarForwardModel(ForwardModelT):
    def __init__(self, *args:int , **kwargs:int):
        super().__init__(*args, **kwargs)

        with tf.device("/gpu:0"):
            ny, nx = self.probe_cmplx_t.get_shape().as_list()
            nz = self._scan_grid.positions_pix.shape[0]
            self.aux_v, self.aux_cmplx_t = self._addComplexVariable(np.zeros((nz, ny, nx)),
                                                                    name="auxiliary")

class FarfieldAuxVarForwardModel(AuxVarForwardModel):

    def predict(self, position_indices_t: tf.Tensor, scope_name:str="") -> tf.Tensor:

        scope_name = scope_name + "_predict" if scope_name else "predict"
        ndiffs = position_indices_t.get_shape().as_list()[0]

        with tf.name_scope(scope_name) as scope:
            if ndiffs == 0:
                return tf.zeros(shape=[], dtype='complex64', name=scope)

            batch_obj_views_t = tf.gather(self._obj_views_all_t, position_indices_t)
            batch_obj_views_t = fftshift_t(batch_obj_views_t)
            exit_waves_t = batch_obj_views_t *  self.probe_cmplx_t

            farfield_waves_t = propFF_t(exit_waves_t)
            return farfield_waves_t#tf.reshape(tf.stack([tf.real(farfield_waves_t), tf.imag(farfield_waves_t)]), [-1])

    def predict_second_order(self, obj_v: tf.Variable,
                             probe_v: tf.Variable,
                             position_indices_t: tf.Tensor,
                             scope_name:str="") -> tf.Tensor:

        scope_name = scope_name + "_predict" if scope_name else "predict"
        ndiffs = position_indices_t.get_shape().as_list()[0]

        with tf.name_scope(scope_name) as scope:
            if ndiffs == 0:
                return tf.zeros(shape=[], dtype='float32', name="scope")

            obj_cmplx_t = self._reshapeComplexVariable(obj_v, self._obj.shape)
            obj_w_border_t = tf.pad(obj_cmplx_t, self._obj.border_shape, constant_values=self._obj.border_const)

            probe_cmplx_t = self._reshapeComplexVariable(probe_v, self._probe.shape)

            batch_obj_views_t = self._getPtychoObjViewStack(obj_w_border_t, probe_cmplx_t, position_indices_t)
            batch_obj_views_t = fftshift_t(batch_obj_views_t)
            exit_waves_t = batch_obj_views_t * probe_cmplx_t
            farfield_waves_t = propFF_t(exit_waves_t)
            return tf.reshape(tf.stack([tf.real(farfield_waves_t), tf.imag(farfield_waves_t)]), [-1])

class FarfieldPALMForwardModel(FarfieldAuxVarForwardModel):

    def exact_solve_aux_var(self, predicted_ff_waves_t, measured_ff_amplitudes_t):
        phases_t = tf.angle(predicted_ff_waves_t)
        aux_update_t = (tf.cast(measured_ff_amplitudes_t, tf.complex64) *
                        tf.complex(tf.cos(phases_t), tf.sin(phases_t)))
        return aux_update_t
