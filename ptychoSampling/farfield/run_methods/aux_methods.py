import numpy as np
import tensorflow as tf
from typing import Callable
from ptychoSampling.reconstruction.optimization_t import LMAOptimizer, CurveballOptimizer
from ptychoSampling.reconstruction.recons import ReconstructionT, SecondOrderReconstructionT
from ptychoSampling.logger import logger
from ptychoSampling.reconstruction.utils.utils import getComputationalCostInFlops
from ptychoSampling.reconstruction.aux_var_fwd_model_t import FarfieldPALMForwardModel
from ptychoSampling.reconstruction.lossfn_t import LossFunctionT
from ptychoSampling.reconstruction.optimization_t import Optimizer
from ptychoSampling.reconstruction.wavefront_t import fftshift_t

class AuxAssignOptimizer(Optimizer):
    def __init__(self, aux_v: tf.Variable,
                 aux_new_t: tf.Tensor,
                 initial_update_delay: int = 0,
                 update_frequency: int = 1.):
        super().__init__(initial_update_delay, update_frequency)
        self._aux_v = aux_v
        self._aux_new_t = aux_new_t
        self._optimizer = None


    def setupMinimizeOp(self):
        self._minimize_op = self._aux_v.assign(self._aux_new_t)

    @property
    def minimize_op(self):
        return self._minimize_op

class AuxLossFunctionT(LossFunctionT):
    """
    Parameters
    ----------
    scope_name : str
            Optional parameter that supplies a tensorflow "scope" name to the loss function calculation.
    """

    def map_preds_fn(self, x):
        amplitudes_wo_bg = tf.abs(x)
        intensities = amplitudes_wo_bg**2 + self._background_level
        amplitudes_bg = intensities**0.5
        phases = tf.angle(x)
        return tf.math.complex(amplitudes_bg * tf.cos(phases), amplitudes_bg * tf.sin(phases))


    def map_data_fn(self, x):
        return x

    @property
    def data_type(self):
        return "amplitude"

    def loss_fn(self, predicted_data_t, measured_data_t):
        return 0.5 * tf.reduce_sum(tf.abs(predicted_data_t - measured_data_t)**2)

class PALMReconstructionT(ReconstructionT):

    def __init__(self, *args: int,
                 obj_array_true: np.ndarray = None,
                 probe_wavefront_true: np.ndarray = None,
                 obj_abs_proj: bool = True,
                 update_delay_probe: int = 0,
                 update_delay_obj: int = 0,
                 update_delay_aux: int = 0,
                 update_frequency_probe: int = 1,
                 update_frequency_obj: int = 1,
                 update_frequency_aux: int = 1,
                 reconstruct_probe: bool = True,
                 registration_log_frequency: int = 10,
                 r_factor_log: bool = True,
                 loss_type: str = "gaussian",
                 learning_rate_scaling_probe: float = 1.0,
                 learning_rate_scaling_obj: float = 1.0,
                 loss_init_extra_kwargs: dict = None,
                 **kwargs: int):
        logger.info('initializing...')
        super().__init__(*args, **kwargs)
        if self.training_batch_size != self.n_train:
            e = ValueError("PALM reconstruction does not support minibatch reconstruction.")
            logger.error(e)
            raise e

        if loss_type != "gaussian":
            e = ValueError("PALM reconstruction does not support other loss functions.")
            logger.error(e)
            raise e

        logger.info('attaching fwd model...')
        self._attachCustomForwardModel(FarfieldPALMForwardModel, obj_abs_proj=obj_abs_proj)

        logger.info('creating loss fn...')
        self._attachCustomLossFunction(AuxLossFunctionT, loss_init_extra_kwargs)

        logger.info("create learning rates")
        self._learning_rate_scaling_probe = learning_rate_scaling_probe
        self._learning_rate_scaling_obj = learning_rate_scaling_obj
        self._lr_obj = self._getObjLearningRate() * learning_rate_scaling_obj
        self._lr_probe = self._getProbeLearningRate() * learning_rate_scaling_probe

        logger.info('creating optimizers...')
        aux_new_t = self._getAuxUpdate()
        self._attachCustomOptimizerForVariable(AuxAssignOptimizer,
                                               optimizer_init_args={"aux_v": self.fwd_model.aux_v,
                                                                    "aux_new_t": aux_new_t},
                                               initial_update_delay=update_delay_aux,
                                               update_frequency=update_frequency_aux)

        update_delay_obj = update_delay_obj if update_delay_obj is not None else 0
        self.attachTensorflowOptimizerForVariable("obj",
                                                  optimizer_type="gradient",
                                                  optimizer_init_args={"learning_rate": self._lr_obj},
                                                  initial_update_delay=update_delay_obj,
                                                  update_frequency=update_frequency_obj)

        update_delay_probe = update_delay_probe if update_delay_probe is not None else 0
        if reconstruct_probe:
            self.attachTensorflowOptimizerForVariable("probe",
                                                      optimizer_type="gradient",
                                                      optimizer_init_args={"learning_rate": self._lr_probe},
                                                      initial_update_delay=update_delay_probe,
                                                      update_frequency=update_frequency_probe)

        if obj_array_true is not None:
            self.addCustomMetricToDataLog(title="obj_error",
                                          tensor=self.fwd_model.obj_cmplx_t,
                                          log_epoch_frequency=registration_log_frequency,
                                          registration_ground_truth=obj_array_true)
        if reconstruct_probe and (probe_wavefront_true is not None):
            self.addCustomMetricToDataLog(title="probe_error",
                                          tensor=self.fwd_model.probe_cmplx_t,
                                          log_epoch_frequency=registration_log_frequency,
                                          registration_ground_truth=probe_wavefront_true)
        self._addRFactorLog(r_factor_log, 1)

    def run(self, *args, sequential_updates: bool=True, **kwargs):
        super().run(*args, sequential_updates=sequential_updates, **kwargs)

    def _attachModelPredictions(self, map_preds_fn: Callable = None, map_data_fn: Callable=None):

        if map_preds_fn is None:
            map_preds_fn = lambda x: x
        if map_data_fn is None:
            map_data_fn = lambda x: x
        with self.graph.as_default():
            self._batch_train_predictions_t = self.fwd_model.predict(self._batch_train_input_v,
                                                                     scope_name="train")
            self._batch_validation_predictions_t = self.fwd_model.predict(self._batch_validation_input_v,
                                                                                       scope_name="validation")

            self._batch_train_aux_t = tf.gather(self.fwd_model.aux_cmplx_t, self._batch_train_input_v)
            self._batch_validation_aux_t = tf.gather(self.fwd_model.aux_cmplx_t, self._batch_validation_input_v)

            self._batch_train_data_t = map_data_fn(tf.reshape(tf.gather(self._amplitudes_t,
                                                                         self._batch_train_input_v), [-1]))
            self._batch_validation_data_t = map_data_fn(tf.reshape(tf.gather(self._amplitudes_t,
                                                                              self._batch_validation_input_v), [-1]))

    def _attachCustomLossFunction(self, loss_method: AuxLossFunctionT,
                                  loss_init_extra_kwargs: dict = None):
        if loss_init_extra_kwargs is None:
            loss_init_extra_kwargs = {}
        if 'background_level' in loss_init_extra_kwargs and self.background_level is not None:
            raise ValueError("Cannot supply background level in loss argument if " +
                             "'background_level' in main class is not None.")
        loss_init_args = {'background_level': self.background_level}
        print('bg', self.background_level)
        loss_init_args.update(loss_init_extra_kwargs)
        # if "epsilon" not in loss_init_args:
        #    loss_init_args["epsilon"] = np.max(self.amplitudes**2) * self._eps
        self._loss_method = loss_method(dtype=self.dtype, **loss_init_args)
        self._attachModelPredictions(self._loss_method.map_preds_fn, self._loss_method.map_data_fn)

        with self.graph.as_default():
            self._train_loss_t = self._loss_method.loss_fn(self._batch_train_predictions_t, self._batch_train_aux_t)
            self._validation_loss_t = self._loss_method.loss_fn(self._batch_validation_predictions_t,
                                                                self._batch_validation_aux_t)

    def _getObjLearningRate(self) -> tf.Tensor:
        with self.graph.as_default():
            probe_sq = tf.abs(fftshift_t(self.fwd_model.probe_cmplx_t)) ** 2
            batch_obj_view_indices = tf.gather(self.fwd_model._obj_view_indices_t, self._batch_train_input_v)
            batch_obj_view_indices = tf.unstack(batch_obj_view_indices)
            size = self.obj.bordered_array.size

            tf_mat = tf.zeros(size, dtype=tf.float32)
            for b in batch_obj_view_indices:
                mat_this = tf.scatter_nd(indices=tf.reshape(b, [-1, 1]),
                                        shape=[size],
                                        updates=tf.reshape(probe_sq, [-1]))
                tf_mat = tf_mat + mat_this
            return 1.0 / tf.reduce_max(tf_mat)

    def _getProbeLearningRate(self) -> tf.Tensor:
        with self.graph.as_default():
            batch_obj_views = tf.gather(self.fwd_model._obj_views_all_t, self._batch_train_input_v)
            return 1.0 / tf.reduce_max(tf.reduce_sum(tf.abs(fftshift_t(batch_obj_views)) ** 2, axis=0))



    def _getAuxUpdate(self):
        if not hasattr(self, "_predicted_ff_all_t"):
            position_indices_t = tf.range(tf.shape(self._amplitudes_t)[0])
            self._predicted_ff_all_t = self.fwd_model.predict(position_indices_t)
        aux_update_t = self.fwd_model.exact_solve_aux_var(self._predicted_ff_all_t,
                                                          self._amplitudes_t)
        aux_update_reshaped_t = tf.reshape(tf.stack([tf.real(aux_update_t),
                                                     tf.imag(aux_update_t)]),
                                           [-1])
        return aux_update_reshaped_t

    def _addRFactorLog(self, r_factor_log: bool, log_frequency: int):
        if not r_factor_log:
            return
        with self.graph.as_default():
            if not hasattr(self, "_predicted_ff_all_t"):
                position_indices_t = tf.range(tf.shape(self._amplitudes_t)[0])
                self._predicted_ff_all_t = self.fwd_model.predict(position_indices_t)
            predicted_amplitudes_all = tf.abs(self._predicted_ff_all_t)
            self._r_factor_t = (tf.reduce_sum(tf.abs(predicted_amplitudes_all - self._amplitudes_t))
                                / tf.reduce_sum(self._amplitudes_t))
            self.addCustomMetricToDataLog(title="r_factor",
                                          tensor=self._r_factor_t,
                                          log_epoch_frequency=log_frequency)