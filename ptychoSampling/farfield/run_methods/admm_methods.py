import numpy as np
import tensorflow as tf
from typing import Tuple, Union, Callable
from ptychoSampling.reconstruction.recons import ReconstructionT
from ptychoSampling.logger import logger
from ptychoSampling.reconstruction.utils.utils import getComputationalCostInFlops
from ptychoSampling.reconstruction.forwardmodel_t import ForwardModelT
from ptychoSampling.reconstruction.optimization_t import ProjectedGradientOptimizer
from ptychoSampling.reconstruction.wavefront_t import fftshift_t, propFF_t

class ADMMFarfieldForwardModelT(ForwardModelT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, position_indices_t: tf.Tensor, scope_name: str="") -> tf.Tensor:

        scope_name = scope_name + "_predict" if scope_name else "predict"
        ndiffs = position_indices_t.get_shape().as_list()[0]

        with tf.name_scope(scope_name) as scope:
            if ndiffs == 0:
                return (tf.zeros(shape=[], dtype='complex64'),
                        tf.zeros(shape=[], dtype='complex64'),
                        tf.zeros(shape=[], dtype='float32', name=scope))
            batch_obj_views_t = tf.gather(self._obj_views_all_t, position_indices_t, name="test_gather")
            batch_obj_views_t = fftshift_t(batch_obj_views_t)

            exit_waves_t = batch_obj_views_t * self.probe_cmplx_t
            out_wavefronts_t = propFF_t(exit_waves_t)

            amplitudes_t = tf.abs(out_wavefronts_t)
            if self.upsampling_factor > 1:
                amplitudes_t = self._downsample(amplitudes_t)

        return batch_obj_views_t, tf.reshape(out_wavefronts_t, [-1]), tf.reshape(amplitudes_t, [-1])

    def predict_second_order(self, obj_v: tf.Variable,
                             probe_v: tf.Variable,
                             position_indices_t: tf.Tensor,
                             scope_name: str = "") -> tf.Tensor:
        pass


class SimpleADMMReconstructionT(ReconstructionT):
    def __init__(self, *args: int,
                 beta: float,
                 loss_type: str = "least_squared",
                 obj_array_true: np.ndarray = None,
                 probe_wavefront_true: np.ndarray = None,
                 r_factor_log: bool = True,
                 reconstruct_obj: bool = True,
                 reconstruct_probe: bool = True,
                 registration_log_frequency: int = 1,
                 obj_abs_proj: bool = True,
                 loss_init_extra_kwargs: dict = None,
                 update_delay_obj: int = 0,
                 update_delay_probe: int = 0,
                 **kwargs: int):
        logger.info('initializing...')
        super().__init__(*args, **kwargs)
        self._reconstruct_obj = reconstruct_obj
        self._reconstruct_probe = reconstruct_probe

        logger.info('attaching fwd model...')

        self._attachCustomForwardModel(ADMMFarfieldForwardModelT, obj_abs_proj=obj_abs_proj)
        self.fwd_model.obj_cmplx_t = tf.stop_gradient(self.fwd_model.obj_cmplx_t)
        self.fwd_model.probe_cmplx_t = tf.stop_gradient(self.fwd_model.probe_cmplx_t)


        logger.info('creating loss fn...')
        self.attachLossFunction(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)

        logger.info('attaching aux and multiplier vars...')
        self.attachAuxAndMultiplier(beta)

        logger.info('setting up obj and probe updates...')
        self._update_delay_obj = update_delay_obj if update_delay_obj is not None else 0
        self._update_delay_probe = update_delay_probe if update_delay_probe is not None else 0
        self._setupObjProbeUpdate(reconstruct_obj, reconstruct_probe)

        logger.info('setting up the lagrangian...')
        self._lagrangian_t = self._calcLagrangian(self._z_abs_v)

        logger.info('setting up the multiplier (and any other required parameter)...')
        self._setupMultiplierUpdate()

        logger.info('Setting up the phase update for the auxiliary variable...')
        self._setupAuxPhaseUpdate()
        logger.info('creating optimizers...')
        self._attachCustomOptimizerForVariable(optimize_method=ProjectedGradientOptimizer,
                                               optimizer_init_args={"input_var": self._z_abs_v,
                                                                    "loss_fn": self._calcLagrangian,
                                                                    "name": "aux_opt"})

        self.addCustomMetricToDataLog(title="ls_iters",
                                      tensor=self.optimizers[0]._optimizer._linesearch_iters,
                                      log_epoch_frequency=1)
        self.addCustomMetricToDataLog(title="alpha",
                                      tensor=self.optimizers[0]._optimizer._linesearch._alpha,
                                      log_epoch_frequency=1)

        self._default_log_items['admm_inner_iters'] = None
        self.datalog.addSimpleMetric('admm_inner_iters')

        if reconstruct_obj and (obj_array_true is not None):
            self.addCustomMetricToDataLog(title="obj_error",
                                          tensor=self.fwd_model.obj_cmplx_t,
                                          log_epoch_frequency=registration_log_frequency,
                                          registration_ground_truth=obj_array_true)
        if reconstruct_probe and (probe_wavefront_true is not None):
            self.addCustomMetricToDataLog(title="probe_error",
                                          tensor=self.fwd_model.probe_cmplx_t,
                                          log_epoch_frequency=registration_log_frequency,
                                          registration_ground_truth=probe_wavefront_true)

        self.addCustomMetricToDataLog(title='lagrangian',
                                      tensor=self._lagrangian_t,
                                      log_epoch_frequency=registration_log_frequency)

        self._addRFactorLog(r_factor_log, registration_log_frequency)

    def _attachModelPredictions(self, map_preds_fn: Callable = None, map_data_fn: Callable=None):
        if map_preds_fn is None:
            map_preds_fn = lambda x: x
        if map_data_fn is None:
            map_data_fn = lambda x : x
        with self.graph.as_default():

            [
                self._batch_train_obj_views_shifted_t,
                self._batch_train_predicted_ff_t,
                batch_train_predicted_amplitudes_t
            ] = self.fwd_model.predict(self._batch_train_input_v,
                                       scope_name="train")
            self._batch_train_predictions_t = map_preds_fn(batch_train_predicted_amplitudes_t)

            #self._batch_train_predictions_t = map_preds_fn(self.fwd_model.predict(self._batch_train_input_v,
            #                                                                      scope_name="train"))
            [
                self._batch_validation_obj_views_shifted_t,
                self._batch_validation_predicted_ff_t,
                batch_validation_predicted_amplitudes_t
            ] = self.fwd_model.predict(self._batch_validation_input_v,
                                       scope_name="train")
            self._batch_validation_predictions_t = map_preds_fn(batch_validation_predicted_amplitudes_t)

            self._batch_train_data_t = map_data_fn(tf.reshape(tf.gather(self._amplitudes_t,
                                                                         self._batch_train_input_v), [-1]))

            self._batch_validation_data_t = map_data_fn(tf.reshape(tf.gather(self._amplitudes_t,
                                                                              self._batch_validation_input_v), [-1]))

    def attachAuxAndMultiplier(self, betaval: float):
        with self.graph.as_default():
            with tf.device("/gpu:0"):
                self._beta_v = tf.Variable(betaval, dtype='float32', name='beta')
                self._beta_placeholder = tf.placeholder('float32', shape=[])
                self._beta_assign_op = self._beta_v.assign(self._beta_placeholder)

                abs_constraint_fn = lambda x: tf.clip_by_value(x, 0, np.inf)
                self._z_abs_v = tf.Variable(np.zeros(self.amplitudes.size),
                                            dtype='float32',
                                            constraint=abs_constraint_fn)
                self._z_phase_v = tf.Variable(np.zeros(self.amplitudes.size),
                                              dtype='float32')
                self._z_t = tf.complex(self._z_abs_v * tf.math.cos(self._z_phase_v),
                                       self._z_abs_v * tf.math.sin(self._z_phase_v))

                self._z_init_op = tf.group([self._z_abs_v.assign(tf.abs(self._batch_train_predicted_ff_t)),
                                            self._z_phase_v.assign(tf.angle(self._batch_train_predicted_ff_t))])
                [
                    self._multipliers_v,
                    self._multipliers_t
                ] = self._createComplexVariable(np.zeros(self.amplitudes.size), name='multipliers')


    def _createComplexVariable(self, init: Union[complex, np.ndarray],
                            name: str,
                            constraint_fn: Callable = None,
                            dtype: str = "float32" ) -> Tuple[tf.Variable, tf.Tensor]:
        with self.graph.as_default():
            init_reals = np.concatenate((np.real(init).flat, np.imag(init).flat))
            var = tf.Variable(init_reals, constraint=constraint_fn, dtype=dtype, name=name)
            output = self.fwd_model._reshapeComplexVariable(var, init.shape)
        return var, output

    def _setupObjProbeUpdate(self, reconstruct_obj, reconstruct_probe):

        with self.graph.as_default():
            z_hat = self._z_t + self._multipliers_t / tf.cast(self._beta_v, 'complex64')
            z_hat_ift = propFF_t(tf.reshape(z_hat, self.amplitudes.shape), backward=True)

            if reconstruct_probe:
                probe_numerator = (tf.reduce_sum(z_hat_ift
                                                 * tf.conj(self._batch_train_obj_views_shifted_t), axis=0))
                probe_denominator = tf.reduce_sum(tf.abs(self._batch_train_obj_views_shifted_t) ** 2,
                                                  axis=0)

                probe_update_cmplx = (probe_numerator /
                                      tf.cast(probe_denominator + 1e-8 * tf.reduce_max(probe_denominator), 'complex64'))

                self._probe_update =  tf.concat((tf.reshape(tf.real(probe_update_cmplx), [-1]),
                                                 tf.reshape(tf.imag(probe_update_cmplx), [-1])),
                                                axis=0)
                self._probe_update_op = self.fwd_model.probe_v.assign(self._probe_update)

            if reconstruct_obj:
                probe_sq = tf.reshape(tf.abs(fftshift_t(self.fwd_model.probe_cmplx_t)) ** 2, [-1])
                batch_obj_view_indices = tf.gather(self.fwd_model._obj_view_indices_t, self._batch_train_input_v)
                batch_obj_view_indices = tf.unstack(batch_obj_view_indices)
                size = self.obj.bordered_array.size

                obj_numerator = tf.zeros(size, dtype=tf.complex64)
                obj_denominator = tf.zeros(size, dtype=tf.float32)
                for i, b in enumerate(batch_obj_view_indices):
                    indices = tf.reshape(b, [-1, 1])
                    num_term = tf.reshape(fftshift_t(z_hat_ift[i] * tf.conj(self.fwd_model.probe_cmplx_t)),
                                          [-1])

                    num_real = tf.scatter_nd(indices=indices, shape=[size], updates=tf.real(num_term))
                    num_imag = tf.scatter_nd(indices=indices, shape=[size], updates=tf.imag(num_term))

                    obj_numerator += tf.complex(num_real, num_imag)
                    obj_denominator = tf.tensor_scatter_nd_add(tensor=obj_denominator,
                                                               indices=indices,
                                                               updates=probe_sq)
                obj_denominator = obj_denominator + 1e-8 * tf.reduce_max(obj_denominator)
                obj_update_bordered = obj_numerator / tf.cast(obj_denominator, 'complex64')
                obj_update_bordered_reshaped = tf.reshape(obj_update_bordered, self.obj.bordered_array.shape)

                (b1, b2), (b3, b4) = self.obj.border_shape
                obj_update_cropped = obj_update_bordered_reshaped[b1:-b2, b3:-b4]
                self._obj_update = tf.concat((tf.reshape(tf.real(obj_update_cropped), [-1]),
                                              tf.reshape(tf.imag(obj_update_cropped), [-1])),
                                             axis=0)
                if self.fwd_model.obj_v.constraint is not None:
                    self._obj_update = self.fwd_model.obj_v.constraint(self._obj_update)

                self._obj_update_op = self.fwd_model.obj_v.assign(self._obj_update)

    def _calcPrimalResidualNorm(self, aux_t, predicted_ff_t):
        with self.graph.as_default():
            return tf.reduce_sum(tf.abs(aux_t - predicted_ff_t)**2)

    def _calcDualResidualNorm(self):
        with self.graph.as_default():
            return self._beta_v * tf.reduce_sum((self._complexToRealSingleDim(self._z_t) - self._z_old)**2)

    def _calcLagrangian(self, aux_abs_v):
        with self.graph.as_default():

            aux_cmplx_t = tf.complex(aux_abs_v * tf.math.cos(self._z_phase_v),
                                     aux_abs_v * tf.math.sin(self._z_phase_v))
            aux_preds_t = self._loss_method.map_preds_fn(tf.abs(aux_cmplx_t))
            loss = self._loss_method.loss_fn(aux_preds_t,  self._batch_train_data_t)
            t1 = aux_cmplx_t - self._batch_train_predicted_ff_t
            t2 = 2 * tf.real(tf.reduce_sum(tf.conj(t1) * self._multipliers_t))
            p = self._calcPrimalResidualNorm(aux_cmplx_t, self._batch_train_predicted_ff_t)

            #with tf.control_dependencies([tf.print('loss', loss, 't2', t2, 'p', p)]):
            t3 = self._beta_v * p
            return loss + t2 + t3

    def _setupAuxPhaseUpdate(self):
        with self.graph.as_default():
            self._z_plus_t = (self._batch_train_predicted_ff_t
                              - self._multipliers_t / tf.cast(self._beta_v, 'complex64'))
            self._z_phase_update_op = self._z_phase_v.assign(tf.angle(self._z_plus_t))

    def _setupMultiplierUpdate(self):
        with self.graph.as_default():
            m = (self._multipliers_t +
                 tf.cast(self._beta_v, 'complex64') * (self._z_t - self._batch_train_predicted_ff_t))

            self._multiplier_update_op = self._multipliers_v.assign(self._complexToRealSingleDim(m))

    @staticmethod
    def _complexToRealSingleDim(complex_t: tf.Tensor)-> tf.Tensor:
        complex_t = tf.reshape(complex_t, [-1])
        r = tf.real(complex_t)
        im = tf.imag(complex_t)
        return tf.concat([r, im], axis=0)

    def run(self, max_iterations: int = 50,
            max_inner_iterations: int = 1,
            debug_output: bool = True,
            debug_output_epoch_frequency: int = 10,
            **unused_kwargs):

        if not hasattr(self, "session"):
            self.finalizeSetup()
        print_debug_header = True
        epochs_start = self.epoch
        epochs_this_run = 0

        def finalizeIter(iter_log_dict, debug_output_this_iter, print_debug_header):
            self.datalog.logStep(self.iteration, iter_log_dict)
            out = print_debug_header
            if debug_output_this_iter:
                out = self._printDebugOutput(debug_output_epoch_frequency,
                                             epochs_this_run,
                                             print_debug_header)
            return out

        if self.iteration == 0:
            self.session.run(self._new_train_batch_op)
            self.session.run(self._z_init_op)

        for i in range(max_iterations):
            iter_log_dict = {}
            self.iteration += 1
            self.session.run([self._new_iteration_op])

            if self._reconstruct_obj and self.iteration > self._update_delay_obj:
                self.session.run(self._obj_update_op)

            if self._reconstruct_probe and self.iteration > self._update_delay_probe:
                self.session.run(self._probe_update_op)

            self.session.run(self._z_phase_update_op)
            for inner_iter in range(max_inner_iterations):
                self.session.run(self.optimizers[0].minimize_op)

            _, lossval = self.session.run([self._multiplier_update_op, self._train_loss_t])

            self._default_log_items["epoch"] = self.epoch
            self._default_log_items["train_loss"] = lossval
            self._default_log_items['admm_inner_iters'] = inner_iter + 1
            iter_log_dict.update(self._default_log_items)


            if i % self._iterations_per_epoch != 0:
                print_debug_header = finalizeIter(iter_log_dict, False, print_debug_header)
                # self.datalog.logStep(self.iteration, iter_log_dict)
                continue

            epochs_this_run = self.epoch - epochs_start

            custom_metrics = self.datalog.getCustomTensorMetrics(epochs_this_run)
            custom_metrics_tensors = list(custom_metrics.values())
            if len(custom_metrics_tensors) > 0:
                custom_metrics_values = self.session.run(custom_metrics_tensors)
                log_dict = dict(zip(custom_metrics.keys(), custom_metrics_values))
                iter_log_dict.update(log_dict)

            print_debug_header = finalizeIter(iter_log_dict, True, print_debug_header)
        self._updateOutputs()

    def _updateOutputs(self):
        if "obj" in self.fwd_model.model_vars:
            self.obj.array = self.session.run(self.fwd_model.model_vars["obj"]["output"])
        if "probe" in self.fwd_model.model_vars:
            self.probe.wavefront[:] = self.session.run(self.fwd_model.model_vars["probe"]["output"])

    def _addRFactorLog(self, r_factor_log: bool, log_frequency: int):
        if not r_factor_log:
            return
        with self.graph.as_default():
            if not hasattr(self, "_predictions_all_t"):
                position_indices_t = tf.range(tf.shape(self._amplitudes_t)[0])
                _, __, self._predictions_all_t = self.fwd_model.predict(position_indices_t)
            self._r_factor_t = (tf.reduce_sum(tf.abs(self._predictions_all_t -
                                                     tf.reshape(self._amplitudes_t, [-1])))
                                / tf.reduce_sum(self._amplitudes_t))
            self.addCustomMetricToDataLog(title="r_factor",
                                          tensor=self._r_factor_t,
                                          log_epoch_frequency=log_frequency)

    def getFlopsPerIter(self) -> dict:

        if self.n_validation > 0:
            e = RuntimeWarning("n_validation > 0 can give misleading flops.")
            logger.warning(e)

        if not hasattr(self, "session"):
            self.run(1)

        total_flops = getComputationalCostInFlops(self.graph)

        aux_ls_flops = getComputationalCostInFlops(self.graph,
                                                   keywords=[("aux_opt",#_minimize_step",
                                                              "backtracking_linesearch")
                                                             ],
                                                   exclude_keywords=False)

        aux_only_flops = getComputationalCostInFlops(self.graph,
                                                     keywords=["aux_opt"],
                                                     exclude_keywords=False)
        early_stop_flops = getComputationalCostInFlops(self.graph,
                                                       keywords=['early_stop_check'],
                                                       exclude_keywords=False)

        flops_without_ls = total_flops - aux_ls_flops
        d = {"total_flops": total_flops,
             "aux_ls_flops": aux_ls_flops,
             "aux_only_flops": aux_only_flops,
             "early_stop_only_flops": early_stop_flops,
             "flops_without_ls": flops_without_ls}
        return d




class PartiallyInexactADMMReconstructionT(ReconstructionT):
    "This is under construction and might not work correctly right now."
    def __init__(self, *args: int,
                 beta: float,
                 loss_type: str = "least_squared",
                 obj_array_true: np.ndarray = None,
                 probe_wavefront_true: np.ndarray = None,
                 r_factor_log: bool = True,
                 reconstruct_obj: bool = True,
                 reconstruct_probe: bool = True,
                 registration_log_frequency: int = 1,
                 obj_abs_proj: bool = True,
                 loss_init_extra_kwargs: dict = None,
                 update_delay_obj: int = 0,
                 update_delay_probe: int = 0,
                 apply_inner_early_stop: bool = False,
                 log_primal_dual: bool = False,
                 sigmaval: float = 0.99,
                 reset_opt_every_iter: bool = False,
                 **kwargs: int):
        logger.info('initializing...')
        super().__init__(*args, **kwargs)
        self._reconstruct_obj = reconstruct_obj
        self._reconstruct_probe = reconstruct_probe
        self._apply_inner_early_stop = apply_inner_early_stop
        self._log_primal_dual = log_primal_dual

        logger.info('attaching fwd model...')
        self._attachCustomForwardModel(ADMMFarfieldForwardModelT, obj_abs_proj=obj_abs_proj)
        self.fwd_model.obj_cmplx_t = tf.stop_gradient(self.fwd_model.obj_cmplx_t)
        self.fwd_model.probe_cmplx_t = tf.stop_gradient(self.fwd_model.probe_cmplx_t)

        logger.info('creating loss fn...')
        self.attachLossFunction(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)

        logger.info('attaching aux and multiplier vars...')
        self.attachAuxAndMultiplier(beta)

        logger.info('setting up obj and probe updates...')
        self._update_delay_obj = update_delay_obj if update_delay_obj is not None else 0
        self._update_delay_probe = update_delay_probe if update_delay_probe is not None else 0
        self._setupObjProbeUpdate(reconstruct_obj, reconstruct_probe)

        logger.info('setting up the lagrangian...')
        self._lagrangian_t = self._calcLagrangian(self._z_abs_v)

        logger.info('setting up the multiplier (and any other required parameter)...')
        self._setupMultiplierUpdate()

        logger.info('Setting up the phase update for the auxiliary variable...')
        self._setupAuxPhaseUpdate()
        logger.info('creating optimizers...')
        self._attachCustomOptimizerForVariable(optimize_method=ProjectedGradientOptimizer,
                                               optimizer_init_args={"input_var": self._z_abs_v,
                                                                    "loss_fn": self._calcLagrangian,
                                                                    "name": "aux_opt"})
        self._reset_opt_every_iter = reset_opt_every_iter
        if self._apply_inner_early_stop:
            self._primal_norm_t = self._calcPrimalResidualNorm(self._z_t, self._batch_train_predicted_ff_t)
            self._setupEarlyStoppingCheck(sigmaval)

        self.addCustomMetricToDataLog(title="ls_iters",
                                      tensor=self.optimizers[0]._optimizer._linesearch_iters,
                                      log_epoch_frequency=1)
        self.addCustomMetricToDataLog(title="alpha",
                                      tensor=self.optimizers[0]._optimizer._linesearch._alpha,
                                      log_epoch_frequency=1)

        self._default_log_items['admm_inner_iters'] = None
        self.datalog.addSimpleMetric('admm_inner_iters')

        if reconstruct_obj and (obj_array_true is not None):
            self.addCustomMetricToDataLog(title="obj_error",
                                          tensor=self.fwd_model.obj_cmplx_t,
                                          log_epoch_frequency=registration_log_frequency,
                                          registration_ground_truth=obj_array_true)
        if reconstruct_probe and (probe_wavefront_true is not None):
            self.addCustomMetricToDataLog(title="probe_error",
                                          tensor=self.fwd_model.probe_cmplx_t,
                                          log_epoch_frequency=registration_log_frequency,
                                          registration_ground_truth=probe_wavefront_true)

        self.addCustomMetricToDataLog(title='lagrangian',
                                      tensor=self._lagrangian_t,
                                      log_epoch_frequency=registration_log_frequency)

        self._setupPrimalDualLog(log_primal_dual, registration_log_frequency)
        self._addRFactorLog(r_factor_log, registration_log_frequency)

    def _attachModelPredictions(self, map_preds_fn: Callable = None, map_data_fn: Callable=None):

        if map_preds_fn is None:
            map_preds_fn = lambda x: x
        with self.graph.as_default():

            [
                self._batch_train_obj_views_shifted_t,
                self._batch_train_predicted_ff_t,
                batch_train_predicted_amplitudes_t
            ] = self.fwd_model.predict(self._batch_train_input_v,
                                       scope_name="train")
            self._batch_train_predictions_t = map_preds_fn(batch_train_predicted_amplitudes_t)

            #self._batch_train_predictions_t = map_preds_fn(self.fwd_model.predict(self._batch_train_input_v,
            #                                                                      scope_name="train"))

            [
                self._batch_validation_obj_views_shifted_t,
                self._batch_validation_predicted_ff_t,
                batch_validation_predicted_amplitudes_t
            ] = self.fwd_model.predict(self._batch_validation_input_v,
                                       scope_name="train")
            self._batch_validation_predictions_t = map_preds_fn(batch_validation_predicted_amplitudes_t)

            self._batch_train_data_t = map_data_fn(tf.reshape(tf.gather(self._amplitudes_t,
                                                                         self._batch_train_input_v), [-1]))

            self._batch_validation_data_t = map_data_fn(tf.reshape(tf.gather(self._amplitudes_t,
                                                                              self._batch_validation_input_v), [-1]))

    def attachAuxAndMultiplier(self, betaval: float):
        with self.graph.as_default():
            with tf.device("/gpu:0"):
                self._beta_v = tf.Variable(betaval, dtype='float32', name='beta')
                self._beta_placeholder = tf.placeholder('float32', shape=[])
                self._beta_assign_op = self._beta_v.assign(self._beta_placeholder)

                abs_constraint_fn = lambda x: tf.clip_by_value(x, 0, np.inf)
                self._z_abs_v = tf.Variable(np.zeros(self.amplitudes.size),
                                            dtype='float32',
                                            constraint=abs_constraint_fn)
                self._z_phase_v = tf.Variable(np.zeros(self.amplitudes.size),
                                              dtype='float32')
                self._z_t = tf.complex(self._z_abs_v * tf.math.cos(self._z_phase_v),
                                       self._z_abs_v * tf.math.sin(self._z_phase_v))

                self._z_init_op = tf.group([self._z_abs_v.assign(tf.abs(self._batch_train_predicted_ff_t)),
                                            self._z_phase_v.assign(tf.angle(self._batch_train_predicted_ff_t))])
                [
                    self._multipliers_v,
                    self._multipliers_t
                ] = self._createComplexVariable(np.zeros(self.amplitudes.size), name='multipliers')

                if self._log_primal_dual:
                    self._z_old = tf.Variable(tf.zeros_like(self._multipliers_v))

                # This is for the partially inexact aux update
                if self._apply_inner_early_stop:
                    self._w = tf.Variable(tf.zeros_like(self._multipliers_v))
                    self._y_old = tf.Variable(tf.zeros_like(self._multipliers_v))


    def _createComplexVariable(self, init: Union[complex, np.ndarray],
                            name: str,
                            constraint_fn: Callable = None,
                            dtype: str = "float32" ) -> Tuple[tf.Variable, tf.Tensor]:
        with self.graph.as_default():
            init_reals = np.concatenate((np.real(init).flat, np.imag(init).flat))
            var = tf.Variable(init_reals, constraint=constraint_fn, dtype=dtype, name=name)
            output = self.fwd_model._reshapeComplexVariable(var, init.shape)
        return var, output

    def _setupObjProbeUpdate(self, reconstruct_obj, reconstruct_probe):

        with self.graph.as_default():
            z_hat = self._z_t + self._multipliers_t / tf.cast(self._beta_v, 'complex64')
            z_hat_ift = propFF_t(tf.reshape(z_hat, self.amplitudes.shape), backward=True)

            if reconstruct_probe:
                probe_numerator = (tf.reduce_sum(z_hat_ift
                                                 * tf.conj(self._batch_train_obj_views_shifted_t), axis=0))
                probe_denominator = tf.reduce_sum(tf.abs(self._batch_train_obj_views_shifted_t) ** 2,
                                                  axis=0)

                probe_update_cmplx = (probe_numerator /
                                      tf.cast(probe_denominator + 1e-8 * tf.reduce_max(probe_denominator), 'complex64'))

                self._probe_update =  tf.concat((tf.reshape(tf.real(probe_update_cmplx), [-1]),
                                                 tf.reshape(tf.imag(probe_update_cmplx), [-1])),
                                                axis=0)
                self._probe_update_op = self.fwd_model.probe_v.assign(self._probe_update)

            if reconstruct_obj:
                probe_sq = tf.reshape(tf.abs(fftshift_t(self.fwd_model.probe_cmplx_t)) ** 2, [-1])
                batch_obj_view_indices = tf.gather(self.fwd_model._obj_view_indices_t, self._batch_train_input_v)
                batch_obj_view_indices = tf.unstack(batch_obj_view_indices)
                size = self.obj.bordered_array.size

                obj_numerator = tf.zeros(size, dtype=tf.complex64)
                obj_denominator = tf.zeros(size, dtype=tf.float32)
                for i, b in enumerate(batch_obj_view_indices):
                    indices = tf.reshape(b, [-1, 1])
                    num_term = tf.reshape(fftshift_t(z_hat_ift[i] * tf.conj(self.fwd_model.probe_cmplx_t)),
                                          [-1])

                    num_real = tf.scatter_nd(indices=indices, shape=[size], updates=tf.real(num_term))
                    num_imag = tf.scatter_nd(indices=indices, shape=[size], updates=tf.imag(num_term))

                    obj_numerator += tf.complex(num_real, num_imag)
                    obj_denominator = tf.tensor_scatter_nd_add(tensor=obj_denominator,
                                                               indices=indices,
                                                               updates=probe_sq)
                obj_denominator = obj_denominator + 1e-8 * tf.reduce_max(obj_denominator)
                obj_update_bordered = obj_numerator / tf.cast(obj_denominator, 'complex64')
                obj_update_bordered_reshaped = tf.reshape(obj_update_bordered, self.obj.bordered_array.shape)

                (b1, b2), (b3, b4) = self.obj.border_shape
                obj_update_cropped = obj_update_bordered_reshaped[b1:-b2, b3:-b4]
                self._obj_update = tf.concat((tf.reshape(tf.real(obj_update_cropped), [-1]),
                                              tf.reshape(tf.imag(obj_update_cropped), [-1])),
                                             axis=0)
                if self.fwd_model.obj_v.constraint is not None:
                    self._obj_update = self.fwd_model.obj_v.constraint(self._obj_update)

                self._obj_update_op = self.fwd_model.obj_v.assign(self._obj_update)

    def _calcPrimalResidualNorm(self, aux_t, predicted_ff_t):
        with self.graph.as_default():
            return tf.reduce_sum(tf.abs(aux_t - predicted_ff_t)**2)

    def _calcDualResidualNorm(self):
        with self.graph.as_default():
            return self._beta_v * tf.reduce_sum((self._complexToRealSingleDim(self._z_t) - self._z_old)**2)

    def _calcLagrangian(self, aux_abs_v):
        with self.graph.as_default():

            aux_cmplx_t = tf.complex(aux_abs_v * tf.math.cos(self._z_phase_v),
                                     aux_abs_v * tf.math.sin(self._z_phase_v))
            aux_preds_t = self._loss_method.map_preds_fn(tf.abs(aux_cmplx_t))
            loss = self._loss_method.loss_fn(aux_preds_t,  self._batch_train_data_t)
            t1 = aux_cmplx_t - self._batch_train_predicted_ff_t
            t2 = 2 * tf.real(tf.reduce_sum(tf.conj(t1) * self._multipliers_t))
            p = self._calcPrimalResidualNorm(aux_cmplx_t, self._batch_train_predicted_ff_t)

            #with tf.control_dependencies([tf.print('loss', loss, 't2', t2, 'p', p)]):
            t3 = self._beta_v * p
            return loss + t2 + t3

    def _setupAuxPhaseUpdate(self):
        with self.graph.as_default():
            self._z_plus_t = (self._batch_train_predicted_ff_t
                              - self._multipliers_t / tf.cast(self._beta_v, 'complex64'))
            self._z_phase_update_op = self._z_phase_v.assign(tf.angle(self._z_plus_t))

    def _setupMultiplierUpdate(self):
        with self.graph.as_default():
            m = (self._multipliers_t +
                 tf.cast(self._beta_v, 'complex64') * (self._z_t - self._batch_train_predicted_ff_t))

            self._multiplier_update_op = self._multipliers_v.assign(self._complexToRealSingleDim(m))

    def _setupEarlyStoppingCheck(self, sigmaval):
        with self.graph.as_default():
            with tf.name_scope('early_stop_check'):
                # We need the gradient not of z_abs, but of the full auxiliary variable z
                grad_z_abs_t = self.optimizers[0]._optimizer._grads_t
                grad_z_phase_t = (-self._beta_v * self._z_abs_v * tf.abs(self._z_plus_t)
                                  * tf.math.sin(tf.angle(self._z_plus_t) - self._z_phase_v))
                t1 = tf.complex(grad_z_abs_t, grad_z_phase_t * self._z_abs_v)
                t2 = tf.complex(tf.math.cos(self._z_phase_v), tf.math.sin(self._z_phase_v))
                grad_z_t = self._complexToRealSingleDim(t1 * t2)

                self._y_old_update_op = self._y_old.assign(grad_z_t)
                self._w_update_op = self._w.assign_sub(self._beta_v * self._y_old)

                self._grad_z_abs_t = grad_z_abs_t
                self._grad_z_phase_t = grad_z_phase_t
                self._grad_z_t = grad_z_t

                self._sigma_v = tf.Variable(sigmaval, dtype='float32')

                self._sigma_placeholder = tf.placeholder(shape=[], dtype='float32')
                self._sigma_assign_op = self._sigma_v.assign(self._sigma_placeholder)

                check_lhs = (2 / self._beta_v * tf.linalg.norm((self._w - self._complexToRealSingleDim(self._z_t))
                                                               * grad_z_t)
                             + tf.reduce_sum(grad_z_t**2))
                check_rhs = self._sigma_v * self._primal_norm_t
                self._check_lhs = check_lhs
                self._check_rhs = check_rhs
                #print_op = tf.print('lhs', check_lhs, 'rhs', check_rhs,
                #                    'loss', self.optimizers[0]._optimizer._loss_t,
                #                     'grad_norm', tf.linalg.norm(self.optimizers[0]._optimizer._grads_t, ord=np.inf))
                #with tf.control_dependencies([print_op]):
                self._early_stop_check_t = tf.math.less_equal(check_lhs, check_rhs)

    def _setupPrimalDualLog(self, log_primal_dual, registration_log_frequency):
        if not log_primal_dual:
            return
        with self.graph.as_default():
            if not hasattr(self, "_primal_norm_t"):
                self._primal_norm_t = self._calcPrimalResidualNorm(self._z_t, self._batch_train_predicted_ff_t)
            self._dual_norm_t = self._calcDualResidualNorm()
            self._z_old_update_op = self._z_old.assign(self._complexToRealSingleDim(self._z_t))
        self.addCustomMetricToDataLog(title='primal_norm',
                                      tensor=self._primal_norm_t,
                                      log_epoch_frequency=registration_log_frequency)
        self.addCustomMetricToDataLog(title='dual_norm',
                                      tensor=self._dual_norm_t,
                                      log_epoch_frequency=registration_log_frequency)

    @staticmethod
    def _complexToRealSingleDim(complex_t: tf.Tensor)-> tf.Tensor:
        complex_t = tf.reshape(complex_t, [-1])
        r = tf.real(complex_t)
        im = tf.imag(complex_t)
        return tf.concat([r, im], axis=0)

    def run(self, max_iterations: int = 50,
            max_inner_iterations: int = 1,
            debug_output: bool = True,
            debug_output_epoch_frequency: int = 10,
            **unused_kwargs):

        if not hasattr(self, "session"):
            self.finalizeSetup()
        print_debug_header = True
        epochs_start = self.epoch
        epochs_this_run = 0

        def finalizeIter(iter_log_dict, debug_output_this_iter, print_debug_header, inner_iters):
            self.datalog.logStep(self.iteration, iter_log_dict)
            out = print_debug_header
            if debug_output_this_iter:
                out = self._printDebugOutput(debug_output_epoch_frequency,
                                             epochs_this_run,
                                             print_debug_header)
            self.session.run(self.optimizers[0]._optimizer._linesearch_iters.assign(0))
            if self._apply_inner_early_stop and (inner_iters >= (max_inner_iterations - 1)) and (max_inner_iterations > 1):
                self.session.run(self.optimizers[0]._optimizer.reset)
            return out

        if self.iteration == 0:
            self.session.run(self._new_train_batch_op)
            self.session.run(self._z_init_op)

        for i in range(max_iterations):
            iter_log_dict = {}
            self.iteration += 1
            self.session.run([self._new_iteration_op])

            if self._reconstruct_obj and self.iteration > self._update_delay_obj:
                self.session.run(self._obj_update_op)

            if self._reconstruct_probe and self.iteration > self._update_delay_probe:
                self.session.run(self._probe_update_op)

            if self._log_primal_dual:
                self.session.run(self._z_old_update_op)
            if self._apply_inner_early_stop:
                self.session.run(self._y_old_update_op)
            self.session.run(self._z_phase_update_op)
            for inner_iter in range(max_inner_iterations):
                if self._apply_inner_early_stop:
                    _, early_stop = self.session.run([self.optimizers[0].minimize_op,
                                                      self._early_stop_check_t])
                    if early_stop:
                        break
                    continue
                self.session.run(self.optimizers[0].minimize_op)

            if self._apply_inner_early_stop:
                _, __, lossval  = self.session.run([self._multiplier_update_op,
                                                    self._w_update_op,
                                                    self._train_loss_t])
            else:
                _, lossval = self.session.run([self._multiplier_update_op, self._train_loss_t])
            if self._reset_opt_every_iter:
                self.session.run(self.optimizers[0]._optimizer.reset)
            self._default_log_items["epoch"] = self.epoch
            self._default_log_items["train_loss"] = lossval
            self._default_log_items['admm_inner_iters'] = inner_iter + 1
            iter_log_dict.update(self._default_log_items)


            if i % self._iterations_per_epoch != 0:
                print_debug_header = finalizeIter(iter_log_dict, False, print_debug_header, inner_iter)
                # self.datalog.logStep(self.iteration, iter_log_dict)
                continue

            epochs_this_run = self.epoch - epochs_start

            custom_metrics = self.datalog.getCustomTensorMetrics(epochs_this_run)
            custom_metrics_tensors = list(custom_metrics.values())
            if len(custom_metrics_tensors) > 0:
                custom_metrics_values = self.session.run(custom_metrics_tensors)
                log_dict = dict(zip(custom_metrics.keys(), custom_metrics_values))
                iter_log_dict.update(log_dict)

            print_debug_header = finalizeIter(iter_log_dict, True, print_debug_header, inner_iter)
        self._updateOutputs()

    def _updateOutputs(self):
        if "obj" in self.fwd_model.model_vars:
            self.obj.array = self.session.run(self.fwd_model.model_vars["obj"]["output"])
        if "probe" in self.fwd_model.model_vars:
            self.probe.wavefront[:] = self.session.run(self.fwd_model.model_vars["probe"]["output"])

    def _addRFactorLog(self, r_factor_log: bool, log_frequency: int):
        if not r_factor_log:
            return
        with self.graph.as_default():
            if not hasattr(self, "_predictions_all_t"):
                position_indices_t = tf.range(tf.shape(self._amplitudes_t)[0])
                _, __, self._predictions_all_t = self.fwd_model.predict(position_indices_t)
            self._r_factor_t = (tf.reduce_sum(tf.abs(self._predictions_all_t -
                                                     tf.reshape(self._amplitudes_t, [-1])))
                                / tf.reduce_sum(self._amplitudes_t))
            self.addCustomMetricToDataLog(title="r_factor",
                                          tensor=self._r_factor_t,
                                          log_epoch_frequency=log_frequency)

    def getFlopsPerIter(self) -> dict:

        if self.n_validation > 0:
            e = RuntimeWarning("n_validation > 0 can give misleading flops.")
            logger.warning(e)

        if not hasattr(self, "session"):
            self.run(1)

        total_flops = getComputationalCostInFlops(self.graph)

        aux_ls_flops = getComputationalCostInFlops(self.graph,
                                                   keywords=[("aux_opt_minimize_step",
                                                              "backtracking_linesearch")
                                                             ],
                                                   exclude_keywords=False)

        aux_only_flops = getComputationalCostInFlops(self.graph,
                                                     keywords=["aux_opt"],
                                                     exclude_keywords=False)
        early_stop_flops = getComputationalCostInFlops(self.graph,
                                                       keywords=['early_stop_check'],
                                                       exclude_keywords=False)

        flops_without_ls = total_flops - aux_ls_flops
        d = {"total_flops": total_flops,
             "aux_ls_flops": aux_ls_flops,
             "aux_only_flops": aux_only_flops,
             "early_stop_only_flops": early_stop_flops,
             "flops_without_ls": flops_without_ls}
        return d


