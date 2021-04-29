from typing import Callable
from ptychoSampling.reconstruction.optimization_t import LMAOptimizer, \
    CurveballOptimizer, ConjugateGradientOptimizer, ScaledLMAOptimizer, \
    PCGLMAOptimizer, ScaledPCGLMAOptimizer
import ptychoSampling
import tensorflow as tf
from ptychoSampling.logger import logger
import numpy as np
from ptychoSampling.reconstruction.utils.utils import getComputationalCostInFlops
from ptychoSampling.reconstruction.recons import ReconstructionT
from ptychoSampling.reconstruction.forwardmodel_t import JointFarfieldForwardModelT
from ptychoSampling.reconstruction.wavefront_t import fftshift_t

__all__ = ['JointLMAReconstructionT', 'JointCGReconstructionT', 'JointCurveballReconstructionT']

class JointSecondOrderReconstructionT(ReconstructionT):
    """The run_methods here are very ad-hoc for now"""
    def _attachModelPredictionsSecondOrder(self, map_preds_fn: Callable = None, map_data_fn: Callable=None):
        with self.graph.as_default():

            if map_preds_fn is None:
                map_preds_fn = lambda x: x
            if map_data_fn is None:
                map_data_fn = lambda x: x
            self._predictions_fn = lambda v: map_preds_fn(self.fwd_model.predict_second_order(v, self._batch_train_input_v))
            self._batch_train_predictions_t = self._predictions_fn(self.fwd_model.joint_v)

            self._batch_train_data_t = map_data_fn(tf.reshape(tf.gather(self._amplitudes_t,
                                                            self._batch_train_input_v),
                                                  [-1]))
            self._batch_validation_data_t = map_data_fn(tf.reshape(tf.gather(self._amplitudes_t,
                                                                 self._batch_validation_input_v),
                                                       [-1]))
    def attachLossFunctionSecondOrder(self, loss_type: str, loss_init_extra_kwargs: dict=None):
        losses_all = ptychoSampling.reconstruction.options.OPTIONS["loss functions"]

        self._checkConfigProperty(losses_all, loss_type)
        loss_method = losses_all[loss_type]
        if loss_init_extra_kwargs is None:
            loss_init_extra_kwargs = {}

        if 'background_level' in loss_init_extra_kwargs and self.background_level is not None:
            raise ValueError("Cannot supply background level in loss argument if " +
                             "'background_level' in main class is not None.")
        loss_init_args = {'background_level': self.background_level}
        loss_init_args.update(loss_init_extra_kwargs)
        #if "epsilon" not in loss_init_args:
        #    loss_init_args["epsilon"] = self._eps # * np.max(self.amplitudes**2)

        self._checkAttr("fwd_model", "loss functions")

        with self.graph.as_default():
            self._loss_method = loss_method(**loss_init_args)
            self._attachModelPredictionsSecondOrder(self._loss_method.map_preds_fn, self._loss_method.map_data_fn)

            self._train_loss_fn = lambda p: self._loss_method.loss_fn(p, self._batch_train_data_t)
            self._train_loss_t = self._train_loss_fn(self._predictions_fn(self.fwd_model.joint_v))

            if hasattr(self._loss_method, "hessian_fn"):
                self._train_hessian_fn = lambda p: self._loss_method.hessian_fn(p, self._batch_train_data_t)
            else:
                self._train_hessian_fn = None

            self._batch_validation_predictions_t = self.fwd_model.predict_second_order(
                self.fwd_model.joint_v,
                self._batch_validation_input_v)
            self._validation_loss_t = self._loss_method.loss_fn(self._batch_validation_predictions_t,
                                                                self._batch_validation_data_t)

    def _updateOutputs(self):
        self.obj.array = self.session.run(self.fwd_model.obj_cmplx_t)
        self.probe.wavefront[:] = self.session.run(self.fwd_model.probe_cmplx_t)

    def _setObjRegistration(self, obj_array_true, both_registration_nlse=True):
        self.addCustomMetricToDataLog(title="obj_error",
                                      tensor=self.fwd_model.obj_cmplx_t,
                                      log_epoch_frequency=self._registration_log_frequency,
                                      registration_ground_truth=obj_array_true)
        if both_registration_nlse:
            self.addCustomMetricToDataLog(title="obj_nlse",
                                          tensor=self.fwd_model.obj_cmplx_t,
                                          log_epoch_frequency=self._registration_log_frequency,
                                          registration=False,
                                          normalized_lse=True,
                                          registration_ground_truth=obj_array_true)

    def _setProbeRegistration(self, probe_wavefront_true, both_registration_nlse=True):
        self.addCustomMetricToDataLog(title="probe_error",
                                      tensor=self.fwd_model.probe_cmplx_t,
                                      log_epoch_frequency=self._registration_log_frequency,
                                      registration_ground_truth=probe_wavefront_true)
        if both_registration_nlse:
            self.addCustomMetricToDataLog(title="probe_nlse",
                                          tensor=self.fwd_model.probe_cmplx_t,
                                          log_epoch_frequency=self._registration_log_frequency,
                                          registration=False,
                                          normalized_lse=True,
                                          registration_ground_truth=probe_wavefront_true)


class JointSecondOrderFarfieldReconstructionT(JointSecondOrderReconstructionT):

    def _getObjScaling(self, loss_data_type: str, hessian_t: tf.Tensor) -> tf.Tensor:
        with self.graph.as_default():
            with tf.name_scope('obj_scaling'):
                if loss_data_type == "amplitude":
                    weights = tf.ones_like(hessian_t)
                elif loss_data_type == "intensity":
                    weights = tf.reshape(self._batch_train_predictions_t, [-1, *self.probe.shape])

                weights = tf.reduce_mean(weights * hessian_t, axis=(1,2))
                probe_abs_sq = fftshift_t(tf.abs(self.fwd_model.probe_cmplx_t)) ** 2

                #weights_this = fftshift_t(tf.abs(self.fwd_model.probe_cmplx_t)) ** 2 * 0.5
                batch_obj_view_indices = tf.gather(self.fwd_model._obj_view_indices_t, self._batch_train_input_v)
                batch_obj_view_indices = tf.unstack(batch_obj_view_indices)
                size = self.obj.bordered_array.size

                tf_mat = tf.zeros(size, dtype=self.dtype)
                for i, b in enumerate(batch_obj_view_indices):
                    weights_this = 0.5 * probe_abs_sq * weights[i]
                    mat_this = tf.scatter_nd(indices=tf.reshape(b, [-1, 1]),
                                             shape=[size],
                                             updates=tf.reshape(weights_this, [-1]))
                    tf_mat = tf_mat + mat_this
                # zero_condition = tf.less(tf_mat, 1e-10 * tf.reduce_max(tf_mat))
                # zero_case = tf.ones_like(tf_mat) * (1e-6 * tf.reduce_max(tf_mat))
                # H = tf.where(zero_condition, zero_case, tf_mat)
                H = tf_mat
                H_reshaped = tf.reshape(H, self.obj.bordered_array.shape)
                (s1, s2), (s3, s4) = self.obj.border_shape
                H_trunc = tf.reshape(H_reshaped[s1:-s2, s3:-s4], [-1])
            return H_trunc

    def _getProbeScaling(self, loss_data_type: str, hessian_t: tf.Tensor) -> tf.Tensor:
        with self.graph.as_default():
            with tf.name_scope('probe_scaling'):
                if loss_data_type == "amplitude":
                    weights = tf.ones_like(hessian_t)
                elif loss_data_type == "intensity":
                    weights = tf.reshape(self._batch_train_predictions_t, [-1, *self.probe.shape])

                weights = tf.reduce_mean(weights * hessian_t, axis=(1,2))
                batch_obj_views = tf.gather(self.fwd_model._obj_views_all_t, self._batch_train_input_v)

                weights = 0.5 * tf.abs(fftshift_t(batch_obj_views))**2 * weights[:,None,None]
                t = tf.reduce_sum(weights, axis=0)
                #t = tf.reduce_sum(tf.abs(fftshift_t(batch_obj_views)) ** 2, axis=0) * 0.5
                # zero_condition = tf.less(t, 1e-10 * tf.reduce_max(t))
                # zero_case = tf.ones_like(t) / (1e-6 * tf.reduce_max(t))
                # H = tf.where(zero_condition, zero_case, 1 / t)

                H_reshaped = tf.reshape(t, [-1])

            return H_reshaped


class JointLMAReconstructionT(JointSecondOrderFarfieldReconstructionT):
    def __init__(self, *args: int,
                 loss_type: str = "least_squared",
                 obj_array_true: np.ndarray = None,
                 probe_wavefront_true: np.ndarray = None,
                 max_cg_iter: int = 100,
                 min_cg_tol: float = 0.1,
                 registration_log_frequency: int = 10,
                 both_registration_nlse: bool = True,
                 opt_init_extra_kwargs: dict = None,
                 obj_abs_proj: bool = True,
                 obj_abs_max: float = 1.0,
                 probe_abs_max: float = None,
                 loss_init_extra_kwargs: dict = None,
                 r_factor_log: bool = True,
                 update_delay_probe: int = 0,
                 reconstruct_probe: bool = True,
                 apply_diag_mu_scaling: bool = True,
                 apply_precond: bool = False,
                 apply_precond_and_scaling: bool = False,
                 stochastic_diag_estimator_type: str = None,
                 stochastic_diag_estimator_iters: int = 1,
                 **kwargs: int):
        logger.info('initializing...')
        super().__init__(*args, **kwargs)

        if apply_precond_and_scaling:
            logger.info("Overriding any set values for apply_precond and apply_diag_mu_scaling")
            apply_precond = True
            apply_diag_mu_scaling = True

        if stochastic_diag_estimator_type is not None:
            if apply_diag_mu_scaling or apply_precond:
                e = ValueError("Cannot use analytical precond and diag ggn elems if stochastic calculation is enabled.")
                logger.error(e)
                raise e

        if self.training_batch_size != self.n_train:
            e = ValueError("LMA reconstruction does not support minibatch reconstruction.")
            logger.error(e)
            raise e

        logger.info('attaching fwd model...')
        self._attachCustomForwardModel(JointFarfieldForwardModelT,
                                       obj_abs_proj=obj_abs_proj,
                                       obj_abs_max=obj_abs_max,
                                       probe_abs_max=probe_abs_max)

        logger.info('creating loss fn...')

        print('Loss init args', loss_init_extra_kwargs)
        self.attachLossFunctionSecondOrder(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)

        logger.info("creating optimizer...")
        if opt_init_extra_kwargs is None:
            opt_init_extra_kwargs = {}

        if apply_diag_mu_scaling or apply_precond:

            with self.graph.as_default():

                loss_data_type = self._loss_method.data_type
                hessian_t = (tf.ones_like(self._batch_train_predictions_t)
                             * self._train_hessian_fn(self._batch_train_predictions_t))
                hessian_t = tf.reshape(hessian_t, [-1, *self.probe.shape])
                print(hessian_t)
                self._obj_scaling = self._getObjScaling(loss_data_type, hessian_t)
                self._probe_scaling = self._getProbeScaling(loss_data_type, hessian_t)
                scaling_both = tf.concat((self._obj_scaling, self._probe_scaling), axis=0)

                if apply_diag_mu_scaling:
                    self._joint_scaling_t = tf.concat((scaling_both, scaling_both), axis=0)
                    optimizer = ScaledLMAOptimizer
                    opt_init_extra_kwargs['diag_mu_scaling_t'] = self._joint_scaling_t
                if apply_precond:
                    self._joint_precond_t = tf.concat((scaling_both, scaling_both), axis=0)
                    optimizer = PCGLMAOptimizer
                    opt_init_extra_kwargs['diag_precond_t'] = self._joint_precond_t
                    if apply_diag_mu_scaling:
                        optimizer = ScaledPCGLMAOptimizer
                #
        else:
            optimizer = LMAOptimizer

        opt_init_args = {"input_var": self.fwd_model.joint_v,
                         "predictions_fn": self._predictions_fn,
                         "loss_fn": self._train_loss_fn,
                         "diag_hessian_fn": self._train_hessian_fn,
                         "name": "opt",
                         "max_cg_iter": max_cg_iter,
                         "min_cg_tol": min_cg_tol,
                         "stochastic_diag_estimator_type":stochastic_diag_estimator_type,
                         "stochastic_diag_estimator_iters":stochastic_diag_estimator_iters,
                         "assert_tolerances": False}
        opt_init_args.update(opt_init_extra_kwargs)

        self._attachCustomOptimizerForVariable(optimizer,
                                               optimizer_init_args=opt_init_args)
        self.addCustomMetricToDataLog(title="cg_iters",
                                      tensor=self.optimizers[0]._optimizer._total_cg_iterations,
                                      log_epoch_frequency=1)
        self.addCustomMetricToDataLog(title="ls_iters",
                                      tensor=self.optimizers[0]._optimizer._total_proj_ls_iterations,
                                      log_epoch_frequency=1)
        self.addCustomMetricToDataLog(title="proj_iters",
                                      tensor=self.optimizers[0]._optimizer._projected_gradient_iterations,
                                      log_epoch_frequency=1)
        self.addCustomMetricToDataLog(title="mu",
                                      tensor=self.optimizers[0]._optimizer._mu,
                                      log_epoch_frequency=1)

        self._registration_log_frequency = registration_log_frequency
        if obj_array_true is not None:
            self._setObjRegistration(obj_array_true, both_registration_nlse=both_registration_nlse)
        if probe_wavefront_true is not None:
            self._setProbeRegistration(probe_wavefront_true, both_registration_nlse=both_registration_nlse)
        self._addRFactorLog(r_factor_log, registration_log_frequency)


    def run(self, *args, sequential_updates: bool=True, **kwargs):
        super().run(*args, sequential_updates=sequential_updates, **kwargs)


    def getFlopsPerIter(self) -> dict:

        if self.n_validation > 0:
            e = RuntimeWarning("n_validation > 0 can give misleading flops.")
            logger.warning(e)

        if hasattr(self, "session"):
            e = RuntimeWarning("Calculating computational cost after previously initializing the static graph can "
                               + "include inessential calculations (like training and/or validation loss value) "
                               + "and thus give misleading results.")
            logger.warning(e)

        else:
            self.finalizeSetup()
            self.session.run(self._new_train_batch_op)

        for opt in self.optimizers:
            self.session.run(opt.minimize_op)

        total_flops = getComputationalCostInFlops(self.graph)
        cg_init_flops = getComputationalCostInFlops(self.graph,
                                               keywords=[("opt_minimize_step",
                                                          "conjugate_gradient",
                                                          "cg_init")
                                                         ],
                                                   #    "obj_opt_minimize_step/while/conjugate_gradient/cg_while"],
                                                   exclude_keywords=False)
        cg_while_flops = getComputationalCostInFlops(self.graph,
                                               keywords=[("opt_minimize_step",
                                                          "conjugate_gradient",
                                                          "cg_while")
                                                         ],
                                                   #    "obj_opt_minimize_step/while/conjugate_gradient/cg_while"],
                                                   exclude_keywords=False)
        proj_ls_flops = getComputationalCostInFlops(self.graph,
                                                        keywords=[("opt_minimize_step",
                                                                  "proj_ls_linesearch")
                                                                  ],
                                                        exclude_keywords=False)
        opt_only_flops = getComputationalCostInFlops(self.graph,
                                                     keywords=["opt"],
                                                     exclude_keywords=False)
        flops_without_cg_ls = total_flops - cg_while_flops - proj_ls_flops
        d = {"total_flops": total_flops,
             "obj_cg_flops": cg_while_flops,
             "probe_cg_flops": 0,
             "obj_proj_ls_flops": proj_ls_flops,
             "probe_proj_ls_flops": 0,
             "obj_only_flops": opt_only_flops,
             "probe_only_flops": 0,
             "flops_without_cg_ls": flops_without_cg_ls}
        return d

class JointCurveballReconstructionT(JointSecondOrderFarfieldReconstructionT):
    def __init__(self, *args: int,
                 loss_type: str = "least_squared",
                 obj_array_true: np.ndarray = None,
                 probe_wavefront_true: np.ndarray = None,
                 update_delay_probe: int = 0,
                 update_frequency_probe: int = 1,
                 update_frequency_obj: int = 1,
                 reconstruct_probe: bool = True,
                 registration_log_frequency: int = 1,
                 opt_init_extra_kwargs: dict = None,
                 obj_abs_proj: bool = True,
                 loss_init_extra_kwargs: dict = None,
                 r_factor_log: bool = True,
                 #### These two parameters are experimental only. Not to be used in production simulations.
                 # apply_diag_mu_scaling: bool = False, # Does not help
                 apply_precond: bool = False,
                 ###########################################################################################
                 **kwargs: int):
        print("opt_init_extra_kwargs", opt_init_extra_kwargs)
        print("Loss init args", loss_init_extra_kwargs)
        print("update_delay_probe", update_delay_probe, "update_frequency", update_frequency_probe)
        print("update_frequency_obj", update_frequency_obj)

        logger.info('initializing...')
        super().__init__(*args, **kwargs)

        logger.info('attaching fwd model...')
        self._attachCustomForwardModel(JointFarfieldForwardModelT,
                                       obj_abs_proj=obj_abs_proj)

        logger.info('creating loss fn...')
        self.attachLossFunctionSecondOrder(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)

        logger.info("creating optimizers...")

        if opt_init_extra_kwargs is None:
            opt_init_extra_kwargs = {}

        opt_init_args = {"input_var": self.fwd_model.joint_v,
                         "predictions_fn": self._predictions_fn,
                         "loss_fn": self._train_loss_fn,
                         "diag_hessian_fn": self._train_hessian_fn,
                         "damping_factor": 10.0,
                         "damping_update_frequency": 5,
                         "damping_update_factor": 0.99,
                         "name": "opt"}

        # Experimental only.########################################################################################
        if apply_precond:
            loss_data_type = self._loss_method.data_type

            with self.graph.as_default():
                hessian_t = (tf.ones_like(self._batch_train_predictions_t)
                             * self._train_hessian_fn(self._batch_train_predictions_t))
                hessian_t = tf.reshape(hessian_t, [-1, *self.probe.shape])

                obj_scaling_t = self._getObjScaling(loss_data_type, hessian_t)
                probe_scaling_t = self._getProbeScaling(loss_data_type, hessian_t)
                scaling_both = tf.concat((obj_scaling_t, probe_scaling_t), axis=0)
                self._joint_scaling_t = tf.concat((scaling_both, scaling_both), axis=0)
            opt_init_args['diag_precond_t'] = self._joint_scaling_t
        #############################################################################################################

        opt_init_args.update(opt_init_extra_kwargs)
        self._attachCustomOptimizerForVariable(CurveballOptimizer,
                                               optimizer_init_args=opt_init_args)

        self.addCustomMetricToDataLog(title="mu",
                                      tensor=self.optimizers[0]._optimizer._damping_factor,
                                      log_epoch_frequency=1)

        if obj_array_true is not None:
            self.addCustomMetricToDataLog(title="obj_error",
                                          tensor=self.fwd_model.obj_cmplx_t,
                                          log_epoch_frequency=registration_log_frequency,
                                          registration_ground_truth=obj_array_true)
        if (probe_wavefront_true is not None):
            self.addCustomMetricToDataLog(title="probe_error",
                                          tensor=self.fwd_model.probe_cmplx_t,
                                          log_epoch_frequency=registration_log_frequency,
                                          registration_ground_truth=probe_wavefront_true)
        self._addRFactorLog(r_factor_log, registration_log_frequency)

    def getFlopsPerIter(self) -> int:

        if self.n_validation > 0:
            e = RuntimeWarning("n_validation > 0 can give misleading flops.")
            logger.warning(e)

        if hasattr(self, "session"):
            e = RuntimeWarning("Calculating computational cost after previously initializing the static graph can "
                               + "include inessential calculations (like training and/or validation loss value) "
                               + "and thus give misleading results.")
            logger.warning(e)

        else:
            with self.graph.as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                logger.info("Initializing the session.")
                self.session = tf.Session(config=config)
                self.session.run(tf.global_variables_initializer())

        for opt in self.optimizers:
            self.session.run(opt.minimize_op)

        total_flops = getComputationalCostInFlops(self.graph)
        return total_flops

class JointCGReconstructionT(JointSecondOrderFarfieldReconstructionT):
    def __init__(self, *args: int,
                 loss_type: str = "least_squared",
                 obj_array_true: np.ndarray = None,
                 probe_wavefront_true: np.ndarray = None,
                 update_delay_probe: int = 0,
                 update_delay_obj: int = 0,
                 reconstruct_probe: bool = True,
                 registration_log_frequency: int = 10,
                 opt_init_extra_kwargs: dict = None,
                 obj_abs_proj: bool = True,
                 loss_init_extra_kwargs: dict = None,
                 r_factor_log: bool = True,
                 apply_precond: bool = False,
                 **kwargs: int):
        logger.info('initializing...')
        super().__init__(*args, **kwargs)

        if self.training_batch_size != self.n_train:
            e = ValueError("Conjugate gradient reconstruction does not support minibatch reconstruction.")
            logger.error(e)
            raise e

        logger.info('attaching fwd model...')
        self._attachCustomForwardModel(JointFarfieldForwardModelT,
                                       obj_abs_proj=obj_abs_proj)

        logger.info('creating loss fn...')
        self.attachLossFunctionSecondOrder(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)

        logger.info("creating optimizers...")
        if opt_init_extra_kwargs is None:
            opt_init_extra_kwargs = {}

        self._preconditioner = None
        if apply_precond:
            with self.graph.as_default():
                loss_data_type = self._loss_method.data_type
                hessian_t = (tf.ones_like(self._batch_train_predictions_t)
                             * self._train_hessian_fn(self._batch_train_predictions_t))
                hessian_t = tf.reshape(hessian_t, [-1, *self.probe.shape])

                obj_scaling = self._getObjScaling(loss_data_type, hessian_t)
                probe_scaling = self._getProbeScaling(loss_data_type, hessian_t)
                scaling_both = tf.concat((obj_scaling, probe_scaling), axis=0)
                scaling_both = tf.concat((scaling_both, scaling_both), axis=0)
                zero_condition = tf.less(scaling_both, 1e-10 * tf.reduce_max(scaling_both))
                zero_case = tf.ones_like(scaling_both) / (1e-8 * tf.reduce_max(scaling_both))
                self._preconditioner = tf.where(zero_condition, zero_case, 1/scaling_both)

        opt_init_args = {"input_var": self.fwd_model.joint_v,
                         "predictions_fn": self._predictions_fn,
                         "loss_fn": self._train_loss_fn,
                         "name": "opt",
                         "diag_precondition_t": self._preconditioner}
        opt_init_args.update(opt_init_extra_kwargs)
        self._attachCustomOptimizerForVariable(ConjugateGradientOptimizer,
                                               optimizer_init_args=opt_init_args)
        self.addCustomMetricToDataLog(title="ls_iters",
                                      tensor=self.optimizers[0]._optimizer._linesearch_steps,
                                      log_epoch_frequency=1)
        self.addCustomMetricToDataLog(title="alpha",
                                      tensor=self.optimizers[0]._optimizer._linesearch._alpha,
                                      log_epoch_frequency=1)

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
        self._addRFactorLog(r_factor_log, registration_log_frequency)

    def getFlopsPerIter(self) -> dict:

        if self.n_validation > 0:
            e = RuntimeWarning("n_validation > 0 can give misleading flops.")
            logger.warning(e)

        if hasattr(self, "session"):
            e = RuntimeWarning("Calculating computational cost after previously initializing the static graph can "
                               + "include inessential calculations (like training and/or validation loss value) "
                               + "and thus give misleading results.")
            logger.warning(e)

        else:
            with self.graph.as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                logger.info("Initializing the session.")
                self.session = tf.Session(config=config)
                self.session.run(tf.global_variables_initializer())

        for opt in self.optimizers:
            self.session.run(opt.minimize_op)

        total_flops = getComputationalCostInFlops(self.graph)

        ls_flops = getComputationalCostInFlops(self.graph,
                                                   keywords=[("opt_minimize_step",
                                                              "backtracking_linesearch")
                                                             ],
                                                   exclude_keywords=False)

        opt_only_flops = getComputationalCostInFlops(self.graph,
                                                     keywords=["opt"],
                                                     exclude_keywords=False)

        flops_without_ls = total_flops - ls_flops
        d = {"total_flops": total_flops,
             "obj_ls_flops": ls_flops,
             "probe_ls_flops": 0,
             "obj_only_flops": opt_only_flops,
             "probe_only_flops": 0,
             "flops_without_ls": flops_without_ls}
        return d

    # def _getObjScaling(self) -> tf.Tensor:
    #     with self.graph.as_default():
    #         with tf.name_scope('obj_scaling'):
    #             probe_sq = tf.abs(fftshift_t(self.fwd_model.probe_cmplx_t)) ** 2 * 0.5
    #             batch_obj_view_indices = tf.gather(self.fwd_model._obj_view_indices_t, self._batch_train_input_v)
    #             batch_obj_view_indices = tf.unstack(batch_obj_view_indices)
    #             size = self.obj.bordered_array.size
    #             tf_mat = tf.zeros(size, dtype=tf.float32)
    #             for i, b in enumerate(batch_obj_view_indices):
    #                 mat_this = tf.scatter_nd(indices=tf.reshape(b, [-1, 1]),
    #                                          shape=[size],
    #                                          updates=tf.reshape(probe_sq, [-1]))
    #                 tf_mat = tf_mat + mat_this
    #             H_reshaped = tf.reshape(tf_mat, self.obj.bordered_array.shape)
    #             (s1, s2), (s3, s4) = self.obj.border_shape
    #             H_trunc = tf.reshape(H_reshaped[s1:-s2, s3:-s4], [-1])
    #         return H_trunc
    #
    # def _getProbeScaling(self) -> tf.Tensor:
    #     with self.graph.as_default():
    #         with tf.name_scope('probe_scaling'):
    #             batch_obj_views = tf.gather(self.fwd_model._obj_views_all_t, self._batch_train_input_v)
    #             t = tf.reduce_sum(tf.abs(fftshift_t(batch_obj_views)) ** 2, axis=0) * 0.5
    #             #zero_condition = tf.less(t, 1e-2 * tf.reduce_max(t))
    #             #zero_case = tf.ones_like(t)  * (1e-2 * tf.reduce_max(t))
    #             #H = tf.where(zero_condition, zero_case, t)
    #
    #             H_reshaped = tf.reshape(t, [-1])# + 1e-10
    #         return H_reshaped