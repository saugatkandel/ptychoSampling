import numpy as np
import tensorflow as tf
import abc
from ptychoSampling.reconstruction.optimization_t import LMAOptimizer, ScaledLMAOptimizer, \
    CurveballOptimizer, ConjugateGradientOptimizer, PCGLMAOptimizer, ScaledPCGLMAOptimizer
from ptychoSampling.reconstruction.recons import ReconstructionT, SecondOrderReconstructionT
from ptychoSampling.logger import logger
from ptychoSampling.reconstruction.utils.utils import getComputationalCostInFlops
from ptychoSampling.reconstruction.wavefront_t import fftshift_t




__all__ = ["LMAReconstructionT",
           "CurveballReconstructionT",
           "CGReconstructionT",
           "AcceleratedWirtingerFlowT"]

class SecondOrderFarfieldReconstructionT(SecondOrderReconstructionT):

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
                                          registration_ground_truth=np.array(probe_wavefront_true))


class LMAReconstructionT(SecondOrderFarfieldReconstructionT):
    def __init__(self, *args: int,
                 loss_type: str = "least_squared",
                 obj_array_true: np.ndarray = None,
                 probe_wavefront_true: np.ndarray = None,
                 update_delay_probe: int = 0,
                 update_delay_obj: int = 0,
                 max_cg_iter_obj: int = 100,
                 max_cg_iter_probe: int = 100,
                 min_cg_tol: float = 0.1,
                 reconstruct_probe: bool = False,
                 registration_log_frequency: int = 10,
                 both_registration_nlse: bool = True,
                 obj_opt_init_extra_kwargs: dict = None,
                 probe_opt_init_extra_kwargs: dict = None,
                 obj_abs_proj: bool = True,
                 loss_init_extra_kwargs: dict = None,
                 r_factor_log: bool = True,
                 apply_diag_mu_scaling: bool = False,
                 apply_precond: bool = False,
                 apply_precond_and_scaling: bool = False,
                 stochastic_diag_estimator_type: str = None,
                 stochastic_diag_estimator_iters: int = 1,
                 **kwargs: int):
        logger.info('initializing...')
        super().__init__(*args, **kwargs)
        self._reconstruct_probe = reconstruct_probe
        # this is just a convenience variable
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
        self.attachForwardModel("farfield", obj_abs_proj=obj_abs_proj)

        logger.info('creating loss fn...')

        #if loss_init_extra_kwargs is None:
        #    loss_init_extra_kwargs = {}
        #if loss_type == "poisson_log_likelihood_surrogate":
        #    if n_epsilons == 0:
        #        logger.error("For the poisson surrogate model, the n_epsilons should be > 0")
        #        raise ValueError
        #    loss_init_extra_kwargs = {"n_epsilons": n_epsilons}
        print('Loss init args', loss_init_extra_kwargs)
        self.attachLossFunctionSecondOrder(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)

        logger.info("creating optimizers...")
        self._stochastic_diag_estimator_type = stochastic_diag_estimator_type
        self._stochastic_diag_estimator_iters = stochastic_diag_estimator_iters
        obj_opt_init_args = {"input_var": self.fwd_model.obj_v,
                             "predictions_fn": self._obj_predictions_fn,
                             "loss_fn": self._train_loss_fn,
                             "diag_hessian_fn": self._train_hessian_fn,
                             "name": "obj_opt",
                             "max_cg_iter": max_cg_iter_obj,
                             "min_cg_tol": min_cg_tol,
                             "stochastic_diag_estimator_type":stochastic_diag_estimator_type,
                             "stochastic_diag_estimator_iters":stochastic_diag_estimator_iters,
                             "assert_tolerances": False}

        if obj_opt_init_extra_kwargs is None:
            obj_opt_init_extra_kwargs = {}

        if apply_diag_mu_scaling or apply_precond:
            loss_data_type = self._loss_method.data_type

            with self.graph.as_default():
                hessian_t = (tf.ones_like(self._batch_train_predictions_t)
                             * self._train_hessian_fn(self._batch_train_predictions_t))
                hessian_t = tf.reshape(hessian_t, [-1, *self.probe.shape])
                obj_scaling_t = self._getObjScaling(loss_data_type, hessian_t)
                self._obj_scaling_t = tf.concat((obj_scaling_t, obj_scaling_t), axis=0)

                if reconstruct_probe:
                    probe_scaling_t = self._getProbeScaling(loss_data_type, hessian_t)
                    self._probe_scaling_t = tf.concat((probe_scaling_t, probe_scaling_t), axis=0)

            if apply_diag_mu_scaling:
                optimizer_class = ScaledLMAOptimizer
                obj_opt_init_args['diag_mu_scaling_t'] = self._obj_scaling_t

            if apply_precond:
                optimizer_class = PCGLMAOptimizer
                obj_opt_init_args['diag_precond_t'] = self._obj_scaling_t
                if apply_diag_mu_scaling:
                    optimizer_class = ScaledPCGLMAOptimizer
        else:
            optimizer_class = LMAOptimizer

        print('obj_opt_init_extra_kwargs', obj_opt_init_extra_kwargs)
        obj_opt_init_args.update(obj_opt_init_extra_kwargs)
        self._attachCustomOptimizerForVariable(optimizer_class,
                                               optimizer_init_args=obj_opt_init_args,
                                               initial_update_delay=update_delay_obj)
        self.addCustomMetricToDataLog(title="obj_cg_iters",
                                      tensor=self.optimizers[0]._optimizer._total_cg_iterations,
                                      log_epoch_frequency=1)
        self.addCustomMetricToDataLog(title="obj_ls_iters",
                                      tensor=self.optimizers[0]._optimizer._total_proj_ls_iterations,
                                      log_epoch_frequency=1)
        self.addCustomMetricToDataLog(title="obj_proj_iters",
                                      tensor=self.optimizers[0]._optimizer._projected_gradient_iterations,
                                      log_epoch_frequency=1)
        self.addCustomMetricToDataLog(title="obj_mu",
                                      tensor=self.optimizers[0]._optimizer._mu,
                                      log_epoch_frequency=1)


        if reconstruct_probe:
            if probe_opt_init_extra_kwargs is None:
                probe_opt_init_extra_kwargs = {}
            probe_opt_init_args = {"input_var": self.fwd_model.probe_v,
                                   "predictions_fn": self._probe_predictions_fn,
                                   "loss_fn": self._train_loss_fn,
                                   "diag_hessian_fn": self._train_hessian_fn,
                                   "name": "probe_opt",
                                   "max_cg_iter": max_cg_iter_probe,
                                   "min_cg_tol": min_cg_tol,
                                   "stochastic_diag_estimator_type":stochastic_diag_estimator_type,
                                   "stochastic_diag_estimator_iters":stochastic_diag_estimator_iters,
                                   "assert_tolerances": False}

            if apply_diag_mu_scaling:
                probe_opt_init_args['diag_mu_scaling_t'] = self._probe_scaling_t
            if apply_precond:
                probe_opt_init_args['diag_precond_t'] = self._probe_scaling_t


            probe_opt_init_args.update(probe_opt_init_extra_kwargs)
            self._attachCustomOptimizerForVariable(optimizer_class,
                                                   optimizer_init_args=probe_opt_init_args,
                                                   initial_update_delay=update_delay_probe)
            self.addCustomMetricToDataLog(title="probe_cg_iters",
                                          tensor=self.optimizers[1]._optimizer._total_cg_iterations,
                                          log_epoch_frequency=1)
            self.addCustomMetricToDataLog(title="probe_ls_iters",
                                          tensor=self.optimizers[1]._optimizer._total_proj_ls_iterations,
                                          log_epoch_frequency=1)
            self.addCustomMetricToDataLog(title="probe_proj_iters",
                                          tensor=self.optimizers[1]._optimizer._projected_gradient_iterations,
                                          log_epoch_frequency=1)
            self.addCustomMetricToDataLog(title="probe_mu",
                                          tensor=self.optimizers[1]._optimizer._mu,
                                          log_epoch_frequency=1)

        self._registration_log_frequency = registration_log_frequency
        if obj_array_true is not None:
            self._setObjRegistration(obj_array_true, both_registration_nlse=both_registration_nlse)
        if reconstruct_probe and (probe_wavefront_true is not None):
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
            with self.graph.as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                logger.info("Initializing the session.")
                self.session = tf.Session(config=config)
                self.session.run(tf.global_variables_initializer())

        for opt in self.optimizers:
            self.session.run(opt.minimize_op)

        total_flops = getComputationalCostInFlops(self.graph)
        obj_cg_flops = getComputationalCostInFlops(self.graph,
                                                   keywords=[("obj_opt_minimize_step",
                                                              "conjugate_gradient",
                                                              "cg_while")
                                                             ],
                                                   #    "obj_opt_minimize_step/while/conjugate_gradient/cg_while"],
                                                   exclude_keywords=False)
        obj_proj_ls_flops = getComputationalCostInFlops(self.graph,
                                                        keywords=[("obj_opt_minimize_step",
                                                                  "proj_ls_linesearch")
                                                                  ],
                                                        exclude_keywords=False)
        probe_cg_flops = getComputationalCostInFlops(self.graph,
                                                   keywords=[
                                                       "probe_opt_minimize_step/while/conjugate_gradient/cg_while"],
                                                   exclude_keywords=False)
        obj_only_flops = getComputationalCostInFlops(self.graph,
                                                     keywords=["obj_opt"],
                                                     exclude_keywords=False)
        probe_only_flops = getComputationalCostInFlops(self.graph,
                                                       keywords=["probe_opt"],
                                                       exclude_keywords=False)
        probe_proj_ls_flops = getComputationalCostInFlops(self.graph,
                                                          keywords=[("probe_opt_minimize_step",
                                                                     "proj_ls_linesearch")
                                                                  ],
                                                          exclude_keywords=False)
        flops_without_cg_ls = total_flops - obj_cg_flops - probe_cg_flops - obj_proj_ls_flops - probe_proj_ls_flops
        d = {"total_flops": total_flops,
             "obj_cg_flops": obj_cg_flops,
             "probe_cg_flops": probe_cg_flops,
             "obj_proj_ls_flops": obj_proj_ls_flops,
             "probe_proj_ls_flops": probe_proj_ls_flops,
             "obj_only_flops": obj_only_flops,
             "probe_only_flops": probe_only_flops,
             "flops_without_cg_ls": flops_without_cg_ls}
        return d




class CurveballReconstructionT(SecondOrderFarfieldReconstructionT):
    def __init__(self, *args: int,
                 loss_type: str = "least_squared",
                 obj_array_true: np.ndarray = None,
                 probe_wavefront_true: np.ndarray = None,
                 update_delay_probe: int = 0,
                 update_delay_obj: int = 0,
                 update_frequency_probe: int = 1,
                 update_frequency_obj: int = 1,
                 reconstruct_probe: bool = True,
                 registration_log_frequency: int = 10,
                 both_registration_nlse:bool = True,
                 obj_opt_init_extra_kwargs: dict = None,
                 probe_opt_init_extra_kwargs: dict = None,
                 obj_abs_proj: bool = True,
                 loss_init_extra_kwargs: dict = None,
                 r_factor_log: bool = True,
                 #### These two parameters are experimental only. Not to be used in production simulations.
                 #apply_diag_mu_scaling: bool = False, Not useful.
                 apply_precond: bool = False,
                 ###########################################################################################
                 **kwargs: int):
        print("Loss init args", loss_init_extra_kwargs)
        print("update_delay_probe", update_delay_probe, "update_frequency", update_frequency_probe)
        print("update_frequency_obj", update_frequency_probe)

        logger.info('initializing...')
        super().__init__(*args, **kwargs)

        # this is just a convenience variable
        logger.info('attaching fwd model...')
        self.attachForwardModel("farfield", obj_abs_proj=obj_abs_proj)

        #if loss_init_extra_kwargs is None:
        #    loss_init_extra_kwargs = {}

        logger.info('creating loss fn...')
        self.attachLossFunctionSecondOrder(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)

        logger.info("creating optimizers...")

        if obj_opt_init_extra_kwargs is None:
            obj_opt_init_extra_kwargs = {}

        obj_opt_init_args = {"input_var": self.fwd_model.obj_v,
                             "predictions_fn": self._obj_predictions_fn,
                             "loss_fn": self._train_loss_fn,
                             "diag_hessian_fn": self._train_hessian_fn,
                             "damping_factor": 10.0,
                             "damping_update_frequency": 5,
                             "damping_update_factor": 0.99,
                             "name": "obj_opt"}
        # Experimental only.########################################################################################
        if apply_precond:
            loss_data_type = self._loss_method.data_type

            with self.graph.as_default():
                hessian_t = (tf.ones_like(self._batch_train_predictions_t)
                             * self._train_hessian_fn(self._batch_train_predictions_t))
                hessian_t = tf.reshape(hessian_t, [-1, *self.probe.shape])

                obj_scaling_t = self._getObjScaling(loss_data_type, hessian_t)
                self._obj_scaling_t = tf.concat((obj_scaling_t, obj_scaling_t), axis=0)

                if reconstruct_probe:
                    probe_scaling_t = self._getProbeScaling(loss_data_type, hessian_t)
                    self._probe_scaling_t = tf.concat((probe_scaling_t, probe_scaling_t), axis=0)

            obj_opt_init_args['diag_precond_t'] = self._obj_scaling_t
        #############################################################################################################


        obj_opt_init_args.update(obj_opt_init_extra_kwargs)
        self._attachCustomOptimizerForVariable(CurveballOptimizer,
                                               optimizer_init_args=obj_opt_init_args,
                                               initial_update_delay=update_delay_obj,
                                               update_frequency=update_frequency_obj)
        self.addCustomMetricToDataLog(title="obj_mu",
                                      tensor=self.optimizers[0]._optimizer._damping_factor,
                                      log_epoch_frequency=1)

        if reconstruct_probe:
            if probe_opt_init_extra_kwargs is None:
                probe_opt_init_extra_kwargs = {}
            probe_opt_init_args = {"input_var": self.fwd_model.probe_v,
                                   "predictions_fn": self._probe_predictions_fn,
                                   "loss_fn": self._train_loss_fn,
                                   "diag_hessian_fn": self._train_hessian_fn,
                                   "damping_factor": 10.0,
                                   "damping_update_frequency": 5,
                                   "damping_update_factor": 0.99,
                                   "name": "probe_opt"}

            if apply_precond:
                probe_opt_init_args['diag_precond_t'] = self._probe_scaling_t

            probe_opt_init_args.update(probe_opt_init_extra_kwargs)

            self._attachCustomOptimizerForVariable(CurveballOptimizer,
                                                   optimizer_init_args=probe_opt_init_args,
                                                   initial_update_delay=update_delay_probe,
                                                   update_frequency=update_frequency_probe)
            self.addCustomMetricToDataLog(title="probe_mu",
                                          tensor=self.optimizers[1]._optimizer._damping_factor,
                                          log_epoch_frequency=1)
        self._registration_log_frequency = registration_log_frequency
        if obj_array_true is not None:
            self._setObjRegistration(obj_array_true, both_registration_nlse)

        if reconstruct_probe and (probe_wavefront_true is not None):
            self._setProbeRegistration(probe_wavefront_true, both_registration_nlse)
        self._addRFactorLog(r_factor_log, registration_log_frequency)


    def run(self, *args, sequential_updates: bool=False, **kwargs):
        super().run(*args, sequential_updates=sequential_updates, **kwargs)



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

class CGReconstructionT(SecondOrderFarfieldReconstructionT):
    def __init__(self, *args: int,
                 loss_type: str = "least_squared",
                 obj_array_true: np.ndarray = None,
                 probe_wavefront_true: np.ndarray = None,
                 update_delay_probe: int = 0,
                 update_delay_obj: int = 0,
                 reconstruct_probe: bool = True,
                 registration_log_frequency: int = 10,
                 both_registration_nlse: bool = True,
                 obj_opt_init_extra_kwargs: dict = None,
                 probe_opt_init_extra_kwargs: dict = None,
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
        self.attachForwardModel("farfield", obj_abs_proj=obj_abs_proj)

        logger.info('creating loss fn...')

        self.attachLossFunctionSecondOrder(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)

        logger.info("creating optimizers...")
        if obj_opt_init_extra_kwargs is None:
            obj_opt_init_extra_kwargs = {}

        self._obj_preconditioner = None
        self._probe_preconditioner = None
        if apply_precond:
            loss_data_type = self._loss_method.data_type

            with self.graph.as_default():
                hessian_t = (tf.ones_like(self._batch_train_predictions_t)
                             * self._train_hessian_fn(self._batch_train_predictions_t))
                hessian_t = tf.reshape(hessian_t, [-1, *self.probe.shape])
            self._obj_preconditioner = self._getObjPreconditioner(loss_data_type, hessian_t)
            if reconstruct_probe:
                self._probe_preconditioner = self._getProbePreconditioner(loss_data_type, hessian_t)

        obj_opt_init_args = {"input_var": self.fwd_model.obj_v,
                             "predictions_fn": self._obj_predictions_fn,
                             "loss_fn": self._train_loss_fn,
                             "name": "obj_opt",
                             "diag_precondition_t": self._obj_preconditioner}

        obj_opt_init_args.update(obj_opt_init_extra_kwargs)
        self._attachCustomOptimizerForVariable(ConjugateGradientOptimizer,
                                               optimizer_init_args=obj_opt_init_args,
                                               initial_update_delay=update_delay_obj)
        self.addCustomMetricToDataLog(title="obj_ls_iters",
                                      tensor=self.optimizers[0]._optimizer._linesearch_steps,
                                      log_epoch_frequency=1)
        self.addCustomMetricToDataLog(title="obj_alpha",
                                      tensor=self.optimizers[0]._optimizer._linesearch._alpha,
                                      log_epoch_frequency=1)

        if reconstruct_probe:
            if probe_opt_init_extra_kwargs is None:
                probe_opt_init_extra_kwargs = {}

            probe_opt_init_args = {"input_var": self.fwd_model.probe_v,
                                   "predictions_fn": self._probe_predictions_fn,
                                   "loss_fn": self._train_loss_fn,
                                   "name": "probe_opt",
                                   "diag_precondition_t": self._probe_preconditioner}

            probe_opt_init_args.update(probe_opt_init_extra_kwargs)

            self._attachCustomOptimizerForVariable(ConjugateGradientOptimizer,
                                                   optimizer_init_args=probe_opt_init_args,
                                                   initial_update_delay=update_delay_probe)

            self.addCustomMetricToDataLog(title="probe_ls_iters",
                                          tensor=self.optimizers[1]._optimizer._linesearch_steps,
                                          log_epoch_frequency=1)
            self.addCustomMetricToDataLog(title="probe_alpha",
                                          tensor=self.optimizers[1]._optimizer._linesearch._alpha,
                                          log_epoch_frequency=1)

        self._registration_log_frequency = registration_log_frequency
        if obj_array_true is not None:
            self._setObjRegistration(obj_array_true, both_registration_nlse)

        if reconstruct_probe and (probe_wavefront_true is not None):
            self._setProbeRegistration(probe_wavefront_true, both_registration_nlse)
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
            with self.graph.as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                logger.info("Initializing the session.")
                self.session = tf.Session(config=config)
                self.session.run(tf.global_variables_initializer())

        for opt in self.optimizers:
            self.session.run(opt.minimize_op)

        total_flops = getComputationalCostInFlops(self.graph)

        obj_ls_flops = getComputationalCostInFlops(self.graph,
                                                   keywords=[("obj_opt_minimize_step",
                                                              "backtracking_linesearch")
                                                             ],
                                                   exclude_keywords=False)

        obj_only_flops = getComputationalCostInFlops(self.graph,
                                                     keywords=["obj_opt"],
                                                     exclude_keywords=False)
        probe_only_flops = getComputationalCostInFlops(self.graph,
                                                       keywords=["probe_opt"],
                                                       exclude_keywords=False)
        probe_ls_flops = getComputationalCostInFlops(self.graph,
                                                     keywords=[("probe_opt_minimize_step",
                                                                "backtracking_linesearch")
                                                                  ],
                                                          exclude_keywords=False)
        flops_without_ls = total_flops - obj_ls_flops - probe_ls_flops
        d = {"total_flops": total_flops,
             "obj_ls_flops": obj_ls_flops,
             "probe_ls_flops": probe_ls_flops,
             "obj_only_flops": obj_only_flops,
             "probe_only_flops": probe_only_flops,
             "flops_without_ls": flops_without_ls}
        return d

    def _getObjPreconditioner(self, loss_data_type: str, hessian_t: tf.Tensor) -> tf.Tensor:
        with self.graph.as_default():
            with tf.name_scope('obj_preconditioner'):

                diagH = self._getObjScaling(loss_data_type, hessian_t)
                zero_condition = tf.less(diagH, 1e-8 * tf.reduce_max(diagH))
                zero_case = tf.ones_like(diagH) / (1e-8 * tf.reduce_max(diagH))
                invH = tf.where(zero_condition, zero_case, 1/ diagH)
                H_reals = tf.concat([invH, invH], axis=0)
            return H_reals

    def _getProbePreconditioner(self, loss_data_type: str, hessian_t: tf.Tensor) -> tf.Tensor:
        with self.graph.as_default():
            with tf.name_scope('probe_preconditioner'):
                diagH = self._getProbeScaling(loss_data_type, hessian_t)
                zero_condition = tf.less(diagH, 1e-8 * tf.reduce_max(diagH))
                zero_case = tf.ones_like(diagH) / (1e-8 * tf.reduce_max(diagH))
                invH = tf.where(zero_condition, zero_case, 1 / diagH)
                H_reals = tf.concat([invH, invH], axis=0)
            return H_reals

class AcceleratedWirtingerFlowT(ReconstructionT):
    def __init__(self, *args: int,
                 loss_type: str = "least_squared",
                 obj_array_true: np.ndarray = None,
                 probe_wavefront_true: np.ndarray = None,
                 update_delay_probe: int = 0,
                 update_delay_obj: int = 0,
                 reconstruct_probe: bool = True,
                 registration_log_frequency: int = 10,
                 both_registration_nlse: bool = True,
                 obj_opt_init_extra_kwargs: dict = None,
                 probe_opt_init_extra_kwargs: dict = None,
                 obj_abs_proj: bool = True,
                 loss_init_extra_kwargs: dict = None,
                 r_factor_log: bool = True,
                 use_momentum: bool = False,
                 **kwargs: int):
        logger.info('initializing...')
        super().__init__(*args, **kwargs)

        if loss_type not in ["least_squared", "gaussian", "gaussian_simple"]:
            e = KeyError("Wirtinger/ePIE method supported only for the least squared objective.")
            logger.error(e)
            raise e

        optimizer = "gradient"
        if use_momentum:
            optimizer = "momentum"
            #if reconstruct_probe:
            #    e = KeyError("Momentum supported for obj reconstruction only.")
            #    logger.error(e)
            #    raise e

        logger.info('attaching fwd model...')
        self.attachForwardModel("farfield", obj_abs_proj=obj_abs_proj)
        logger.info('creating loss fn...')
        self.attachLossFunction(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)

        logger.info("create learning rates")
        self._lr_obj = self._getObjLearningRate()

        if reconstruct_probe:
            self._lr_probe = self._getProbeLearningRate()


        logger.info('creating optimizers...')
        if obj_opt_init_extra_kwargs is None:
            obj_opt_init_extra_kwargs = {}
        if use_momentum:
            self._mom_obj = (self._iteration_v + 2) / (self._iteration_v + 5)
            #self._mom_obj = (self._epoch_t + 2) / (self._epoch_t + 5)
            obj_opt_init_extra_kwargs["momentum"] = self._mom_obj
            obj_opt_init_extra_kwargs["use_nesterov"] = True
        self.attachTensorflowOptimizerForVariable("obj",
                                                  optimizer_type=optimizer,
                                                  optimizer_init_args={"learning_rate": self._lr_obj,
                                                                       **obj_opt_init_extra_kwargs},
                                                  initial_update_delay=update_delay_obj)

        if reconstruct_probe:
            if probe_opt_init_extra_kwargs is None:
                probe_opt_init_extra_kwargs = {}
            if use_momentum:
                self._mom_probe = (self._iteration_v + 2) / (self._iteration_v + 5)
                #self._mom_probe = (self._epoch_t + 2) / (self._epoch_t + 5)
                probe_opt_init_extra_kwargs["momentum"] = self._mom_probe
                probe_opt_init_extra_kwargs["use_nesterov"] = True

            self.attachTensorflowOptimizerForVariable("probe",
                                                      optimizer_type=optimizer,
                                                      optimizer_init_args={"learning_rate": self._lr_probe,
                                                                           **probe_opt_init_extra_kwargs},
                                                      initial_update_delay=update_delay_probe)

        self._registration_log_frequency = registration_log_frequency
        if obj_array_true is not None:
            self._setObjRegistration(obj_array_true, both_registration_nlse)

        if reconstruct_probe and (probe_wavefront_true is not None):
            self._setProbeRegistration(probe_wavefront_true, both_registration_nlse)

        self._addRFactorLog(r_factor_log, registration_log_frequency)

    def _getObjLearningRate(self) -> tf.Tensor:
        with self.graph.as_default():
            probe_sq = tf.abs(fftshift_t(self.fwd_model.probe_cmplx_t)) ** 2 #* 0.5
            batch_obj_view_indices = tf.gather(self.fwd_model._obj_view_indices_t, self._batch_train_input_v)
            batch_obj_view_indices = tf.unstack(batch_obj_view_indices)
            size = self.obj.bordered_array.size

            tf_mat = tf.zeros(size, dtype=self.dtype)
            for b in batch_obj_view_indices:
                mat_this = tf.scatter_nd(indices=tf.reshape(b, [-1, 1]),
                                        shape=[size],
                                        updates=tf.reshape(probe_sq, [-1]))
                tf_mat = tf_mat + mat_this
            return 1 / tf.reduce_max(tf_mat)

    def _getProbeLearningRate(self) -> tf.Tensor:
        with self.graph.as_default():
            batch_obj_views = tf.gather(self.fwd_model._obj_views_all_t, self._batch_train_input_v) #* 0.5
            return 1 / tf.reduce_max(tf.reduce_sum(tf.abs(fftshift_t(batch_obj_views)) ** 2, axis=0))


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


class GeneralizedAcceleratedWirtingerFlowT(ReconstructionT):

    def __init__(self, *args: int,
                 loss_type: str = "least_squared",
                 obj_array_true: np.ndarray = None,
                 probe_wavefront_true: np.ndarray = None,
                 update_delay_probe: int = 0,
                 update_delay_obj: int = 0,
                 reconstruct_probe: bool = True,
                 registration_log_frequency: int = 10,
                 both_registration_nlse: bool = True,
                 obj_opt_init_extra_kwargs: dict = None,
                 probe_opt_init_extra_kwargs: dict = None,
                 obj_abs_proj: bool = True,
                 loss_init_extra_kwargs: dict = None,
                 r_factor_log: bool = True,
                 use_momentum: bool = False,
                 **kwargs: int):
        logger.info('initializing...')
        super().__init__(*args, **kwargs)

        optimizer = "gradient"
        if use_momentum:
            optimizer = "momentum"
            #if reconstruct_probe:
            #    e = KeyError("Momentum supported for obj reconstruction only.")
            #    logger.error(e)
            #    raise e

        logger.info('attaching fwd model...')
        self.attachForwardModel("farfield", obj_abs_proj=obj_abs_proj)
        logger.info('creating loss fn...')
        self.attachLossFunction(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)
        loss_data_type = self._loss_method.data_type

        logger.info("create learning rates")
        with self.graph.as_default():
        #   self._lr_obj = self._getObjLearningRateOld()
        #    self._lr_probe = self._getProbeLearningRateOld()
            self._train_hessian_fn = lambda p: self._loss_method.hessian_fn(p, self._batch_train_data_t)
        #with self.graph.as_default():
            hessian_t = (tf.ones_like(self._batch_train_predictions_t)
                         * self._train_hessian_fn(self._batch_train_predictions_t))
            hessian_t = tf.reshape(hessian_t, [-1, *self.probe.shape])
            self._hessian_t  = hessian_t
            self._lr_obj = self._getObjLearningRate(loss_data_type, hessian_t)

            if reconstruct_probe:
                self._lr_probe = self._getProbeLearningRate(loss_data_type, hessian_t)

        logger.info('creating optimizers...')
        if obj_opt_init_extra_kwargs is None:
            obj_opt_init_extra_kwargs = {}

        if use_momentum:
            self._mom_obj = (self._iteration_v + 2) / (self._iteration_v + 5)
            #self._mom_obj = (self._epoch_t + 2) / (self._epoch_t + 5)
            obj_opt_init_extra_kwargs["momentum"] = self._mom_obj
            obj_opt_init_extra_kwargs["use_nesterov"] = True
        self.attachTensorflowOptimizerForVariable("obj",
                                                  optimizer_type=optimizer,
                                                  optimizer_init_args={"learning_rate": self._lr_obj,
                                                                       **obj_opt_init_extra_kwargs},
                                                  initial_update_delay=update_delay_obj)

        if reconstruct_probe:
            if probe_opt_init_extra_kwargs is None:
                probe_opt_init_extra_kwargs = {}
            if use_momentum:
                self._mom_probe = (self._iteration_v + 2) / (self._iteration_v + 5)
                #self._mom_probe = (self._epoch_t + 2) / (self._epoch_t + 5)
                probe_opt_init_extra_kwargs["momentum"] = self._mom_probe
                probe_opt_init_extra_kwargs["use_nesterov"] = True

            self.attachTensorflowOptimizerForVariable("probe",
                                                      optimizer_type=optimizer,
                                                      optimizer_init_args={"learning_rate": self._lr_probe,
                                                                           **probe_opt_init_extra_kwargs},
                                                      initial_update_delay=update_delay_probe)

        self._registration_log_frequency = registration_log_frequency
        if obj_array_true is not None:
            self._setObjRegistration(obj_array_true, both_registration_nlse)

        if reconstruct_probe and (probe_wavefront_true is not None):
            self._setProbeRegistration(probe_wavefront_true, both_registration_nlse)

        self._addRFactorLog(r_factor_log, registration_log_frequency)


    def _getObjLearningRate(self, loss_data_type: str, hessian_t: tf.Tensor) -> tf.Tensor:

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
            return 1 / (tf.reduce_max(H_trunc) * 2)

    def _getProbeLearningRate(self, loss_data_type: str, hessian_t: tf.Tensor) -> tf.Tensor:
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

            return 1 / (tf.reduce_max(H_reshaped) * 2)

    def _getObjLearningRateOld(self) -> tf.Tensor:


        with self.graph.as_default():
            probe_sq = tf.abs(fftshift_t(self.fwd_model.probe_cmplx_t)) ** 2 #* 0.5
            batch_obj_view_indices = tf.gather(self.fwd_model._obj_view_indices_t, self._batch_train_input_v)
            batch_obj_view_indices = tf.unstack(batch_obj_view_indices)
            size = self.obj.bordered_array.size

            tf_mat = tf.zeros(size, dtype=self.dtype)
            for b in batch_obj_view_indices:
                mat_this = tf.scatter_nd(indices=tf.reshape(b, [-1, 1]),
                                        shape=[size],
                                        updates=tf.reshape(probe_sq, [-1]))
                tf_mat = tf_mat + mat_this
            return 1 / tf.reduce_max(tf_mat)

    def _getProbeLearningRateOld(self) -> tf.Tensor:
        with self.graph.as_default():
            batch_obj_views = tf.gather(self.fwd_model._obj_views_all_t, self._batch_train_input_v) #* 0.5
            return 1 / tf.reduce_max(tf.reduce_sum(tf.abs(fftshift_t(batch_obj_views)) ** 2, axis=0))


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