import tensorflow as tf
import abc
from typing import Callable, Union
from sopt.optimizers.tensorflow import LMA, Curveball, NonLinearConjugateGradient, ProjectedGradient
import dataclasses as dt

__all__ = ["Optimizer",
           "GradientDescentOptimizer",
           "AdamOptimizer",
           "LMAOptimizer",
           "ScaledLMAOptimizer",
           "PCGLMAOptimizer",
           "CurveballOptimizer",
           "ConjugateGradientOptimizer",
           "MomentumOptimizer"]


@dt.dataclass
class Optimizer(abc.ABC):
    def __init__(self, initial_update_delay: int = 0,
                 update_frequency: int = 0):
        self.initial_update_delay = initial_update_delay
        self.update_frequency = update_frequency

    @abc.abstractmethod
    def setupMinimizeOp(self, *args, **minimize_kwargs):
        pass

    @property
    @abc.abstractmethod
    def minimize_op(self, *args, **kwargs):
        pass


class AdamOptimizer(Optimizer):
    def __init__(self, initial_update_delay: int = 0,
                 update_frequency: int = 1,
                 learning_rate: float = 1e-2,
                 **extra_init_kwargs: int):
        super().__init__(initial_update_delay, update_frequency)
        self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, **extra_init_kwargs)

    def setupMinimizeOp(self, loss: tf.Tensor,
                        var_list: list,
                        **extra_minimize_kwargs: int):
        self._minimize_op = self._optimizer.minimize(loss=loss,
                                                     var_list=var_list,
                                                     **extra_minimize_kwargs)

    @property
    def minimize_op(self):
        return self._minimize_op

class GradientDescentOptimizer(Optimizer):
    def __init__(self, initial_update_delay: int = 0,
                 update_frequency: int = 1,
                 learning_rate: Union[tf.Tensor, float] = 1e-2,
                 **extra_init_kwargs: int):
        super().__init__(initial_update_delay, update_frequency)
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, **extra_init_kwargs)

    def setupMinimizeOp(self, loss: tf.Tensor,
                        var_list: list,
                        **extra_minimize_kwargs: int):
        self._minimize_op = self._optimizer.minimize(loss=loss,
                                                     var_list=var_list,
                                                     **extra_minimize_kwargs)

    @property
    def minimize_op(self):
        return self._minimize_op


class LMAOptimizer(Optimizer):
    def __init__(self, input_var: tf.Variable,
                 predictions_fn: Callable,
                 loss_fn: Callable,
                 diag_hessian_fn: Callable = None,
                 max_cg_iter: int = 100,
                 initial_update_delay: int = 0,
                 update_frequency: int = 1,
                 **extra_init_kwargs: int):
        super().__init__(initial_update_delay, update_frequency)
        print('Extra initial parameters:', extra_init_kwargs)
        self._optimizer = LMA(input_var=input_var,
                              predictions_fn=predictions_fn,
                              loss_fn=loss_fn,
                              diag_hessian_fn=diag_hessian_fn,
                              max_cg_iter=max_cg_iter,
                              **extra_init_kwargs)


    def setupMinimizeOp(self):
        self._minimize_op = self._optimizer.minimize()

    @property
    def minimize_op(self):
        return self._minimize_op

class ScaledLMAOptimizer(LMAOptimizer):
    """Short class for convenient calling"""
    def __init__(self, *args: int,
                 diag_mu_scaling_t: tf.Tensor,
                 **kwargs: int):
        super().__init__(*args,
                         diag_mu_scaling_t=diag_mu_scaling_t,
                         grad_norm_regularization_power=0.,
                         mu=1.0,
                         **kwargs)

class PCGLMAOptimizer(LMAOptimizer):
    """Short class for convenient calling"""
    def __init__(self, *args: int,
                 diag_precond_t: tf.Tensor,
                 **kwargs: int):
        super().__init__(*args, diag_precond_t=diag_precond_t, **kwargs)

class ScaledPCGLMAOptimizer(LMAOptimizer):
    """Short class for convenient calling"""

    def __init__(self, *args: int,
                 diag_mu_scaling_t: tf.Tensor,
                 diag_precond_t: tf.Tensor,
                 **kwargs: int):
        init_kwargs = {"diag_mu_scaling_t": diag_mu_scaling_t,
                       "diag_precond_t":diag_precond_t,
                       "grad_norm_regularization_power":0,
                       "mu":1.0}
        init_kwargs.update(kwargs)
        #print(init_kwargs)
        super().__init__(*args, **init_kwargs)

class CurveballOptimizer(Optimizer):
    def __init__(self, input_var: tf.Variable,
                 predictions_fn: Callable,
                 loss_fn: Callable,
                 diag_hessian_fn: Callable = None,
                 initial_update_delay: int = 0,
                 update_frequency: int = 1,
                 **extra_init_kwargs: int):
        super().__init__(initial_update_delay, update_frequency)
        self._optimizer = Curveball(input_var=input_var,
                                    predictions_fn=predictions_fn,
                                    loss_fn=loss_fn,
                                    diag_hessian_fn=diag_hessian_fn,
                                    **extra_init_kwargs)


    def setupMinimizeOp(self):
        self._minimize_op = self._optimizer.minimize()

    @property
    def minimize_op(self):
        return self._minimize_op

class MomentumOptimizer(Optimizer):
    def __init__(self, initial_update_delay: int = 0,
                 update_frequency: int = 1,
                 learning_rate: Union[tf.Tensor, float] = 1e-2,
                 momentum: Union[tf.Tensor, float] = 0.9,
                 use_nesterov: bool = True,
                 **extra_init_kwargs: int):
        super().__init__(initial_update_delay, update_frequency)
        self._optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                     momentum=momentum,
                                                     use_nesterov=use_nesterov,
                                                     **extra_init_kwargs)

    def setupMinimizeOp(self, loss: tf.Tensor,
                        var_list: list,
                        **extra_minimize_kwargs: int):
        self._minimize_op = self._optimizer.minimize(loss=loss,
                                                     var_list=var_list,
                                                     **extra_minimize_kwargs)

    @property
    def minimize_op(self):
        return self._minimize_op



class ConjugateGradientOptimizer(Optimizer):
    def __init__(self, input_var: tf.Variable,
                 predictions_fn: Callable,
                 loss_fn: Callable,
                 initial_update_delay: int = 0,
                 update_frequency: int = 1,
                 **extra_init_kwargs: int):
        super().__init__(initial_update_delay, update_frequency)
        self._optimizer = NonLinearConjugateGradient(input_var=input_var,
                                                     predictions_fn=predictions_fn,
                                                     loss_fn=loss_fn,
                                                     **extra_init_kwargs)

    def setupMinimizeOp(self):
        self._minimize_op = self._optimizer.minimize()

    @property
    def minimize_op(self):
        return self._minimize_op

class ProjectedGradientOptimizer(Optimizer):
    def __init__(self, input_var: tf.Variable,
                 loss_fn: Callable,
                 initial_update_delay: int = 0,
                 update_frequency: int = 1,
                 descent_dir_t: tf.Tensor = None,
                 diag_precond_t: tf.Tensor = None,
                 **extra_init_kwargs: int):
        super().__init__(initial_update_delay, update_frequency)
        self._optimizer = ProjectedGradient(input_var=input_var,
                                            loss_fn=loss_fn,
                                            descent_dir_t=descent_dir_t,
                                            diag_precond_t=diag_precond_t,
                                            **extra_init_kwargs)

    def setupMinimizeOp(self):
        self._minimize_op = self._optimizer.minimize()

    @property
    def minimize_op(self):
        return self._minimize_op