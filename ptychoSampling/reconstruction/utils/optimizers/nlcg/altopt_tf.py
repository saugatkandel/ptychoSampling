"""This module implements the core utilities for alternate optimization algorithms
__author__ = 'Saurabh Adya'
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2019 Apple Inc. All Rights Reserved.
"""
__all__ = ['AltOptimizer']

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import gradient_descent
from tensorflow.python.training.optimizer import _get_processor
from tensorflow.python.ops import gradients_impl

from ptychoSampling.reconstruction.utils.optimizers.nlcg import NCG_OPTIMIZER_TFOP


class AltOptimizer(gradient_descent.GradientDescentOptimizer):
    """AltOptimizer implements Alternate DNN training Algorihtms.
         The alternate algorithms implemented are Ncg(Ncg w TF ops)
      Args:
         learning_rate: Base learning rate to use
         use_locking: Whether to use Locking before updating the weights
         optimizer: Which alternate optimizer to use (Ncg)
         ncg_precond: Preconditioner to use in the Ncg algorithm (Bfgs/None)
         ncg_update_rule: Conjugate direction update rule to use in the Ncg algorithm (PR:PolakRibiere / FR:FletcherReaves)
         line_search: Line search strategy(None/Online/Step)
         step_line_search_period: For Step line_search, comparison period to use
         line_search_threshold: For line search, what is the threshold (in percent) before we decrease the learning rate.
                                If loss function increase is less than this threshold, then we increase the LR upto the base LR schedule
                                If loss function increase is greater than this threshold, we reduce the LR in proportion to the function increase
         verbosity: verbosity level, 0=Default, 1=Info, 2=Debug, 3=Verbose
         name: Name of the alternate optimizer
    """
    def __init__(self,
                 learning_rate,
                 use_locking=False,
                 optimizer='Ncg',
                 ncg_precond='Bfgs',
                 ncg_update_rule='FR',
                 line_search='Step',
                 step_line_search_period=5,
                 line_search_threshold=1,
                 loss=None,
                 verbosity=0,
                 name='Alt_optimizer'):
        # We derive from Gradient Descent with learning rate of 1.0
        # Input learning_rate is used in the AltOptimizer
        super(AltOptimizer, self).__init__(learning_rate=1.0,
                                           use_locking=use_locking,
                                           name=name)
        self.lr = learning_rate
        self.optimizer = optimizer
        self.ncg_precond = ncg_precond
        self.ncg_update_rule = ncg_update_rule
        self.line_search = line_search
        self.step_line_search_period = step_line_search_period
        self.line_search_threshold = line_search_threshold
        self.verbosity = verbosity
        self.assign_op = None
        self.loss = loss

    def alt_dir_and_vars(self, grads_and_vars, loss, global_step):
        """Given aggregated grads, find a new direction using alternate optimizer
        Args:
           grads_and_vars: aggregated grads_and_vars from all workers
           loss: scalar loss tensor
           global_step: global_step
        Returns:
           dir_and_vars: Alternate optimizer computed direction and vars
        """
        lr = self.lr
        grads = []
        vars = []
        for g, v in grads_and_vars:
            g_f = tf.reshape(g, [tf.size(g)])
            grads.append(g_f)

            v_f = tf.reshape(v, [tf.size(v)])
            vars.append(v_f)

        vars = tf.concat(vars, 0)
        grads = tf.concat(grads, 0)
        direction = grads

        if loss is None:
            line_search = 'None'
        else:
            line_search = self.line_search

        if self.optimizer == 'Ncg':
            optimizer = NCG_OPTIMIZER_TFOP(grads, precondition=self.ncg_precond,
                                           line_search=line_search,
                                           step_line_search_period=self.step_line_search_period,
                                           line_search_threshold=self.line_search_threshold,
                                           update_rule=self.ncg_update_rule,
                                           verbosity=self.verbosity)
            direction, self.assign_op = optimizer.get_direction(grads, loss, lr, global_step)
        else:
            raise Exception('Unsupported Alternate Optimizer {}'.format(self.optimizer))

        direction = tf.reshape(tf.concat(direction, 1), [tf.size(grads)])

        dir_and_vars = []
        start = 0
        for g, v in grads_and_vars:
            g_shape = tf.shape(g)
            shape_length = tf.size(g)
            r_s = shape_length

            d_t = tf.slice(direction, [start], [r_s])

            start = start + r_s

            d_t = tf.reshape(d_t, g_shape)

            dir_and_vars.append((d_t, v))

        return dir_and_vars

    GATE_NONE = 0
    GATE_OP = 1
    GATE_GRAPH = 2

    def compute_gradients(self, loss, var_list=None,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        """Compute gradients of "loss" for the variables in "var_list"
        This simply wraps the compute_gradients() from the real optimizer.
        Look at args of of TF GradientDescentOptimizer
        Returns:
        A list of (gradient, variable) pairs.
        """
        self.loss = loss
        grads_and_vars = super(AltOptimizer, self).compute_gradients(loss, var_list=var_list,
                                                                     gate_gradients=gate_gradients,
                                                                     aggregation_method=aggregation_method,
                                                                     colocate_gradients_with_ops=colocate_gradients_with_ops,
                                                                     grad_loss=grad_loss)

        return grads_and_vars

    def update_loss(self, loss):
        self.loss = loss

    def update_averaged_hessian_vector_product(self, avg_Hd):
        self.Hd = avg_Hd

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients to variables.
        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.
        Args:
          grads_and_vars: List of (gradient, variable) pairs as returned by
            `compute_gradients()`.
          global_step: Optional `Variable` to increment by one after the
            variables have been updated.
          name: Optional name for the returned operation.  Default to the
            name passed to the `Optimizer` constructor.
        Returns:
          An `Operation` that applies the specified gradients. If `global_step`
          was not None, that operation also increments `global_step`.
        Raises:
          TypeError: If `grads_and_vars` is malformed.
          ValueError: If none of the variables have gradients.
        """
        # This is a default implementation of apply_gradients() that can be shared
        # by most optimizers.  It relies on the subclass implementing the following
        # run_methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().

        loss = self.loss
        grads_and_vars = self.alt_dir_and_vars(grads_and_vars, loss, global_step)

        grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
        if not grads_and_vars:
            raise ValueError("No variables provided.")
        converted_grads_and_vars = []
        for g, v in grads_and_vars:
            if g is not None:
                try:
                    # Convert the grad to Tensor or IndexedSlices if necessary.
                    g = ops.convert_to_tensor_or_indexed_slices(g)
                except TypeError:
                    raise TypeError(
                        "Gradient must be convertible to a Tensor"
                        " or IndexedSlices, or None: %s" % g)
                if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
                    raise TypeError(
                        "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
            p = _get_processor(v)
            converted_grads_and_vars.append((g, v, p))

        converted_grads_and_vars = tuple(converted_grads_and_vars)
        var_list = [v for g, v, _ in converted_grads_and_vars if g is not None]
        if not var_list:
            raise ValueError("No gradients provided for any variable: %s." %
                             ([str(v) for _, _, v in converted_grads_and_vars],))
        with ops.control_dependencies(None):
            self._create_slots(var_list)
        update_ops = []
        with ops.name_scope(name, self._name) as name:
            self._prepare()
            for grad, var, processor in converted_grads_and_vars:
                if grad is None:
                    continue
                # We colocate all ops created in _apply_dense or _apply_sparse
                # on the same device as the variable.
                with ops.name_scope("update_" + var.op.name), ops.colocate_with(var):
                    update_ops.append(processor.update_op(self, grad))
            if global_step is None:
                apply_updates = self._finish(update_ops, name)
            else:
                with ops.control_dependencies([self._finish(update_ops, "update")]):
                    with ops.colocate_with(global_step):
                        apply_updates = state_ops.assign_add(global_step, 1, name=name).op

            if self.assign_op is not None:
                apply_updates = tf.group(apply_updates, self.assign_op)

            train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
            if apply_updates not in train_op:
                train_op.append(apply_updates)

            return apply_updates

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):
        """Add operations to minimize `loss` by updating `var_list`.
        This method simply combines calls `compute_gradients()` and
        `apply_gradients()`. If you want to process the gradient before applying
        them call `compute_gradients()` and `apply_gradients()` explicitly instead
        of using this function.
        Args:
          loss: A `Tensor` containing the value to minimize.
          global_step: Optional `Variable` to increment by one after the
            variables have been updated.
          var_list: Optional list of `Variable` objects to update to minimize
            `loss`.  Defaults to the list of variables collected in the graph
            under the key `GraphKeys.TRAINABLE_VARIABLES`.
          gate_gradients: How to gate the computation of gradients.  Can be
            `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
          aggregation_method: Specifies the method used to combine gradient terms.
            Valid values are defined in the class `AggregationMethod`.
          colocate_gradients_with_ops: If True, try colocating gradients with
            the corresponding op.
          name: Optional name for the returned operation.
          grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
        Returns:
          An Operation that updates the variables in `var_list`.  If `global_step`
          was not `None`, that operation also increments `global_step`.
        Raises:
          ValueError: If some of the variables are not `Variable` objects.
        """
        grads_and_vars = self.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))

        return self.apply_gradients(grads_and_vars, global_step=global_step,
                                    name=name)