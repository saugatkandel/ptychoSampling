"""This module implements the core utilities for alternate optimization algorithms
__author__ = 'Saurabh Adya'
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2019 Apple Inc. All Rights Reserved.
"""
__all__ = ['NCG_OPTIMIZER_TFOP']

import tensorflow as tf


class NCG_OPTIMIZER_TFOP:
    """ Class to implement NCG Optimizer
    Args:
       g: gradient tensor
       precondition: Which preconditioning method to use (None/Bfgs)
       line_search: Type of line search to perform (None/Online/Step)
       update_rule: which conjugate direction update formula to use
                    (PR: PolakRibiere / FR: Fletcher Reaves)
       step_line_search_period: For Step line_search, comparison period to use
       line_search_threshold: For line search, what is the threshold (in percent) before we decrease the learning rate.
                              If loss function increase is less than this threshold, then we increase the LR upto the base LR schedule
                              If loss function increase is greater than this threshold, we reduce the LR in proportion to the function increase
       verbosity: verbosity level, 0=Default, 1=Info, 2=Debug, 3=Verbose
    Returns:
    """

    def __init__(self, g, precondition='Bfgs', line_search='Step', update_rule='FR', step_line_search_period=5,
                 line_search_threshold=1, verbosity=0):
        print('USING NLCG OPTIMIZER. PRECOND:', precondition, 'UPDATE_RULE:', update_rule)
        self.precondition = precondition
        self.line_search = line_search
        self.update_rule = update_rule
        self.step_line_search_period = step_line_search_period
        self.line_search_threshold = line_search_threshold
        self.verbosity = verbosity

        # Non default experimental switches. Not exposed in API yet
        self.self_scale = 'NONE'  # SPECTRAL_SCALE, NONE
        self.fr_reset_iter = -1
        self.precond_reset_iter = -1

        # Non exposed constants
        self.loss_ema_beta = 0.7

        g_shape = g.get_shape().as_list()
        self.identity = tf.fill(g_shape, 1.0, name='identity')

        self.h_diag = tf.get_variable(name='NCG_h_diag', shape=g_shape, dtype=tf.float32,
                                      initializer=tf.initializers.ones, regularizer=None, trainable=False,
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.h_diag_eps = tf.get_variable(name='NCG_h_diag_eps', shape=g_shape, dtype=tf.float32,
                                          initializer=tf.initializers.constant(1e-8), regularizer=None, trainable=False,
                                          collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.g_old = tf.get_variable(name='NCG_g_old', shape=g_shape, dtype=tf.float32,
                                     initializer=tf.initializers.zeros, regularizer=None, trainable=False,
                                     collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.rtd_new = tf.get_variable(name='NCG_rtd_new', shape=[], dtype=tf.float32,
                                       initializer=tf.initializers.zeros, regularizer=None, trainable=False,
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.s = tf.get_variable(name='NCG_s', shape=g_shape, dtype=tf.float32, initializer=tf.initializers.zeros,
                                 regularizer=None, trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.d = tf.get_variable(name='NCG_d', shape=g_shape, dtype=tf.float32, initializer=tf.initializers.zeros,
                                 regularizer=None, trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.eps = tf.fill([], 1e-8, name='eps')

        self.base_lr = tf.get_variable(name='NCG_base_lr', shape=[], dtype=tf.float32, initializer=tf.initializers.ones,
                                       regularizer=None, trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        #self.loss_ema = tf.get_variable(name='NCG_loss_ema', shape=[], dtype=tf.float32,
        #                                initializer=tf.initializers.zeros, regularizer=None, trainable=False,
        #                                collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        #self.loss_ema_old = tf.get_variable(name='NCG_loss_ema_old', shape=[], dtype=tf.float32,
        #                                    initializer=tf.initializers.zeros, regularizer=None, trainable=False,
        #                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES])


        self.global_step = None
        self.local_global_step = False

    def h_diag_identity(self):
        return self.identity

    def compute_bfgs_preconditioner_cond1(self):
        H_diag = tf.add(self.h_diag, self.h_diag_eps)
        H_inverse = tf.reciprocal(H_diag)

        return H_inverse

    def compute_bfgs_preconditioner_cond2(self, ys, y, s, H_diag_s_t, s_t_H_diag_s_t):
        yy_t = tf.multiply(y, y)
        H_diag_s_t_s_t = tf.multiply(H_diag_s_t, s)
        H_diag_s_t_s_t_H_diag = tf.multiply(H_diag_s_t_s_t, self.h_diag)

        ro = tf.add(ys, self.eps)
        ro = tf.reciprocal(ro)
        yy_t = tf.add(yy_t, self.eps)
        yy_t_ro = tf.multiply(yy_t, ro)

        s_t_H_diag_s_t = tf.add(s_t_H_diag_s_t, self.eps)
        H_diag_s_t_s_t_H_diag = tf.add(H_diag_s_t_s_t_H_diag, self.eps)
        correction = tf.divide(H_diag_s_t_s_t_H_diag, s_t_H_diag_s_t)

        if self.self_scale == 'SPECTRAL_SCALE':
            y2 = tf.tensordot(y, y, axes=([0], [0]))
            y2 = tf.add(y2, self.eps)
            y2 = tf.reciprocal(y2)
            yy_t_y2 = tf.multiply(yy_t, y2)
            H_diag = tf.subtract(self.h_diag, correction)
            H_diag = tf.add(H_diag, yy_t_y2)
        else:
            H_diag = tf.subtract(self.h_diag, correction)
            H_diag = tf.add(H_diag, yy_t_ro)

        H_diag = tf.abs(H_diag)

        h_diag_assign_op = tf.assign(self.h_diag, H_diag, name='h_diag_assign')

        with tf.control_dependencies([h_diag_assign_op]):
            H_diag = tf.identity(H_diag)

        H_diag = tf.add(H_diag, self.h_diag_eps)
        H_inverse = tf.reciprocal(H_diag)

        return H_inverse

    def compute_bfgs_preconditioner_reset(self):
        H_inverse = self.h_diag_identity()
        h_diag_reset_assign_op = tf.assign(self.h_diag, H_inverse, name='h_diag_reset_assign')

        with tf.control_dependencies([h_diag_reset_assign_op]):
            H_inverse = tf.identity(H_inverse)

        return H_inverse

    def compute_bfgs_preconditioner(self, g, lr):
        """
        Compute diagonal pre-conditioner for conjugate gradient algorithm
        Args:
         g: Gradient value
         lr: Learning rate
        Returns:
         diag_precond: return the diagonal H_inverse computed from the BFGS algorithm
        """
        g_old = self.g_old

        if self.verbosity > 2:
            g_old = tf.Print(g_old, [g_old], 'GRADIENT_OLD')

        y = tf.subtract(g, g_old)

        if self.verbosity > 2:
            y = tf.Print(y, [y], 'Y')

        lr = tf.multiply(lr, self.base_lr)
        s = tf.multiply(self.d, lr)
        ys = tf.tensordot(y, s, axes=([0], [0]))

        H_diag_s_t = tf.multiply(self.h_diag, s)
        s_t_H_diag_s_t = tf.tensordot(s, H_diag_s_t, axes=([0], [0]))

        H_inverse = tf.cond(tf.logical_or(tf.less(ys, self.eps),
                                          tf.less(s_t_H_diag_s_t, self.eps)),
                            lambda: self.compute_bfgs_preconditioner_cond1(),
                            lambda: self.compute_bfgs_preconditioner_cond2(ys, y, s, H_diag_s_t, s_t_H_diag_s_t))

        return H_inverse

    def compute_preconditioner(self, g, lr, global_step):
        """
        Compute diagonal pre-conditioner for conjugate gradient algorithm
        Args:
         g: Gradient value
         lr: Learning rate
         global_step: global_step
        Returns:
         diag_precond: return the diagonal H_inverse computed from the BFGS algorithm
        """
        if self.precondition == 'None':
            diag_precond = self.h_diag_identity()
        else:
            if self.precond_reset_iter > 0:
                global_step_mod = tf.mod(global_step, 100)
                diag_precond = tf.cond(tf.equal(global_step_mod, 0),
                                       lambda: self.compute_bfgs_preconditioner_reset(),
                                       lambda: self.compute_bfgs_preconditioner(g, lr))
            else:
                diag_precond = tf.cond(tf.equal(global_step, 0),
                                       lambda: self.compute_bfgs_preconditioner_reset(),
                                       lambda: self.compute_bfgs_preconditioner(g, lr))

        if self.verbosity > 0:
            avg_precond = tf.reduce_mean(diag_precond)
            diag_precond = tf.Print(diag_precond, [avg_precond], 'AVG_PRECOND')

        return diag_precond

    def get_initial_dir(self, g, H_inverse):
        """
        Get initial direction
        Args:
         g: Gradient value
         H_inverse: Pre-conditioner (approximation to Hessian inverse)
        Returns:
         d: Direction computed by d = H_inverse*g
        """
        r = tf.negative(g)
        d = tf.multiply(H_inverse, r)
        rtd_new = tf.tensordot(r, d, axes=([0], [0]))

        rtd_new_assign_op = tf.assign(self.rtd_new, rtd_new, name='rtd_new_assign')
        s_assign_op = tf.assign(self.s, d, name='s_assign')
        d_assign_op = tf.assign(self.d, d, name='d_assign')

        with tf.control_dependencies([rtd_new_assign_op, s_assign_op, d_assign_op]):
            d = tf.identity(d)

        return d

    def get_dir(self, g, H_inverse, global_step):
        """
        Core Non-linear conjugate gradient algorithm
        Args:
         g: Gradient value
         H_inverse: Pre-conditioner (approximation to Hessian inverse)
        Returns:
         d: Direction computed from Non-linear Conjugate Gradient algorithm
            NLCG is implemented with Polak-Ribiere formula
        """
        r = tf.negative(g)
        rtd_old = self.rtd_new
        s = tf.multiply(H_inverse, r)
        rtd_new = tf.tensordot(r, s, axes=([0], [0]))

        if self.update_rule == 'PR':
            rtd_mid = tf.tensordot(r, self.s, axes=([0], [0]))
            beta = tf.subtract(rtd_new, rtd_mid)
        else:
            beta = rtd_new

            # optional conjugacy reset conditions in FletcherReaves update
            if self.fr_reset_iter > 0:
                step_mod = tf.mod(global_step, self.fr_reset_iter)
                beta = tf.cond(tf.equal(step_mod, 0),
                               lambda: 0.0,
                               lambda: tf.identity(beta))

        rtd_old = tf.add(rtd_old, self.eps)
        beta = tf.divide(beta, rtd_old)
        beta = tf.reshape(beta, [])
        zero = tf.zeros(shape=[])
        max_beta = tf.fill([], 1.0, name='max_beta')
        beta = tf.maximum(beta, zero)
        beta = tf.minimum(beta, max_beta)

        if self.verbosity > 0:
            beta = tf.Print(beta, [beta], 'NCG_BETA')

        d = tf.scalar_mul(beta, self.d)
        d = tf.add(s, d)

        with tf.control_dependencies([d]):
            rtd_new_assign_op = tf.assign(self.rtd_new, rtd_new, name='rtd_new_assign')
            s_assign_op = tf.assign(self.s, s, name='s_assign')
            d_assign_op = tf.assign(self.d, d, name='d_diag_assign')

        with tf.control_dependencies([rtd_new_assign_op, s_assign_op, d_assign_op]):
            d = tf.identity(d)

        return d

    def online_ls_case1(self):
        return (tf.constant(0.5), tf.constant(0.9))

    def online_ls_case2(self):
        return (tf.constant(0.7), tf.constant(0.7))

    def online_ls_case3(self):
        return (tf.constant(0.8), tf.constant(0.5))

    def online_ls_case4(self):
        return (tf.constant(0.9), tf.constant(0.3))

    def online_ls_case5(self):
        return (tf.constant(0.95), tf.constant(0.2))

    def online_ls_case6(self):
        return (tf.constant(0.975), tf.constant(0.1))

    def online_ls_case7(self):
        return (tf.constant(1.0), tf.constant(0.1))

    def online_ls_casedefault(self):
        return (tf.constant(1.0), tf.constant(0.0))

    def apply_line_search(self, lr, d, base_lr_degrade_ratio, old_dir_ratio, remove_old_dir=True):
        """
        Apply line search
        Args:
         lr: learning rate
         d: computed direction
         base_lr_degrade_ratio: How much to reduce the base Learning Rate
         old_dir_ratio: How much of the last applied direction to remove
        Returns:
         d: direction after line search
         base_lr: New base Learning Rate
        """

        base_lr = tf.multiply(self.base_lr, base_lr_degrade_ratio)
        base_lr = tf.maximum(base_lr, 0.0001)

        base_lr = tf.Print(base_lr, [base_lr], 'BASE_LR: ')

        if remove_old_dir:
            t_new = tf.multiply(lr, base_lr)
            t_old = tf.multiply(lr, self.base_lr)
            dt_old = tf.multiply(self.d, t_old)

            dt_old_remove = tf.multiply(dt_old, old_dir_ratio)
            new_dir_ratio = tf.subtract(tf.constant(1.0), old_dir_ratio)

            dt_new = tf.multiply(d, t_new)
            dt_new_add = tf.multiply(dt_new, new_dir_ratio)

            dt_new = tf.subtract(dt_new_add, dt_old_remove)

            d = tf.cond(tf.greater(t_new, self.eps),
                        lambda: tf.divide(dt_new, t_new),
                        lambda: tf.identity(d))

        return (d, base_lr)

    def default_line_search(self, d):
        """
        Default line search increase the base learning rate by 2% to a max of 1
        Args:
         d: computed direction
        Returns:
         d: direction after line search
        """
        if self.line_search == 'Online':
            base_lr = tf.multiply(self.base_lr, tf.constant(1.02))
        else:
            base_lr = tf.multiply(self.base_lr, tf.constant(1.05))

        base_lr = tf.minimum(base_lr, tf.constant(1.0))
        return (d, base_lr)

    def online_line_search(self, lr, loss, old_loss, d, remove_old_dir=True):
        """
        Online line search will compare the loss. If its high compared to
        old loss, it will try to reduce the learning rate and will also
        try to remove a part of the previous update
        Args:
         lr: learning rate
         loss: Scalar loss tensor
         d: computed direction
        Returns:
         d: direction after online line search
        """
        loss_fn_ratio = tf.divide(loss, old_loss)

        if self.verbosity > 0:
            loss_fn_ratio = tf.Print(loss_fn_ratio, [loss_fn_ratio, loss, old_loss],
                                     '[LOSS_FN_RATIO] [LOSS_FN] [LOSS_FN_OLD] :')

        case_list = []
        case_list.append((tf.greater(loss_fn_ratio, tf.constant(1.0 + 0.15 * self.line_search_threshold)),
                          lambda: self.online_ls_case1()))
        case_list.append((tf.logical_and(
            tf.greater(loss_fn_ratio, tf.constant(1.0 + 0.125 * self.line_search_threshold)),
            tf.less_equal(loss_fn_ratio, tf.constant(1.0 + 0.15 * self.line_search_threshold))),
                          lambda: self.online_ls_case2()))
        case_list.append((tf.logical_and(
            tf.greater(loss_fn_ratio, tf.constant(1.0 + 0.10 * self.line_search_threshold)),
            tf.less_equal(loss_fn_ratio, tf.constant(1.0 + 0.125 * self.line_search_threshold))),
                          lambda: self.online_ls_case3()))
        case_list.append((tf.logical_and(
            tf.greater(loss_fn_ratio, tf.constant(1.0 + 0.075 * self.line_search_threshold)),
            tf.less_equal(loss_fn_ratio, tf.constant(1.0 + 0.10 * self.line_search_threshold))),
                          lambda: self.online_ls_case4()))
        case_list.append((tf.logical_and(
            tf.greater(loss_fn_ratio, tf.constant(1.0 + 0.05 * self.line_search_threshold)),
            tf.less_equal(loss_fn_ratio, tf.constant(1.0 + 0.075 * self.line_search_threshold))),
                          lambda: self.online_ls_case5()))
        case_list.append((tf.logical_and(
            tf.greater(loss_fn_ratio, tf.constant(1.0 + 0.025 * self.line_search_threshold)),
            tf.less_equal(loss_fn_ratio, tf.constant(1.0 + 0.05 * self.line_search_threshold))),
                          lambda: self.online_ls_case6()))
        case_list.append((tf.logical_and(
            tf.greater(loss_fn_ratio, tf.constant(1.0 + 0.01 * self.line_search_threshold)),
            tf.less_equal(loss_fn_ratio, tf.constant(1.0 + 0.025 * self.line_search_threshold))),
                          lambda: self.online_ls_case7()))

        base_lr_degrade_ratio, old_dir_ratio = tf.case(case_list,
                                                       default=lambda: self.online_ls_casedefault())

        if self.verbosity > 1:
            base_lr_degrade_ratio = tf.Print(base_lr_degrade_ratio, [base_lr_degrade_ratio], 'BASE_LR_DEGRADE_RATIO: ')
            old_dir_ratio = tf.Print(old_dir_ratio, [old_dir_ratio], 'OLD_DIR_RATIO: ')

        d, base_lr = tf.cond(tf.less(loss_fn_ratio, tf.constant(1.0 + 0.01 * self.line_search_threshold)),
                             lambda: self.default_line_search(d),
                             lambda: self.apply_line_search(lr, d, base_lr_degrade_ratio, old_dir_ratio,
                                                            remove_old_dir))

        if self.verbosity > 0:
            base_lr = tf.Print(base_lr, [base_lr], 'BASE_LR: ')

        base_lr_assign_op = tf.assign(self.base_lr, base_lr)
        with tf.control_dependencies([base_lr_assign_op]):
            d = tf.identity(d)

        return d

    def assign_loss_ema_old(self, loss):
        """
        Helper function to assign loss to self.loss_ema_old
        Args:
         loss: Scalar loss tensor
        Returns:
         loss: Scalar loss tensor
        """
        loss_ema_old_assign_op = tf.assign(self.loss_ema_old, loss)
        with tf.control_dependencies([loss_ema_old_assign_op]):
            loss = tf.identity(loss)

        return loss

    def step_line_search_dir(self, global_step, lr, loss, d):
        """
        Step Online line search will compare the loss. If its high compared to
        Loss EMA, it will try to reduce the learning rate based on
        loss function ema change in N steps
        Args:
         global_step: global step
         lr: learning rate
         loss: Scalar loss tensor
         d: computed direction
        Returns:
         d: direction after online line search
        """
        mod = tf.floormod(global_step, self.step_line_search_period)
        d = tf.cond(tf.equal(mod, 0),
                    lambda: self.online_line_search(lr, self.loss_ema, self.loss_ema_old, d, remove_old_dir=True),
                    lambda: tf.identity(d))

        with tf.control_dependencies([d]):
            loss = tf.cond(tf.equal(mod, 0),
                           lambda: self.assign_loss_ema_old(self.loss_ema),
                           lambda: tf.identity(loss))

        with tf.control_dependencies([loss]):
            d = tf.identity(d)

        return d

    def step_line_search(self, global_step, lr, loss, d):
        """
        Step Online line search will compare the loss. If its high compared to
        Loss EMA, it will try to reduce the learning rate and will also
        try to remove a part of the previous update
        Args:
         global_step: global step
         lr: learning rate
         loss: Scalar loss tensor
         d: computed direction
        Returns:
         d: direction after online line search
        """
        loss = tf.cond(tf.equal(global_step, 0),
                       lambda: self.assign_loss_ema_old(loss),
                       lambda: tf.identity(loss))

        d = tf.cond(tf.equal(global_step, 0),
                    lambda: tf.identity(d),
                    lambda: self.step_line_search_dir(global_step, lr, loss, d), )

        with tf.control_dependencies([loss]):
            d = tf.identity(d)

        return d

    def compute_loss_ema(self, loss):
        """
        Compute Exponential moving average of the loss
        Args:
         loss: Scalar loss tensor
        Returns:
         loss_ema_assign_op: assign op to update the loss_ema state value
        """
        loss = tf.multiply(loss, 1.0 - self.loss_ema_beta)
        loss_ema = tf.multiply(self.loss_ema, self.loss_ema_beta)
        loss_ema = tf.add(loss_ema, loss)

        if self.verbosity > 1:
            loss_ema = tf.Print(loss_ema, [loss_ema], 'LOSS_EMA: ')

        loss_ema_assign_op = tf.assign(self.loss_ema, loss_ema)
        return loss_ema_assign_op

    def get_direction(self, g, loss, lr, global_step):
        """
        Args:
         g: Gradient value
         loss: Scalar loss tensor
         lr: Learning Rate
         global_step: global_step
        Returns:
         d_g: Direction from Non-linear Conjugate Gradient Optimizer
         assign_op: assignment op that needs to be run before the next training iterations begins
        """

        if global_step is None:
            self.global_step = tf.get_variable(name='NCG_global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.initializers.zeros, regularizer=None, trainable=False,
                                               collections=[tf.GraphKeys.GLOBAL_VARIABLES])
            global_step = self.global_step
            self.local_global_step = True

        if self.verbosity > 0:
            global_step = tf.Print(global_step, [global_step], 'GLOBAL_STEP')
        if self.verbosity > 1:
            lr = tf.Print(lr, [lr], 'LEARNING RATE')
        if self.verbosity > 2:
            g = tf.Print(g, [g], 'GRADIENT')

        H_inverse = self.compute_preconditioner(g, lr, global_step)

        with tf.control_dependencies([H_inverse]):
            g_old_assign_op = tf.assign(self.g_old, g, name='g_old_assign')
        with tf.control_dependencies([g_old_assign_op]):
            H_inverse = tf.identity(H_inverse)

        d = tf.cond(tf.equal(global_step, 0),
                    lambda: self.get_initial_dir(g, H_inverse),
                    lambda: self.get_dir(g, H_inverse, global_step))

        if self.line_search != 'None':
            loss_ema_assign_op = tf.cond(tf.equal(global_step, 0),
                                         lambda: tf.assign(self.loss_ema, loss),
                                         lambda: self.compute_loss_ema(loss))

            with tf.control_dependencies([loss_ema_assign_op]):
                loss = tf.identity(loss)

        if self.line_search == 'Online':
            print('USING ONLINE SEARCH WITH THRESHOLD:', self.line_search_threshold)
            d = tf.cond(tf.equal(global_step, 0),
                        lambda: tf.identity(d),
                        lambda: self.online_line_search(lr, loss, self.loss_ema, d, remove_old_dir=True))
        elif self.line_search == 'Step':
            print('USING STEP LINE SEARCH WITH THRESHOLD: ', self.line_search_threshold, 'AND FN COMPARISON PERIOD',
                  self.step_line_search_period)
            d = self.step_line_search(global_step, lr, loss, d)

        if self.verbosity > 0:
            g_norm = tf.norm(g)
            d_norm = tf.norm(d)
            g_norm = tf.Print(g_norm, [g_norm], 'G_NORM')
            d_norm = tf.Print(d_norm, [d_norm], 'D_NORM')

        normalize_g = tf.nn.l2_normalize(g, 0)
        normalize_d = tf.nn.l2_normalize(d, 0)
        cos_sim = tf.negative(tf.reduce_sum(tf.multiply(normalize_g, normalize_d)))

        if self.verbosity > 0:
            cos_sim = tf.Print(cos_sim, [cos_sim], 'GvsD COS_SIM')

        t = tf.reshape(lr, [])
        t = tf.multiply(t, self.base_lr)
        dt = tf.scalar_mul(t, d)

        d_g = tf.negative(dt)

        if self.verbosity > 2:
            d_g = tf.Print(d_g, [d_g], 'DIRECTION')

        # tf.summary.scalar('G_NORM', g_norm)
        # tf.summary.scalar('D_NORM', d_norm)
        # tf.summary.scalar('GvD_COS_SIM', cos_sim)
        # tf.summary.scalar('BASE_LR', self.base_lr)

        if self.verbosity > 0:
            with tf.control_dependencies([g_norm, d_norm, cos_sim]):
                d_g = tf.identity(d_g)

        if self.local_global_step:
            with tf.control_dependencies([d_g]):
                gs_assign_op = tf.assign(self.global_step, tf.add(self.global_step, tf.constant(1)))
            with tf.control_dependencies([gs_assign_op]):
                d_g = tf.identity(d_g)

        assign_op = None
        return d_g, assign_op