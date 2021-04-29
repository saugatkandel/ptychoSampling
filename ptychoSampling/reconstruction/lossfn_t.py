import tensorflow as tf
import numpy as np
import abc
from typing import Union

class LossFunctionT(abc.ABC):
    """
    Parameters
    ----------
    scope_name : str
            Optional parameter that supplies a tensorflow "scope" name to the loss function calculation.
    """
    def __init__(self, scope_name: str = "",
                 background_level:float = 1e-8,
                 scaling_factor: float = 1.0,
                 dtype: str='float32') -> None:
        self._scope_name = scope_name
        self._background_level = background_level
        self._scaling_factor = scaling_factor
        self._dtype = dtype
        print(f"Setting background_level to {background_level:4.3g}...")

    @abc.abstractmethod
    def loss_fn(self, predictions_t, measured_t):
        pass

    @abc.abstractmethod
    def map_preds_fn(self, amplitude_preds_t):
        pass

    @abc.abstractmethod
    def map_data_fn(self, amplitude_measured_t):
        pass

    @property
    @abc.abstractmethod
    def data_type(self):
        pass

class LeastSquaredLossT(LossFunctionT):
    """Get the amplitude (gaussian) loss function for a minibatch.

    The is the amplitude loss, or the loss function for the gaussian noise model. It is a least squares function
    defined as ``1/2 * sum((predicted_data - measured_data)**2)`` [1]_.

    References
    ----------
    .. [1] Godard, P., Allain, M., Chamard, V. & Rodenburg, J. Opt. Express 20, 25914 (2012).
    """

    def map_preds_fn(self, amplitude_preds_t):
        intensities = amplitude_preds_t**2
        return (intensities + self._background_level)**0.5

    def map_data_fn(self, amplitude_measured_t):
        return amplitude_measured_t

    @property
    def data_type(self):
        return "amplitude"

    def loss_fn(self, predictions_t, measured_t):
        """
        Parameters
        ----------
        predictions_t : tensor(float)
            Diffraction data calculated using the forward model for the current minibatch of scan positions. Uses
            the current values of the object and probe variables.
        measured_t : tensor(float)
            Diffraction data measured experimentally for the current minibatch of scan positions.
        Returns
        -------
        loss_t : tensor(float)
            Scalar value for the current loss function.
        """
        scope_name = self._scope_name + "_loss" if self._scope_name else "loss"
        with tf.name_scope(scope_name) as scope:
            return 0.5 * tf.reduce_sum(tf.abs(predictions_t - measured_t) ** 2, name=scope) * self._scaling_factor

    def hessian_fn(self, predictions_t, measured_t):
        """Hessian is a constant diagonal matrix with entries 1.0.
        Parameters
        ----------
        predictions_t : tensor(float)
            Diffraction data calculated using the forward model for the current minibatch of scan positions. Uses
            the current values of the object and probe variables.
        measured_t : tensor(float)
            Diffraction data measured experimentally for the current minibatch of scan positions.
        Returns
        -------
        hessian_t : tensor(float)

        Notes
        -----
        Since the computational cost for the hvp is miniscule (compared to the rest of the procedure),
        actually implementing this function has very little effect on the overall computational cost.
        """
        scope_name= self._scope_name + "_hessian" if self._scope_name else "hessian"
        with tf.name_scope(scope_name) as scope:
            return tf.constant(1.0, dtype=measured_t.dtype) * self._scaling_factor

class PoissonLogLikelihoodLossT(LossFunctionT):
    """Get the maximum likelihood loss function for a minibatch.

    The is the maximum likelihood loss function for the poisson noise model. It is a least squares function
    defined as ``sum((predicted_intensities - actual_counts * log(predicted_intensities))`` [1]_.

    Currently uses the rescaling constant.
    """
    def map_preds_fn(self, amplitude_preds_t: tf.Tensor):
        intensities = amplitude_preds_t ** 2
        return (intensities + self._background_level) ** 0.5

    def map_data_fn(self, amplitude_measured_t):
        return amplitude_measured_t

    @property
    def data_type(self):
        return "amplitude"#"intensity"

    def loss_fn(self, predictions_t, measured_t):
        """
        Parameters
        ----------
        predictions_t : tensor(float)
            Diffraction data calculated using the forward model for the current minibatch of scan positions. Uses
            the current values of the object and probe variables.
        measured_t : tensor(float)
            Diffraction data measured experimentally for the current minibatch of scan positions.
        Returns
        -------
        loss_t : tensor(float)
            Scalar value for the current loss function.
        """
        scope_name = self._scope_name + "_loss" if self._scope_name else "loss"
        with tf.name_scope(scope_name) as scope:
            preds_t = predictions_t**2
            return 0.5 * tf.reduce_sum(preds_t - measured_t**2 * tf.log(preds_t), name=scope) * self._scaling_factor

    def hessian_fn(self, predictions_t, measured_t):
        """Hessian is a constant diagonal matrix with entries 1.0.
        Parameters
        ----------
        predictions_t : tensor(float)
            Diffraction data calculated using the forward model for the current minibatch of scan positions. Uses
            the current values of the object and probe variables.
        measured_t : tensor(float)
            Diffraction data measured experimentally for the current minibatch of scan positions.
        Returns
        -------
        hessian_t : tensor(float)
        """
        scope_name = self._scope_name + "_loss" if self._scope_name else "loss"
        with tf.name_scope(scope_name) as scope:
            #return measured_t / (predictions_t + self._background_level)**2
            numerator = measured_t**2
            denominator = predictions_t**2
            hessian =  1 + numerator / denominator
            return hessian * self._scaling_factor

class PoissonLogLikelihoodSurrogateLossT(LossFunctionT):
    """Get the maximum likelihood loss function for a minibatch.

    The is the maximum likelihood loss function for the poisson noise model. It is a least squares function
    defined as ``sum((predicted_intensities - actual_counts * log(predicted_intensities))`` [1]_.

    Currently uses the rescaling constant.

    Parameters
    ----------
    background_level : float or tf.placeholder
        Small constant to add to the predicted data. Used to stabilize the loss function. Not currently used in some of
        the  loss functions.
    """
    def __init__(self, *args: int,
                 n_spline_epochs: int = 100,
                 **kwargs: int):
        print('kwargs', kwargs)
        print('n_levels', n_spline_epochs)

        super().__init__(*args, **kwargs)
        print('Background level', self._background_level, "n_iterations", n_spline_epochs)
        self._n_spline_epochs = n_spline_epochs
        self._spline_values = tf.constant(np.logspace(0, np.log10(self._background_level),
                                                      n_spline_epochs), dtype='float32')
        #self._spline_values = tf.constant(np.linspace(1, (self._background_level)**(1/4.), n_spline_epochs)**4, dtype='float32')

    def map_preds_fn(self, amplitude_preds_t):
        epoch = tf.reduce_min([amplitude_preds_t.graph.get_tensor_by_name('epoch:0'),
                               self._n_spline_epochs - 1])
        background_level_this = self._spline_values[epoch]

        intensities = amplitude_preds_t**2
        return (intensities + background_level_this) ** 0.5

    def map_data_fn(self, amplitude_measured_t):
        return amplitude_measured_t

    @property
    def data_type(self):
        return "amplitude"

    @property
    def data_type(self):
        return "amplitude"  # "intensity"

    def loss_fn(self, predictions_t, measured_t):
        """
        Parameters
        ----------
        predictions_t : tensor(float)
            Diffraction data calculated using the forward model for the current minibatch of scan positions. Uses
            the current values of the object and probe variables.
        measured_t : tensor(float)
            Diffraction data measured experimentally for the current minibatch of scan positions.
        Returns
        -------
        loss_t : tensor(float)
            Scalar value for the current loss function.
        """
        scope_name = self._scope_name + "_loss" if self._scope_name else "loss"
        with tf.name_scope(scope_name) as scope:
            preds_t = predictions_t ** 2
            return 0.5 * tf.reduce_sum(preds_t - measured_t ** 2 * tf.log(preds_t),
                                       name=scope) * self._scaling_factor

    def hessian_fn(self, predictions_t, measured_t):
        """Hessian is a constant diagonal matrix with entries 1.0.
        Parameters
        ----------
        predictions_t : tensor(float)
            Diffraction data calculated using the forward model for the current minibatch of scan positions. Uses
            the current values of the object and probe variables.
        measured_t : tensor(float)
            Diffraction data measured experimentally for the current minibatch of scan positions.
        Returns
        -------
        hessian_t : tensor(float)
        """
        scope_name = self._scope_name + "_loss" if self._scope_name else "loss"
        with tf.name_scope(scope_name) as scope:
            # return measured_t / (predictions_t + self._background_level)**2
            numerator = measured_t ** 2
            denominator = predictions_t ** 2
            hessian = 1 + numerator / denominator
            return hessian * self._scaling_factor



class CountingModelLossT(LossFunctionT):
    def map_preds_fn(self, amplitude_preds_t):
        return amplitude_preds_t ** 2

    @property
    def data_type(self):
        return "intensity"

    def loss_fn(self, predictions_t, measured_t):
        """
        Parameters
        ----------
        predictions_t : tensor(float)
            Diffraction data calculated using the forward model for the current minibatch of scan positions. Uses
            the current values of the object and probe variables.
        measured_t : tensor(float)
            Diffraction data measured experimentally for the current minibatch of scan positions.
        Returns
        -------
        loss_t : tensor(float)
            Scalar value for the current loss function.
        """
        scope_name = self._scope_name + "_loss" if self._scope_name else "loss"
        with tf.name_scope(scope_name) as scope:
            if len(predictions_t.get_shape().as_list()) == 0:
                return 0.
            mask = tf.greater(measured_t, 0)
            measured_selected_t = tf.boolean_mask(measured_t, mask)
            predicted_selected_t = tf.boolean_mask(predictions_t, mask)
            predicted_remaining_t = tf.boolean_mask(predictions_t, tf.logical_not(mask))
            #term1 = 0.5 * tf.reduce_sum((measured_selected_t - (predicted_selected_t ** 2 + self._background_level)
            #                              / measured_selected_t) ** 2)
            #term2 = 0.5 * tf.reduce_sum(predicted_remaining_t ** 2 + self._background_level)
            #return tf.add(term1, term2, name=scope_name) #/ tf.cast(tf.size(predictions_t), tf.float32)
            term1 = 0.5 * tf.reduce_sum((measured_selected_t - predicted_selected_t - self._background_level)**2
                                         / measured_selected_t)
            term2 = 0.5 * tf.reduce_sum(predicted_remaining_t + self._background_level)
            return term1 + term2

    def hessian_fn(self, predictions_t, measured_t):
        """Hessian is a constant diagonal matrix with entries 1.0.

        Warning: This is NOT positive definite.
        Parameters
        ----------
        predictions_t : tensor(float)
            Diffraction data calculated using the forward model for the current minibatch of scan positions. Uses
            the current values of the object and probe variables.
        measured_t : tensor(float)
            Diffraction data measured experimentally for the current minibatch of scan positions.
        Returns
        -------
        hessian_t : tensor(float)
        """
        scope_name = self._scope_name + "_hessian" if self._scope_name else "hessian"
        with tf.name_scope(scope_name) as scope:
            if len(predictions_t.get_shape().as_list()) == 0:
                return 0.

            mask = tf.greater(measured_t, 0)
            #term1 = (6.0 * predictions_t**2 + 2 * self._background_level) / measured_t**2 - 2.0
            #term2 = tf.ones_like(predictions_t)
            term1 = 1 / measured_t
            term2 = tf.zeros_like(predictions_t)

            hessian = tf.where(mask, term1, term2)
            return hessian #/ tf.cast(tf.size(predictions_t), tf.float32)

class IntensityLeastSquaredLossT(LossFunctionT):
    def map_preds_fn(self, amplitude_preds_t):
        intensities = amplitude_preds_t ** 2
        return intensities + self._background_level

    def map_data_fn(self, amplitude_measured_t):
        return amplitude_measured_t**2

    @property
    def data_type(self):
        return "intensity"

    def loss_fn(self, predictions_t, measured_t):
        """
        Parameters
        ----------
        predictions_t : tensor(float)
            Diffraction data calculated using the forward model for the current minibatch of scan positions. Uses
            the current values of the object and probe variables.
        measured_t : tensor(float)
            Diffraction data measured experimentally for the current minibatch of scan positions.
        Returns
        -------
        loss_t : tensor(float)
            Scalar value for the current loss function.
        """
        scope_name = self._scope_name + "_loss" if self._scope_name else "loss"
        with tf.name_scope(scope_name) as scope:
                return 0.5 * tf.reduce_sum((predictions_t - measured_t) ** 2, name=scope) * self._scaling_factor

    def hessian_fn(self, predictions_t, measured_t):
        """Hessian is a constant diagonal matrix with entries 1.0.
        Parameters
        ----------
        predictions_t : tensor(float)
            Diffraction data calculated using the forward model for the current minibatch of scan positions. Uses
            the current values of the object and probe variables.
        measured_t : tensor(float)
            Diffraction data measured experimentally for the current minibatch of scan positions.
        Returns
        -------
        hessian_t : tensor(float)
        """
        scope_name= self._scope_name + "_hessian" if self._scope_name else "hessian"
        with tf.name_scope(scope_name) as scope:
            return 1.0 * self._scaling_factor