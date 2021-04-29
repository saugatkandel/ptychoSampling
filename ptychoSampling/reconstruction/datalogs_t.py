from pandas import DataFrame
import numpy as np
from ptychoSampling.logger import logger
import tensorflow as tf
import dataclasses as dt
from skimage.feature import register_translation as _register_translation_2d
from ptychoSampling.utils.register_translation_3d import register_translation_3d as _register_translation_3d

__all__ = ["DataLogs"]

@dt.dataclass
class SimpleMetric:
    title: str


@dt.dataclass
class CustomTensorMetric(SimpleMetric):
    tensor: tf.Tensor
    log_epoch_frequency: int
    registration: bool = False
    normalized_lse: bool = False
    true: np.ndarray = None
    columns: list = dt.field(default_factory=list, init=False)  # not in use right now

    def __post_init__(self):
        if self.registration and self.true is None:
            e = ValueError("Require true data for registration.")
            logger.error(e)
            raise e
        if self.normalized_lse and self.true is None:
            e = ValueError("Require true data for normalized error.")
            logger.error(e)
            raise e
        if not (self.registration or self.normalized_lse):
            self.columns = [self.title]
        elif self.registration:
            appends = ['_err']  # , '_shift', '_phase']
            self.columns = [self.title + string for string in appends]
        else:
            appends = ['_nlse']
            self.columns = [self.title + string for string in appends]


class DataLogs:
    def __init__(self, save_name: str = "", save_update_freq: int=100):
        self._save_name = save_name
        self._save_update_freq = save_update_freq
        self._datalog_items = []


    def addSimpleMetric(self, title: str):
        self._datalog_items.append(SimpleMetric(title))


    def addCustomTensorMetric(self, title: str,
                              tensor: tf.Tensor,
                              log_epoch_frequency,
                              **kwargs):

        self._datalog_items.append(CustomTensorMetric(title=title,
                                                      tensor=tensor,
                                                      log_epoch_frequency=log_epoch_frequency,
                                                      **kwargs))


    @staticmethod
    def _register(test: np.ndarray,
                  true: np.ndarray) -> float:
        if len(test.shape) == 2:
            registration_fn = _register_translation_2d
        elif len(test.shape) == 3:
            registration_fn = _register_translation_3d
        else:
            e = ValueError("Subpixel registration only available for 2d and 3d objects.")
            logger.error(e)
            raise e
        shift, err, phase = registration_fn(test, true, upsample_factor=10)
        shift, err, phase = registration_fn(test * np.exp(-1j * phase), true, upsample_factor=10)
        return err

    @staticmethod
    def _normalized_lse(test: np.ndarray,
                        true: np.ndarray) -> float:
        # sometimes these are subclasses of ndarray, which screws up the formatting of the datalog
        test = np.array(test)
        true = np.array(true)
        arg = np.angle(np.sum(np.conj(test) * true))
        phased_test = test * np.exp(1j * arg)
        error = np.sum(np.abs(phased_test - true)**2) / np.sum(np.abs(true)**2)
        return error


    def _getItemFromTitle(self, key: str):
        for item in self._datalog_items:
            if item.title == key:
                return item

    def getCustomTensorMetrics(self, epoch):
        tensors = {}
        for item in self._datalog_items:
            if isinstance(item, CustomTensorMetric):
                if epoch % item.log_epoch_frequency == 0:
                    tensors[item.title] = item.tensor
        return tensors

    def logStep(self, step, log_values_this_step: dict):
        values_this_step = {}
        for key in log_values_this_step:
            item = self._getItemFromTitle(key)
            if getattr(item, "registration", False):
                value = self._register(log_values_this_step[key], item.true)
            elif getattr(item, "normalized_lse", False):
                value = self._normalized_lse(log_values_this_step[key], item.true)
            else:
                value = log_values_this_step[key]
            values_this_step[key] = value
            #self.dataframe.loc[step, key] = value
        self.dataframe = self.dataframe.append(values_this_step, ignore_index=True)
    #def logStepToBuffer(self, step, log_values_this_step: dict):
    #    for key in self.buffer:
    #        if key not in log_values_this_step:
    #            self.buffer[key].append(np.nan)
    #            continue
    #        item = self._getItemFromTitle(key)
    #        if getattr(item, "registration", False):
    #            value = self._register(log_values_this_step[key], item.true)
    #        else:
    #            value = log_values_this_step[key]
    #        self.buffer[key].append(value)





    def printDebugOutput(self, header=False):
        print(self.dataframe.iloc[-1:].to_string(float_format="%10.3g",
                                                 header=header))


    def finalize(self):
        columns = []
        #self.buffer = {}
        for item in self._datalog_items:
            columns.append(item.title)
            #self.buffer[item.title] = []
        #logger.info("Initializing the log outputs...")
        self.dataframe = DataFrame(columns=columns, dtype='float32')
        self.dataframe.loc[0] = np.nan

    def _checkFinalized(self):
        if not hasattr(self, "dataframe"):
            e = AttributeError("Cannot add item to the log file after starting the optimization. "
                               + "The log file remains unchanged. Only the print output is affected.")
            logger.warning(e)

    #def _saveCheckpoint(self, iteration):
    #    if not hasattr(self, "_name"):
    #        self._name = self._checkpoint_name + ".csv"
    #        self.dataframe.to_csv(self._name, sep="\t", header=True, float_format=)



