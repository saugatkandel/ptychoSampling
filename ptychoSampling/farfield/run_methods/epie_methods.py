from copy import deepcopy
from skimage.feature import register_translation
from pandas import DataFrame
import numpy as np
from tqdm import tqdm_notebook as tqdm

class ePiePhaseRetrieverCPU:
    def __init__(self,
                 intensities,
                 scan_grid,
                 probe_guess,
                 obj_guess,
                 obj_true=None,
                 probe_true=None,
                 reconstruct_probe=True,
                 r_factor_log=True,
                 registration_log_frequency=1,
                 obj_abs_proj=True,
                 update_delay_probe=0):

        self.intensities = intensities.astype('float32')
        self.amplitudes = self.intensities ** 0.5
        self.n_i, self.dy, self.dx = intensities.shape
        self.scan_grid = scan_grid

        self.probe = deepcopy(probe_guess)
        self.obj = deepcopy(obj_guess)

        self._obj_bordered_array = self.obj.bordered_array.copy()

        self.registration_log_frequency = registration_log_frequency
        self.reconstruct_probe = reconstruct_probe
        self.obj_abs_proj = obj_abs_proj
        self.r_factor_log = r_factor_log

        self.update_delay_probe = update_delay_probe
        self.iteration = 0

        self._datalog_columns = ["epoch", "train_loss"]
        self.datalog_fns = {}

        if obj_true is not None:
            self.obj_true = obj_true
            self._datalog_columns.append("obj_error")
            self.datalog_fns["obj_error"] = self.getObjRegistrationError

        if probe_true is not None:
            self.probe_true = probe_true
            if reconstruct_probe:
                self._datalog_columns.append("probe_error")
                self.datalog_fns["probe_error"] = self.getProbeRegistrationError

        if self.r_factor_log:
            self._datalog_columns.append("r_factor")
            self.datalog_fns["r_factor"] = self.getRFactor

        self.datalog = DataFrame(columns=self._datalog_columns, dtype='float32')
        self._datalog_per_epoch = []
        self.datalog.loc[0] = np.nan

    @staticmethod
    def registration(true, test):
        shift, err, phase = register_translation(test, true, upsample_factor=10)
        shift, err, phase = register_translation(test * np.exp(-1j * phase), true, upsample_factor=10)
        return err

    @property
    def epoch(self):
        return self.iteration // self.n_i

    def getRFactor(self):
        amplitudes_pred = []
        for i, (py, px) in enumerate(self.scan_grid.positions_pix):
            obj_slice = self._obj_bordered_array[py: py + self.dy, px: px + self.dx]
            wv_out = self.probe.wavefront * np.fft.fftshift(obj_slice)
            wv_det = wv_out.fft2()
            amplitudes_pred.append(np.abs(wv_det))

        num = np.sum(np.abs(amplitudes_pred - self.amplitudes))
        denom = np.sum(self.amplitudes)
        return num / denom

    def getObjRegistrationError(self):
        return self.registration(self.obj_true.array, self.obj.array)

    def getProbeRegistrationError(self):
        return self.registration(self.probe_true.wavefront, self.probe.wavefront)

    def run(self, n_epochs, debug_output=False, debug_output_epoch_frequency=1):
        (s1, s2), (s3, s4) = self.obj.border_shape
        mask = np.ones(self._obj_bordered_array.shape, dtype='bool')
        mask[s1:-s2, s3:-s4] = False
        header = True
        for n in tqdm(range(n_epochs)):
            shuffled_order = np.random.permutation(range(self.n_i))

            self._datalog_per_epoch = []
            for i in shuffled_order:
                py, px = self.scan_grid.positions_pix[i]

                probe = self.probe.wavefront
                obj_slice = self._obj_bordered_array[py: py + self.dy, px: px + self.dx]
                wv_out = probe * np.fft.fftshift(obj_slice)
                wv_det = wv_out.fft2()

                wv_det_corrected = wv_det * self.amplitudes[i] / (np.abs(wv_det) + 1e-30)
                wv_out_corrected = wv_det_corrected.ifft2()
                wv_diff = (wv_out - wv_out_corrected)

                lr_obj = 1 / np.max(np.abs(probe))**2
                lr_probe = 1 / np.max(np.abs(obj_slice))**2

                obj_slice_new = obj_slice - lr_obj * np.fft.ifftshift(np.conjugate(probe) * wv_diff)
                if self.obj_abs_proj:
                    obj_amps = np.abs(obj_slice_new)
                    obj_slice_new = obj_slice_new * np.clip(obj_amps, 0, 1.0) / (obj_amps + 1e-30)

                self._obj_bordered_array[py:py + self.dy, px: px + self.dx] = obj_slice_new
                self._obj_bordered_array[mask] = self.obj.border_const


                if self.reconstruct_probe and self.iteration >= self.update_delay_probe:
                    probe -= lr_probe * np.fft.fftshift(np.conjugate(obj_slice)) * wv_diff

                datalog_list_this = [self.epoch]
                train_loss = (0.5 * np.sum((np.abs(wv_det)  - self.amplitudes[i]) ** 2))
                datalog_list_this.append(train_loss)

                if self.iteration % (self.registration_log_frequency * self.n_i) == 0:
                    for key, fn in self.datalog_fns.items():
                        datalog_list_this.append(fn())
                else:
                    for key in self.datalog_fns:
                        datalog_list_this.append(np.nan)

                if (self.iteration % (debug_output_epoch_frequency * self.n_i)  == 0) and debug_output:
                    #print(self.datalog.iloc[-1:].to_string(float_format="%10.3g", header=header))
                    if header: print(self._datalog_columns)
                    arr = np.array(datalog_list_this, dtype='float32')
                    print(np.array2string(arr, suppress_small = True, formatter={'float_kind':lambda x: "%8.3g" % x}))
                    header = False

                self.iteration += 1
                self._datalog_per_epoch.append(datalog_list_this)

            df_temp = DataFrame(self._datalog_per_epoch, columns=self._datalog_columns, dtype='float32')
            self.datalog = self.datalog.append(df_temp, ignore_index=True)
            self.obj.array.flat = self._obj_bordered_array[~mask]

    def _updateOutputs(self):
        pass