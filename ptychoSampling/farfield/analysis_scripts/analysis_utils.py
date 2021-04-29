import numpy as np
import dataclasses as dt
import dill
dill._dill._reverse_typemap['ClassType'] = type

import pandas as pd
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

def generator(choices):
    i = 0
    while True:
        ix = i % len(choices)
        i = i + 1
        yield  choices[ix]


def getFlops(datasets, **iterable_params_kwargs):
    if len(iterable_params_kwargs) == 0:
        raise ValueError("Should supply one or more parameters.")
    def _checkIterableParams(dataset):
        for k, v in iterable_params_kwargs.items():
            if dataset.iterable_params[k] != v:
                return False
        return True

    for d in datasets:
        if _checkIterableParams(d):
            return d.flops_per_iter
    raise ValueError("Dataset matching the supplied parameters not found.")

def getConvergenceIndex(dataframe, epochs_per_registration, iterations_per_epoch, index_shift,
                        window_size_epochs=100, verbose=True, epsilon=1e-3, variable_epsilon=False,
                        method_name='', recons_type='', key='obj_error',
                        append_to_csv_filename=None):
    dataframe_new = dataframe.copy()
    dataframe_new['ms'] = dataframe[key][::-1].rolling(window_size_epochs // epochs_per_registration).std()[::-1]
    dataframe_new['ma'] = dataframe[key][::-1].rolling(window_size_epochs // epochs_per_registration).mean()[::-1]
    #dataframe_new = dataframe_new.dropna()
    min_indx = 0
    while True:
        ms = dataframe_new.ms.iloc[min_indx:]
        ma = dataframe_new.ma.iloc[min_indx:]
        if variable_epsilon:
            epsilon = np.maximum(ms.min(), epsilon)
        indices_all = np.where(ms <= epsilon)[0]
        if np.size(indices_all) == 0:
            min_indx = -1#dataframe_new.ms.index[-1]
            break
        index = np.where(ms <= epsilon)[0][0]
        condition = (ma.iloc[index + 1:] < ma.iloc[index] - epsilon).sum()
        min_indx += index + 1

        if condition == 0:
            break
    if min_indx == -1:
        index = dataframe.index[-1]
        index -= 1 if index_shift == 0 else 0
    else:
        index = min_indx * epochs_per_registration * iterations_per_epoch + index_shift
    if verbose:
        print('index', index, 'epsilon', epsilon)
        print(dataframe.loc[index])
    if append_to_csv_filename is not None:
        dataframe_this = pd.DataFrame(dataframe.loc[index]).T
        dataframe_this.insert(0, 'Algorithm', [method_name])
        dataframe_this.insert(1, 'Problem', [recons_type])
        try:
            data = pd.read_csv(append_to_csv_filename)
            data = data.append(dataframe_this, ignore_index=True)
        except FileNotFoundError:
            data = pd.DataFrame(dataframe_this)
        data.to_csv(append_to_csv_filename, mode='w', index=False)
    return index

def getConvergenceIndexOld(dataframe, epochs_per_registration, iterations_per_epoch, index_shift,
                        window_size_epochs=100, verbose=True, epsilon=1e-3, minimum_epochs=0,
                        method_name='', recons_type='',
                        append_to_csv_filename=None):
    dataframe = dataframe.loc[dataframe.epoch > minimum_epochs]
    rolling_stds = dataframe[::-1].rolling(window_size_epochs // epochs_per_registration).std()[::-1]
    epsilon = np.maximum(rolling_stds.obj_error.min(), epsilon)
    #expanding_stds = dataframe_dict[key][::-1].expanding().obj_error.std()[::-1]
    #epsilon = np.maximum(expanding_stds.min(), 1e-3)
    i1 = np.where(rolling_stds.obj_error <= epsilon)[0][0] * epochs_per_registration
    index = (i1 + minimum_epochs) * iterations_per_epoch + index_shift
    #index = (np.where(rolling_stds.obj_error <= epsilon)[0][0]
    #         * epochs_per_registration * iterations_per_epoch
    #         + index_shift)
    #index += minimum_epochs

    if verbose:
        print('index', index, 'epsilon', epsilon)
        print(dataframe.loc[index])
    if append_to_csv_filename is not None:
        dataframe_this = pd.DataFrame(dataframe.loc[index]).T
        dataframe_this.insert(0, 'Algorithm', [method_name])
        dataframe_this.insert(1, 'Problem', [recons_type])
        try:
            data = pd.read_csv(append_to_csv_filename)
            data = data.append(dataframe_this, ignore_index=True)
        except FileNotFoundError:
            data = pd.DataFrame(dataframe_this)
        data.to_csv(append_to_csv_filename, mode='w', index=False)
    return index


def organizeDataForProblemType(fname, reconstruct_probe,
                               df_keys_ordered=['training_batch_size']):
    with open(fname, 'rb') as f:
        data_all = dill.load(f)
    return collectDatasetsAndDataframes(data_all, reconstruct_probe, df_keys_ordered)

def collectDatasetsAndDataframes(data_all, reconstruct_probe,
                                 df_keys_ordered=['training_batch_size']):
    ds_collected = []
    for dataset in data_all:
        #if 'obj_abs_proj' in dataset.iterable_params:
        #    if not dataset.iterable_params['obj_abs_proj']:
        #        continue
        if dataset.iterable_params['reconstruct_probe'] != reconstruct_probe:
            continue
        ds_new = dt.replace(dataset, dataframes=pd.concat(dataset.dataframes).dropna())
        ds_collected.append(ds_new)

    df_collected = {}
    for d in ds_collected:
        key = tuple(d.iterable_params[k] for k in df_keys_ordered)
        ti = df_keys_ordered.index('training_batch_size')
        if 'method' in df_keys_ordered:
            mi = df_keys_ordered.index('method')
            if key[mi] == 'cb_alt' or (key[ti]==256 and key[mi]=='CB' and reconstruct_probe==True):
                df_temp = d.dataframes.loc[(d.dataframes.index - 1) % 8 == 0].copy()
                df_temp.index = (df_temp.index - 1) // 2 + 1
                df_temp.epoch = df_temp.epoch // 2
                d.dataframes = df_temp

        print(key)
        addFlopsInfoToDataframe(d.dataframes, d.flops_per_iter, d.init_params, key[1])

        if key in df_collected:
            df_collected[key] = pd.concat([df_collected[key], d.dataframes])
        else:
            df_collected[key] = d.dataframes

    df_grouped = {k: d.groupby(d.index) for k,d in df_collected.items()}
    df_means = {k: d.mean() for k, d in df_grouped.items()}
    df_lows = {k: d.min() for k,d in df_grouped.items()}
    df_highs = {k: d.max() for k, d in df_grouped.items()}

    return ds_collected, df_grouped, df_means, df_lows, df_highs


def addFlopsInfoToDataframe(dataframe, flops_per_iter, init_params, method_test=None):

    method = str(type(init_params))
    if 'JointLMAReconsInitParams' in method:
        _addFlopsToJointLMADataframe(dataframe, flops_per_iter)
    elif 'LMAReconsInitParams' in method:
        _addFlopsToLMADataframe(dataframe, flops_per_iter)
    elif 'JointCGReconsInitParams' in method:
        _addFlopsToJointCGDataframe(dataframe, flops_per_iter)
    elif 'CGReconsInitParams' in method:
        _addFlopsToCGDataframe(dataframe, flops_per_iter)
    elif 'ADMMReconsInitParams' in method:
        _addFlopsToADMMDataframe(dataframe, flops_per_iter)
    # this is temporary! reomvoe afterwards#-------------------------------
    elif 'LM-J' in method_test:
        _addFlopsToJointLMADataframe(dataframe, flops_per_iter)
    # ----------------------------------------------------------------------
    else:
        _addFlopsSimple(dataframe, flops_per_iter)

def _addFlopsToLMADataframe(dataframe, flops_per_iter):
    flops = (dataframe.index * flops_per_iter['flops_without_cg_ls']
             + dataframe.obj_cg_iters * flops_per_iter['obj_cg_flops']
             + dataframe.obj_ls_iters * flops_per_iter['obj_proj_ls_flops'])
    if 'probe_cg_iters' in dataframe:
        flops += dataframe.probe_cg_iters * flops_per_iter['probe_cg_flops']
    dataframe['flops'] = flops

def _addFlopsToJointLMADataframe(dataframe, flops_per_iter):

    flops = (dataframe.index * flops_per_iter['flops_without_cg_ls']
             + dataframe.cg_iters * flops_per_iter['obj_cg_flops']
             + dataframe.ls_iters * flops_per_iter['obj_proj_ls_flops'])
    dataframe['flops'] = flops

def _addFlopsToCGDataframe(dataframe, flops_per_iter):
    flops = (dataframe.index * flops_per_iter['flops_without_ls']
             + dataframe.obj_ls_iters * flops_per_iter['obj_ls_flops'])
    if 'probe_ls_iters' in dataframe:
        flops += dataframe.probe_ls_iters * flops_per_iter['probe_ls_flops']
    dataframe['flops'] = flops

def _addFlopsToJointCGDataframe(dataframe, flops_per_iter):
    flops = (dataframe.index * flops_per_iter['flops_without_ls']
             + dataframe.ls_iters * flops_per_iter['obj_ls_flops'])
    dataframe['flops'] = flops

def _addFlopsToADMMDataframe(dataframe, flops_per_iter):

    total_flops = flops_per_iter['total_flops']
    aux_ls_flops = flops_per_iter['aux_ls_flops']
    aux_only_flops = flops_per_iter['aux_only_flops']
    early_stop_flops = flops_per_iter['early_stop_only_flops']
    flops1 = (total_flops - aux_only_flops - early_stop_flops) * dataframe.index
    flops2 = aux_ls_flops * dataframe.ls_iters
    flops3 =  (early_stop_flops + aux_only_flops - aux_ls_flops) * dataframe.admm_inner_iters
    flops = flops1 + flops2 + flops3
    dataframe['flops'] = flops

def _addFlopsSimple(dataframe, flops_per_iter):
    flops = dataframe.index * flops_per_iter
    dataframe['flops'] = flops


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def combineDatasets(key_fname_dict, key_name='method',
                    lm_include_both_cg_tol=False, lm_cg_tol=0.1):
    data_all = []
    for k, v in key_fname_dict.items():
        print(k, v)
        with open(v, 'rb') as f:
            datasets = dill.load(f)
        for d in datasets:
            key_this = k
            #if 'min_cg_tol' in d.iterable_params:
            #
            #    if lm_include_both_cg_tol:
            #        suffix = 1 if d.iterable_params["min_cg_tol"] == lm_cg_tol else 2
            #        key_this += f'{suffix}'
            #    else:
            #        if d.iterable_params["min_cg_tol"] != lm_cg_tol:
            #            continue
            if 'apply_precond' in d.iterable_params:
                prefix = 'P' if d.iterable_params["apply_precond"] else ''
                key_this = f'{prefix}' + key_this
            if 'apply_precond_and_scaling' in d.iterable_params:
                prefix = 'P' if d.iterable_params["apply_precond_and_scaling"] else ''
                key_this = f'{prefix}' + key_this
            elif hasattr(d.init_params, 'apply_precond_and_scaling'):
                prefix = 'P' if d.init_params.apply_precond_and_scaling else ''
                key_this = f'{prefix}' + key_this

            d.iterable_params[key_name] = key_this
            data_all.append(d)
    return data_all