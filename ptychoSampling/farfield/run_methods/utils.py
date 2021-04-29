import dill
import numpy as np
from copy import deepcopy
from ptychoSampling.utils.utils import getRandomComplexArray
from ptychoSampling.farfield.run_methods.params import *
from ptychoSampling.farfield.run_methods.sim_methods import Simulation
from ptychoSampling.probe import EllipticalProbe
import dataclasses as dt
import pandas as pd
from typing import List
import tensorflow as tf


def getSimulationAndGuesses(default_path_str: bool, reload_simulation: bool = True, probe_type="airy"):
    items = ["obj", "probe", "scan_grid", "intensities"]
    sim_items = {}
    if reload_simulation:
        try:
            for name in items:
                name_this = name
                if default_path_str is not None:
                    name_this = default_path_str + name_this
                with open(f"{name_this}.pkl", 'rb') as f:
                    value = dill.load(f)
                    sim_items[name] = value
            print("Loaded successfully")
        except:
            sim = Simulation()
            for name in items:
                value = sim.__dict__[name]
                name_this = name
                if default_path_str is not None:
                    name_this = default_path_str + name_this
                with open(f'{name_this}.pkl', 'wb') as f:
                    dill.dump(value, f, byref=False)
                sim_items[name] = value
            print("Regenerated data.")
    else:
        sim = Simulation()
        for name in items:
            value = sim.__dict__[name]
            sim_items[name] = value

    obj_guess = deepcopy(sim_items["obj"])
    obj_guess.array = getRandomComplexArray(shape=obj_guess.array.shape)
    sim_items["default_obj_guess"] = obj_guess

    ptemp = sim_items["probe"]
    if probe_type == 'airy':
        sim_items["default_probe_guess"] = EllipticalProbe(wavelength=ptemp.wavelength,
                                                           pixel_size=ptemp.pixel_size,
                                                           n_photons=ptemp.n_photons,
                                                           width_dist=ptemp.width_dist,
                                                           shape=ptemp.shape)
    elif probe_type == 'gaussian':
        sim_items["probe"]._calculateGaussianFit()
        sim_items["default_probe_guess"] = EllipticalProbe(wavelength=sim_items['probe'].wavelength,
                                                           pixel_size=sim_items['probe'].pixel_size,
                                                           n_photons=sim_items['probe'].n_photons,
                                                           width_dist=sim_items["probe"]._gaussian_fwhm * 2,
                                                           shape=sim_items['probe'].shape)
    else:
        raise ValueError("Probe type not identified.")
    return sim_items


def reload(loss_type, metaparams):
    # fname_prefix = fname_prefix + "_" + loss_type
    fname = metaparams.fname_prefix + "_" + loss_type + "_data.pkl"
    with open(fname, "rb") as f:
        data = dill.load(f)
    return data


def runSingleReconstruction(recons_method: ReconstructionT, loss_type: str, init_params: dict, run_params: dict):
    r = recons_method(**init_params)
    r.run(**run_params)
    return r

    max_iterations = metaparams.max_epochs * r._iterations_per_epoch
    r.run(max_iterations=max_iterations, **dt.asdict(metaparams.recons_run_params))
    print(r.datalog.dataframe.dropna().tail(1))


def getSingleRunParamsFromIterables(iterables: dict):
    from itertools import product

    kv_pairs = []
    for it, vlist in iterables.items():
        kv_pair_this = [(it, v) for v in vlist]
        kv_pairs.append(kv_pair_this)

    iter_params_list = []
    for t in product(*kv_pairs):
        if ('reconstruct_probe', False) not in t:
            iter_params_list.append(dict(t))
            continue
        # This part is specifically for the Adam case
        dict_wo_probe_lr = {}
        for tup in t:
            if 'learning_rate_probe' in tup:
                continue
            dict_wo_probe_lr.update([tup])
        if dict_wo_probe_lr not in iter_params_list:
            iter_params_list.append(dict_wo_probe_lr)
    return iter_params_list


@dt.dataclass
class ReconsDataNew:
    iterable_params: dict
    init_params: dict
    run_params: dict
    flops_per_iter: int
    dataframes: List[pd.DataFrame] = dt.field(default_factory=list)
    final_objs: List[np.ndarray] = dt.field(default_factory=list)
    final_probes: List[np.ndarray] = dt.field(default_factory=list)


def runMultipleReconstructionsNew(loss_type: str,
                                  sim_data: dict,
                                  metaparams: MetaParams,
                                  base_path: str) -> dict:
    data_all = []

    iterables = metaparams.iterables
    recons_init_params = dt.asdict(metaparams.recons_init_params)
    if iterables is None:
        iterables = {}
    iterables.setdefault("reconstruct_probe", [False])
    iterables.setdefault("training_batch_size", [0])

    iter_params_list = getSingleRunParamsFromIterables(iterables)
    for par in iter_params_list:
        print('Params for this iteration', par)
        if par['reconstruct_probe']:
            probe_guess = sim_data['default_probe_guess']
            if 'obj_abs_proj' not in recons_init_params:
                par['obj_abs_proj'] = True
        else:
            probe_guess = sim_data['probe']
            if 'obj_abs_proj' not in recons_init_params:
                par['obj_abs_proj'] = False

        obj_guess = deepcopy(sim_data['default_obj_guess'])

        init_params = {'obj': obj_guess,
                       'probe': probe_guess,
                       'loss_type': loss_type,
                       'grid': sim_data["scan_grid"],
                       'intensities': sim_data["intensities"],
                       'obj_array_true': sim_data["obj"].array,
                       'probe_wavefront_true': sim_data["probe"].wavefront}
        # this order of operations is important ----------------------------------
        for k in par:
            if k in recons_init_params:
                if recons_init_params[k] is not None:
                    print(f"Overwriting parameter {k} in recons_init_params " +
                          "with the supplied value from iterables.")
                    del recons_init_params[k]
        init_params.update(par)
        init_params.update(recons_init_params)
        # --------------------------------------------------------------------------
        print('Calculating flops per iter...')
        flops_per_iter = getReconsFlopsPerIterNew(metaparams.recons_method, init_params)

        iterations_per_epoch = 1
        if par["training_batch_size"] != 0:
            iterations_per_epoch = sim_data["intensities"].shape[0] // par["training_batch_size"]

        if metaparams.update_delay_probe_epochs is not None:
            if init_params["update_delay_probe"] is not None:
                raise KeyError("Cannot supply update_delay_probe in both metaparams and init_params.")
            init_params["update_delay_probe"] = metaparams.update_delay_probe_epochs * iterations_per_epoch

        if metaparams.loss_n_spline_epochs is not None:
            if 'n_spline_epochs' in init_params['loss_init_extra_kwargs']:
                raise KeyError("Cannot supply n_spline_epochs in both metaparams and init_params.")
            init_params['loss_init_extra_kwargs']['n_spline_epochs'] = (metaparams.loss_n_spline_epochs
                                                                            * iterations_per_epoch)

        run_params = dt.asdict(metaparams.recons_run_params)
        run_params["max_iterations"] = metaparams.max_epochs * iterations_per_epoch

        print('Calculated flops per iter. Main reconstruction simulation...')
        # Only storing the parameters supplied through metaparams. The other parameters can be calculated from there.
        data = ReconsDataNew(iterable_params=par,
                             init_params=metaparams.recons_init_params,
                             run_params=metaparams.recons_run_params,
                             flops_per_iter=flops_per_iter)

        for j in range(metaparams.n_recons):
            print(f'Starting run {j} with params', par)
            print('loss_init_extra_kwargs', init_params['loss_init_extra_kwargs'])
            init_params["obj"].array = getRandomComplexArray(shape=obj_guess.array.shape)
            r = metaparams.recons_method(**init_params)
            r.run(**run_params)

            print(r.datalog.dataframe.dropna().tail(1))
            data.dataframes.append(r.datalog.dataframe)
            data.final_objs.append(r.obj.array)
            data.final_probes.append(r.probe.wavefront)

        data_all.append(data)

    fname = f'{base_path}/{metaparams.fname_prefix}_{loss_type}_data.pkl'
    with open(fname, "wb") as f:
        dill.dump(data_all, f)
    return data_all


def getReconsFlopsPerIterNew(recons_method: ReconstructionT, init_params: dict) -> float:
    from ptychoSampling.reconstruction.utils.utils import getComputationalCostInFlops
    #init_params_this = deepcopy(init_params)

    #--------------------------------------------------------------------------------------
    # this is a workaround because of some issue with deepcopy.
    # I am copying the initial value, then putting it back afterwards
    r_fact_log_init_val = init_params["r_factor_log"]
    init_params_this = init_params
    init_params_this["r_factor_log"] = False
    #---------------------------------------------------------------------------------------
    r = recons_method(**init_params_this)
    if hasattr(r, "getFlopsPerIter"):
        flops_per_iter = r.getFlopsPerIter()
    else:
        with r.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            r.session = tf.Session(config=config)
            r.session.run(tf.global_variables_initializer())

        for opt in r.optimizers:
            r.session.run(opt.minimize_op)
        # r.run(max_iterations=1, debug_output=False)
        flops_per_iter = getComputationalCostInFlops(r.graph)

    #-----------------------------------------------------------------------------------------
    init_params_this["r_factor_log"] = r_fact_log_init_val
    #------------------------------------------------------------------------------------------
    return flops_per_iter


def filterMetrics(data_mean_std: dict,
                  epoch_max: int = 500,
                  oerr_max: float = 0.2,
                  iters_per_minibatch_epoch: int = 5) -> dict:
    data_filtered = {}
    for k in data_mean_std:
        data_filtered[k] = {}
    for b, iters_per_lr in data_mean_std["iters"].items():
        for k, val in data_filtered.items():
            val[b] = {}
        iter_max = epoch_max if b == 0 else epoch_max * iters_per_minibatch_epoch
        for lr, (mean, std) in iters_per_lr.items():
            if mean > iter_max or data_mean_std["oerr"][b][lr][0] > oerr_max:
                # print(b, lr, mean, oerr_mean_std[b][lr])
                continue
            for k in ["vmin", "oerr", "iters", "flops"]:
                data_filtered[k][b][lr] = data_mean_std[k][b][lr]

            lp, lo = lr
            if lp > 0:
                data_filtered["perr"][b][lr] = data_mean_std["perr"][b][lr]

    return data_filtered

