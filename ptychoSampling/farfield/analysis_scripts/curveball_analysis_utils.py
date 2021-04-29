import ptychoSampling.farfield.analysis_scripts.analysis_utils as anut
import matplotlib.pyplot as plt
import dill
import dataclasses as dt
import numpy as np
from copy import deepcopy

def modifyAlternDataset(filename):
    with open(filename, 'rb') as f:
        datasets = dill.load(f)
        datasets_new = []
        for d in datasets:
            if d.iterable_params['training_batch_size'] != 256:
                datasets_new.append(d)
                continue
            dataframes_new = []
            for df in d.dataframes:
                df2 = df.copy()
                df3 = df2.loc[(df2.index +1) %2 == 0].copy()
                df3.index = (df3.index - 1) // 2 + 1
                df3.epoch = df3.epoch // 2
                dataframes_new.append(df3)
                print(df2.shape, df3.shape)
            d = dt.replace(deepcopy(d), dataframes=dataframes_new)
            datasets_new.append(d)
    with open('modified_' + filename, 'wb') as f:
        dill.dump(datasets_new, f)


def combineDatasets(key_fname_dict):
    data_all = []
    for k, v in key_fname_dict.items():
        with open(v, 'rb') as f:
            datasets = dill.load(f)
        for d in datasets:
            d.iterable_params['method'] = k
            data_all.append(d)
    return data_all



def plotData(means, lows, highs, suptitle,
             df_keys_ordered,
             label_key,
             obj_error_ylim_max=0.4,
             rfactor_ylim_max=0.7,
             xlim_max=None,
             log_yscale=False,
             log_xscale=False,
             row_key=None,
             plot_until_convergence=False):

    colors = ['red', 'black', 'blue', 'green', 'orange', 'gray', 'pink']
    linestyles = [':', '--', '-.', '-']
    markers = ['o', '<', '*', 's', 'x', '>']

    lix = df_keys_ordered.index(label_key)
    if row_key is not None:
        rix = df_keys_ordered.index(row_key)
        rows_list = []
        for k in means:
            if k[rix] not in rows_list:
                rows_list.append(k[rix])
        rows = len(rows_list)
    else:
        rows = 1

    cgens = [anut.generator(colors) for r in range(rows)]
    mgens = [anut.generator(markers) for r in range(rows)]
    lgens = [anut.generator(linestyles) for r in range(rows)]

    fig, axs = plt.subplots(rows, 3, figsize=[14, 4 * rows])
    if rows == 1:
        axs = axs[None, :]

    for key, v in means.items():
        label = str(key[lix])
        if 'awf' in key:
            label = str(key[0]) + '_' + str(key[1])

        ix = rows_list.index(key[rix]) if rows > 1 else 0
        c = next(cgens[ix])
        l = next(lgens[ix])
        m = next(mgens[ix])

        x = v.epoch
        #x = v.epoch / 2 if 'alt' in key else v.epoch

        axs[ix, 0].fill_between(x, lows[key].obj_error, highs[key].obj_error,
                                color=c, alpha=0.1)
        axs[ix, 0].plot(x, v.obj_error, color=c, ls=l, marker=m,
                        label=label, markevery=200)
        axs[ix, 0].set_ylim(top=obj_error_ylim_max)

        if log_yscale:
            axs[ix, 0].set_yscale('log')
        if log_xscale:
            axs[ix, 0].set_xscale('log')

        axs[ix, 1].fill_between(x, lows[key].r_factor, highs[key].r_factor,
                                color=c, alpha=0.1)
        axs[ix, 1].plot(x, v.r_factor, color=c, ls=l, marker=m,
                        label=label, markevery=200)
        axs[ix, 1].set_ylim(top=rfactor_ylim_max)
        if log_yscale:
            axs[ix, 1].set_yscale('log')
        if log_xscale:
            axs[ix, 1].set_xscale('log')


        axs[ix, 2].fill_between(x, lows[key].flops, highs[key].flops,
                                color=c, alpha=0.1)
        axs[ix, 2].plot(x, v.flops, color=c, ls=l, marker=m,
                        label=label, markevery=200)
        axs[ix, 2].set_yscale('log')

        if xlim_max is not None:
            for i in range(3):
                axs[ix, i].set_xlim(right=xlim_max)

        axs[ix, 0].set_ylabel('Recons. Err.', fontsize=17)
        axs[ix, 1].set_ylabel(r'$R_f$', fontsize=17)
        axs[ix, 2].set_ylabel('Flops', fontsize=17)

        axs[ix, 0].set_xlabel('Epochs', fontsize=17)
        axs[ix, 1].set_xlabel('Epochs', fontsize=17)
        axs[ix, 2].set_xlabel('Epochs', fontsize=17)

    axs[0, 1].legend(loc='best')#bbox_to_anchor=(0.75, 1.2), ncol=3, labelspacing=0.3)

    if row_key is not None:
        for i, r in enumerate(rows_list):
            plt.figtext(.5, 1.0 / (i+1), f'{row_key}={r}', fontsize=15, ha='center')
            if i+1 >= rows:
                continue
            axs[i + 1, 1].legend(loc='best')#bbox_to_anchor=(0.75, 1.2), ncol=3, labelspacing=0.3)
    #plt.figtext(.5, 0.5, r'$b=256$', fontsize=15, ha='center')

    plt.suptitle(suptitle, fontsize=17, x=0.5, y=1.1, ha='center')
    plt.tight_layout(h_pad=3.0)
    plt.show()