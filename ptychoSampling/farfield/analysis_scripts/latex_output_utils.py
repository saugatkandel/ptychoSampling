import pandas as pd
import ptychoSampling.farfield.analysis_scripts.analysis_utils as anut
import os
import numpy as np

def writeConvergenceValsToCSV(dataset,
                              filename,
                              recons_type,
                              reset_file=False,
                              verbose=False,
                              convergence_key='obj_error',
                              convergence_epsilon=1e-3,
                              convergence_window_size=100):
    if reset_file:
        if os.path.isfile(filename):
            print(f'Removing {filename}.')
            os.remove(filename)
        else:
            print(f'File {filename} not found.')

    for key in dataset.keys():
        print(key)

        epr = 5 if key[1] == 'Adam' else 1
        if key[0] == 256:
            ipe = 4
            shift = 1
        elif key[0] == 1:
            ipe = 1024
            shift = 1
        else:
            ipe = 1
            shift = 0

        method_name = key[1]
        if key[0] == 256:
            method_name += '-M'

        anut.getConvergenceIndex(dataset[key], epr, ipe, shift,
                                 recons_type=recons_type,
                                 method_name=method_name,
                                 append_to_csv_filename=filename,
                                 verbose=verbose,
                                 epsilon=convergence_epsilon,
                                 key=convergence_key,
                                 window_size_epochs=convergence_window_size)



def combine_cg_ls_iters(x):

    it_o = '-'
    it_p = '-'
    if 'NCG' in x.Algorithm:
        it_o = str(int(np.round(x.obj_ls_iters, 0)))
        it_p = str(int(np.round(x.probe_ls_iters, 0))) if not np.isnan(x.probe_ls_iters) else '-'
    elif 'LM' in x.Algorithm:
        #print(x)
        if not np.isnan(x.obj_cg_iters):
            n1 = int(np.round(x.obj_cg_iters, 0))
            n2 = int(np.round(x.obj_proj_iters, 0))
            n3 = int(np.round(x.obj_ls_iters, 0))
            it_o = f'{n1}/{n2}/{n3}'
        else:
            it_o = '-'

        if not np.isnan(x.probe_cg_iters):
            n1 = int(np.round(x.probe_cg_iters, 0))
            n2 = int(np.round(x.probe_proj_iters, 0))
            n3 = int(np.round(x.probe_ls_iters, 0))
            it_p = f'{n1}/{n2}/{n3}'
        else:
            it_p = '-'
    elif 'ADMM' in x.Algorithm:
        n1 = int(np.round(x.admm_inner_iters))
        n2 = int(np.round(x.ls_iters))
        #it_o = f'{n1}/{n2}'
        it_o = f'{n2}'
        it_p = '-'
    else:
        it_o = '-'
        it_p = '-'

    flops = x.flops / 1e11
    x['it_o'] = it_o
    if x.Problem == 'BPR':
        x['it_p'] = it_p
    x['flops2'] = flops
    return x

def filterCSVAndGetLatexOutput(filename, output_file_buffer=None):
    df = pd.read_csv(filename)
    #print(df)
    if 'obj_cg_iters' in df:
        df.obj_cg_iters = df.obj_cg_iters.fillna(df.cg_iters)
    if 'obj_proj_iters' in df:
        df.obj_proj_iters = df.obj_proj_iters.fillna(df.proj_iters)
    df.obj_ls_iters = df.obj_ls_iters.fillna(df.ls_iters)
    # df.epoch = df.epoch.map(lambda x: f'{x:5g}' if not np.isnan(x) else '-')
    # df.obj_error = df.obj_error.map(lambda x: f'{x:3.3f}' if not np.isnan(x) else '-')
    #print(df)
    df_spr = df.loc[df.Problem == 'SPR']
    df_bpr = df.loc[df.Problem == 'BPR']
    df_spr = df_spr.apply(combine_cg_ls_iters, axis=1)
    df_bpr = df_bpr.apply(combine_cg_ls_iters, axis=1)

    df_spr2 = df_spr[['Algorithm', 'epoch', 'it_o', 'train_loss', 'obj_error',
                      'flops2']].copy()
    df_bpr2 = df_bpr[['Algorithm', 'epoch', 'it_o', 'it_p', 'train_loss', 'obj_error',
                      'probe_error', 'flops2']].copy()
    alg_names_new = []
    for alg in df_spr2.Algorithm:
        alg_names_new.append(alg.replace('-A', ''))
    df_spr2.Algorithm = alg_names_new
    df_spr2.it_o = df_spr2.it_o.fillna('-')
    df_spr2 = df_spr2.set_index('Algorithm')
    df_spr2 = df_spr2.rename(columns={'Algorithm':'Alg.',
                                      'obj_error':'oerror',
                                      'probe_error':'perror',
                                      'train_loss':'loss',
                                      'flops2':'flops'})


    epochfn = lambda x: f'{x:4g}' if not np.isnan(x) else '-'
    oerrfn = lambda x: f'{x:3.2g}' if not np.isnan(x) else '-'
    perrfn = lambda x: f'{x:3.2g}' if not np.isnan(x) else '-'
    flopsfn = lambda x: f'{x:3.0f}' if not np.isnan(x) else '-'
    lossfn = lambda x: f'{x:3.3g}' if not np.isnan(x) else '-'

    spr_latex_out = df_spr2.to_latex(formatters={'epoch': epochfn,
                                                 'oerror': oerrfn,
                                                 'loss': lossfn,
                                                 'flops': flopsfn},
                                     buf=output_file_buffer,
                                     caption='SPR',
                                     col_space=8)

    df_bpr2.it_o = df_bpr2.it_o.fillna('-')
    df_bpr2.it_p = df_bpr2.it_p.fillna('-')
    df_bpr2 = df_bpr2.set_index('Algorithm')
    df_bpr2 = df_bpr2.rename(columns={'Algorithm':'Alg.',
                                      'obj_error':'oerror',
                                      'probe_error':'perror',
                                      'train_loss':'loss',
                                      'flops2':'flops'})

    bpr_latex_out = df_bpr2.to_latex(formatters={'epoch': epochfn,
                                                 'oerror': oerrfn,
                                                 'perror': perrfn,
                                                 'loss': lossfn,
                                                 'flops': flopsfn},
                                     buf=output_file_buffer,
                                     caption='BPR',
                                     col_space=8)
    return spr_latex_out + bpr_latex_out

    #
    # df3 = pd.merge(df_spr2, df_bpr2, on='Algorithm', how='outer')
    # df3 = df3.drop(labels=['Problem_x', 'Problem_y'], axis=1)
    # df3 = df3.set_index('Algorithm')
    # df4 = df3.reindex(['LM-A', 'LM-A-S',
    #                    'PLM-A', 'PLM-A-S',
    #                    'PLM-J', 'PLM-J-S',
    #                    'St-PLM-A', 'St-PLM-A-S', 'St-PLM-J', 'St-PLM-J-S',
    #                    'NCG', 'PNCG', 'NAG', 'ADMM', 'PHeBIE', 'ePIE'])
    #
    # df4.it_o_x = df4.it_o_x.fillna('-')
    # df4.it_o_y = df4.it_o_y.fillna('-')
    # df4.it_p = df4.it_p.fillna('-')
    #
    # epochfn = lambda x: f'{x:5g}' if not np.isnan(x) else '-'
    # oerrfn = lambda x: f'{x:3.2g}' if not np.isnan(x) else '-'
    # perrfn = lambda x: f'{x:3.2g}' if not np.isnan(x) else '-'
    # flopsfn = lambda x: f'{x:3.0f}' if not np.isnan(x) else '-'
    # lossfn = lambda x: f'{x:3.3g}' if not np.isnan(x) else '-'
    #
    # latex_out = df4.to_latex(formatters={'epoch_x': epochfn,
    #                                      'epoch_y': epochfn,
    #                                      'obj_error_x': oerrfn,
    #                                      'obj_error_y': oerrfn,
    #                                      'probe_error_x': perrfn,
    #                                      'probe_error_y': perrfn,
    #                                      'train_loss_x': lossfn,
    #                                      'train_loss_y': lossfn,
    #                                      'flops2_x': flopsfn,
    #                                      'flops2_y': flopsfn},
    #                          buf=output_filename)
    # return latex_out






