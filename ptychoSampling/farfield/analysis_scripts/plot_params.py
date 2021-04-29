plot_bpr_cg = False
plot_only_poisson_surrogate = False

excludes = {'spr': {'gaussian': ['NCG', 'St-PLM-J', 'St-PLM-A', 'St-PLM-A-S', 'St-PLM-J-S'],
                    'poisson': ['NCG', 'St-PLM-J', 'St-PLM-A', 'St-PLM-A-S', 'St-PLM-J-S']},
            'bpr': {'gaussian': ['NCG', 'St-PLM-J', 'St-PLM-A', 'St-PLM-A-S', 'St-PLM-J-S'],
                    'poisson': ['NCG', 'St-PLM-J', 'St-PLM-A', 'St-PLM-A-S', 'St-PLM-J-S']}}

suffixes = {'spr': {'gaussian': '', 'poisson': ''},
            'bpr': {'gaussian': '', 'poisson': ''}}

if not plot_bpr_cg:
    excludes['bpr']['gaussian'] += ['PNCG']
    excludes['bpr']['poisson'] += ['PNCG']
    suffixes['bpr']['gaussian'] += '_no_cg'
    suffixes['bpr']['poisson'] += '_no_cg'

if plot_only_poisson_surrogate:
    excludes['spr']['poisson'] += ['LM-A', 'PLM-A']
    excludes['bpr']['poisson'] += ['LM-A', 'PLM-A', 'PLM-J']
    suffixes['spr']['poisson'] += '_only_surrg'
    suffixes['bpr']['poisson'] += '_only_surrg'

admm_betas = {1e3: {'gaussian': 1.0, 'poisson': 1.0},
              1e4: {'gaussian': 10**(-0.5), 'poisson': 1.0},
              1e6: {'gaussian': 10**(-0.5), 'poisson': 1.0},
               0: {'gaussian': 0.1, 'poisson':0.1}}

conv_epsilons = {1e3: 3e-3,
                 1e4: 2e-3,
                 1e6: 1e-3,
                 0: 1e-8}
