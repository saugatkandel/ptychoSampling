colors = { 'LM-A': 'red', 'PLM-A':'green', 'PLM-J': 'magenta',
           'LM-A-S': 'brown', 'PLM-A-S': 'darkblue', 'LM-A-S2': 'brown',
           'PLM-A-S2': 'darkblue', 'PLM-J-S': 'cyan',  'PLM-J-S2': 'grey',
           'St-PLM-A': 'darkblue', 'St-PLM-J': 'brown',
           'NCG':'darkblue', 'PNCG':'black', 'ADMM': 'gray',
           'PHeBIE': 'purple','NAG':'purple', 'ePIE': 'orange',
           'Adam': 'brown'}
linestyles = {'LM-A':':', 'PLM-A':'--', 'PLM-J': '-.',
              'LM-A-S':'-', 'PLM-A-S': '-', 'LM-A-S2': ':', 'PLM-A-S': '-.', 'PLM-A-S2': '-.',
              'PLM-J-S': '-.', 'PLM-J-S2': '-.',
              'St-PLM-A': '-.','St-PLM-J': ':',
              'PNCG': '-', 'NCG':':', 'ADMM':'--',
              'PHeBIE':'-.', 'NAG':'-.', 'ePIE':':', 'Adam': ':'}
markers = {'LM-A':'o', 'PLM-A': '<', 'PHeBIE':'*', 'NCG':'<', 'ADMM':'s', 'NAG':'x', 'PNCG':'*','PLM-A-S': 'o',
           'LM-A-S': 'o', 'LM-A-S2': '<', 'PLM-A-S': '*', 'PLM-A-S2': 's', 'PLM-J-S': '*', 'PLM-J-S2': 's',
           'St-PLM-A': 's','St-PLM-J': '<',
           'PLM-J':'*', 'ePIE':'^', 'Adam': '<'}


admm_beta_markers = {0.01:'o', 10**(-1.5): 'x', 0.1: 's', 10**(-0.5): '*', 1.0: '<', 10**0.5:'p', 10.0: '^'}
admm_beta_linestyles = {0.01: '-', 0.1: ':', 1.0: '-.', 10.0: '--', 10**(-1.5): '--',  10**(-0.5): '-.', 10**0.5:':'}
admm_beta_colors = {0.01: 'red', 0.1: 'green', 1.0: 'blue', 10.0: 'black',
                    10**(-1.5): 'orange',  10**(-0.5): 'magenta', 10**0.5:'cyan'}