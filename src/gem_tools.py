import pandas as pd
import seaborn as sns
import itertools
import matplotlib.patches as mpatches
import warnings


def read_active_reactions_group(group='all_control', cohort='MSBB'):
    bname = '../../resources/tunahan/Busra-2023-02-05/active_reactions/'
    suffix = '.xlsx'
    fpath = bname + cohort + '/' + group + suffix
    df = pd.read_excel(fpath, 'Sheet1', index_col='rxn_ID', dtype=bool)
    return(df)

def read_active_reactions(groupdict={'m-control': ('all_control', 'MSBB'),
                                     'm-AD-B2': ('SubtypeB2_AD', 'MSBB')}):
    ar = {k: read_active_reactions_group(*v) for k, v in groupdict.items()}
    return(ar)

def read_gem_excel(index_col='ID', usecols=['ID', 'SUBSYSTEM'],
                   fpath='/Users/jonesa7/CTNS/resources/human-GEM/v1.11.0/model/Human-GEM.xlsx'):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        gem = pd.read_excel(fpath, usecols=usecols, index_col=index_col, engine='openpyxl')
    return(gem)

def extract_subsystem(subsystems, ar):
    gemsubsys = read_gem_excel()['SUBSYSTEM']
    l = [ar[k] if subsystems is None else ar[k].loc[gemsubsys.loc[gemsubsys.isin(subsystems)].index] for k in ar.keys()]
    df = pd.concat(l, axis=1)
    return(df)


def ar_clustermap(subsystems, ar, col_cluster=False):
    '''
    Active reactions cluster map
    '''
    df = extract_subsystem(subsystems, ar=ar)
    coldict = {g: 'C' + str(i) for i, g in enumerate(ar.keys())}
    col_colors = list(itertools.chain(*[[v] * ar[k].shape[1] for k, v in coldict.items()]))
    cmap = ['white', 'gray']
    g = sns.clustermap(df, row_cluster=False, col_cluster=col_cluster, col_colors=col_colors, cmap=cmap, figsize=(7,7), dendrogram_ratio=0.15)
    handles = [mpatches.Patch(color=c) for c in coldict.values()]
    g.fig.legend(handles, ar.keys(), loc='lower center', bbox_to_anchor=(0.3, 0.9 + 0.1 * col_cluster, 0.5, 0.1), ncol=2)
    g.ax_heatmap.set_xlabel(str(df.shape[1]) + ' samples')
    g.ax_heatmap.set_xticklabels('')
    g.ax_heatmap.set_xticks(range(df.shape[1]))
    if subsystems is not None:
        subsys_str = ', '.join(subsystems)
        suffix = ' in ' + subsys_str if len(subsys_str) <= 50 else ''
    else:
        suffix = ''
    g.ax_heatmap.set_ylabel(str(df.shape[0]) + ' reactions' + suffix)
    g.ax_heatmap.set_yticklabels('')
    g.ax_heatmap.set_yticks(range(df.shape[0]) if df.shape[0] <= 100 else [])
    g.ax_cbar.set_position((0, 0.4, 0.05, 0.2))
    g.ax_cbar.set_title('reaction state')
    g.ax_cbar.set_yticks([0.25, 0.75])
    g.ax_cbar.set_yticklabels(['inactive', 'active'])
    g.ax_cbar.set_ylim([0, 1])
    for spine in g.ax_heatmap.spines.values():
        spine.set_visible(True)
    for spine in g.ax_cbar.spines.values():
        spine.set_visible(True)
    return(g)
