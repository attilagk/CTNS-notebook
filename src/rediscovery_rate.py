import pandas as pd
import numpy as np

def make_drugs_df(screenpath='/Users/jonesa7/CTNS/results/proximity-runs/rmta-apoe3-apoe4/rmta-apoe3-apoe4.csv',
                  #bbbpath='/Users/jonesa7/CTNS/results/2021-12-13-chembl-drug-info/drug-info-bbb.csv',
                  indicationspath='/Users/jonesa7/CTNS/results/2021-12-13-chembl-drug-info/drug-indication.csv',
                  sort_by_z=True):
    screen = pd.read_csv(screenpath, index_col=0).rename_axis('drug_chembl_id', axis=0)
    #info_bbb = pd.read_csv(bbbpath, index_col=0).drop(['drug_name'], axis=1)
    indications = pd.read_csv(indicationspath, index_col='drug_chembl_id')
    # D000544 is Alzheimer disease
    mp4ad = indications.loc[indications.mesh_id == 'D000544', ['max_phase_for_ind']]
    mp4ad = mp4ad.rename({'max_phase_for_ind': 'max_phase_for_AD'}, axis=1)
    # merging
    drugs = screen
    #drugs = pd.merge(drugs, info_bbb, how='left', on='drug_chembl_id')
    drugs = pd.merge(drugs, mp4ad, how='left', on='drug_chembl_id')
    if sort_by_z:
        drugs = drugs.sort_values(by='z')
        drugs['rank by z'] = np.array(range(len(drugs))) + 1
    return(drugs)


def rel_rediscovery_rate(drugs, topk=100, min_max_phase_for_ind=1,
               ind_col='max_phase_for_AD', bottoml=600):
    df = drugs.copy()
    df['tested4AD'] = drugs[ind_col] >= min_max_phase_for_ind
    top = df.iloc[:topk]
    bottom = df.iloc[-bottoml:]
    top_avg, bottom_avg = \
        [df.loc[df['tested4AD'], ind_col].sum() / xk for df, xk in \
         zip([top, bottom], [topk, bottoml])]
    ratio = top_avg / bottom_avg
    return(ratio)


def rel_rediscovery_rates(drugs, step=10, min_max_phase_for_ind=1,
               ind_col='max_phase_for_AD', bottoml=600, topl=1600):
    topks = np.arange(start=step, stop=topl, step=step)
    # calculate rediscovery rate ratio
    rrrs = [rel_rediscovery_rate(drugs, topk=k, min_max_phase_for_ind=min_max_phase_for_ind,
                      ind_col=ind_col, bottoml=bottoml) for k in topks]
    df = pd.DataFrame({'top-t': topks, 'rel rediscovery rate': rrrs})
    return(df)
