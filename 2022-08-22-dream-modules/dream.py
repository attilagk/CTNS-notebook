import itertools
import scipy.stats
import pandas as pd
import numpy as np

def module_memberships(module, enrichment, genes):
    gsets = genes.columns[genes.columns.to_list().index('knowledge'):].to_list()
    module_members = set(enrichment.loc[module, 'Genes'])
    n = len(module_members) # module size
    M = len(genes) # number of genes in the genome
    memberships = genes.loc[module_members.intersection(genes.index), gsets]
    def helper(gset):
        is_part_of_gset = memberships[gset] == 1
        l = memberships.index[is_part_of_gset].to_list()
        l.sort()
        genelist = ' '.join(l)
        num_member_genes = sum(is_part_of_gset)
        frac_member_genes = sum(is_part_of_gset) / len(module_members)
        gset_members = set(l)
        # Preparing for Fisher's exact test.  See:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
        N = len(gset_members) # gene set size
        a = len(module_members.intersection(gset_members))
        b = len(module_members.difference(gset_members))
        table = [[a, n - b], [N - a, M - (n + N) + a]]
        fisher_exact_p = scipy.stats.fisher_exact(table)[1]
        val = (genelist, num_member_genes, frac_member_genes, fisher_exact_p,
               np.log10(fisher_exact_p))
        return(val)
    vals = itertools.chain.from_iterable([helper(gset) for gset in gsets])
    vals = list(vals)
    return(vals)

def all_module_memberships(enrichment, genes):
    gsets = genes.columns[genes.columns.to_list().index('knowledge'):].to_list()
    def helper(l):
        l.sort()
        val = ' '.join(l)
        return(val)
    enrich = enrichment.copy()
    data = [module_memberships(mod, enrich, genes) for mod in enrich.index]
    enrich['Genes'] = enrich.Genes.apply(helper)
    variables = ['genes', 'genes_num', 'genes_frac', 'FisherE_p', 'FisherE_log_p']
    columns = [gset + '_' + var for gset in gsets for var in variables]
    df = pd.DataFrame(data, index=enrich.index, columns=columns)
    val = pd.concat([enrich, df], axis=1)
    return(val)
