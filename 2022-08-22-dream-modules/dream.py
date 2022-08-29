import itertools
import pandas as pd

def module_memberships(module, enrichment, genes):
    gsets = genes.columns[genes.columns.to_list().index('knowledge'):].to_list()
    module_members = set(enrichment.loc[module, 'Genes'])
    memberships = genes.loc[module_members.intersection(genes.index), gsets]
    def helper(gset):
        b = memberships[gset] == 1
        l = memberships.index[b].to_list()
        l.sort()
        genelist = ' '.join(l)
        num_member_genes = sum(b)
        frac_member_genes = sum(b) / len(module_members)
        val = (genelist, num_member_genes, frac_member_genes)
        return(val)
    genelists = [' '.join(memberships.index[memberships[gset] == 1].to_list()) for gset in gsets]
    frac_member_genes = [sum(memberships[gset] == 1) / len(module_members) for gset in gsets]
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
    variables = ['genes', 'genes_num', 'genes_frac']
    columns = [gset + '_' + var for gset in gsets for var in variables]
    df = pd.DataFrame(data, index=enrich.index, columns=columns)
    val = pd.concat([enrich, df], axis=1)
    return(val)
