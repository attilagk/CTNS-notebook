
def module_memberships(module, enrichment, genes):
    gsets = genes.columns[genes.columns.to_list().index('knowledge'):].to_list()
    module_members = set(enrichment.loc[module, 'Genes'])
    memberships = genes.loc[module_members.intersection(genes.index), gsets]
    # TODO: alphabetically sort gsets
    genelists = [' '.join(memberships.index[memberships[gset] == 1].to_list()) for gset in gsets]
    return(genelists)
