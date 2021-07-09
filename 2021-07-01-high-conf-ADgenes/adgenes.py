import pandas as pd

def correct_symbols(df, alt_symbols):
    '''
    Correct gene symbols in df using alt_symbols dict
    '''
    val = df.copy()
    for sym in alt_symbols.keys():
        val.loc[df.Gene == sym, 'Gene'] = alt_symbols[sym]
    return(val)
