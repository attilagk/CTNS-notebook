#!/usr/bin/env bash

cd /Users/jonesa7/CTNS/results/2022-08-11-pascal
PVALS=/Users/jonesa7/CTNS/resources/GWAS/Bellenguez-2022/GCST90027158/harmonised/35379992-GCST90027158-MONDO_0004975-Build38-corrected.f.tsv.pvals
~/tools/PASCAL/Pascal --runpathway=on --pval=$PVALS
