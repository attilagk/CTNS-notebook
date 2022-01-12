#! /usr/bin/env Rscript

library(TopKLists)

args <- commandArgs(TRUE)
out.csv <- args[1]

input.csv <- "/Users/jonesa7/CTNS/results/2021-12-31-rank-aggregation/chembl-input.csv"
space.csv <- "/Users/jonesa7/CTNS/results/2021-12-31-rank-aggregation/chembl-space.csv"
d.w <- c(2, # knowledge
				 2, # TWAS
				 2, # agora2
				 0, # agora
				 0, # DESudhir
				 3, # ADDE.APOE3.APOE3
				 4, # ADDE.APOE4.APOE4 
				 5, # APOE34.DE.neuron
				 5, # APOE34.DE.astrocyte
				 5) # APOE34.DE.microglia

#input.csv <- "/Users/jonesa7/CTNS/results/2021-12-31-rank-aggregation/breast-input.csv"
#space.csv <- "/Users/jonesa7/CTNS/results/2021-12-31-rank-aggregation/breast-space.csv"
#d.w <- c(0, 1, 0, 0, 0, 1)

input <- read.csv(input.csv)
space <- read.csv(space.csv)
d.w <- d.w / sum(d.w)

start <- Sys.time()
outCEMC <- CEMC(input, space, N = 2000, d.w = d.w)
runtime <- Sys.time() - start
print(runtime)

TopK <- data.frame(gene = outCEMC$TopK)
write.csv(TopK, out.csv)
