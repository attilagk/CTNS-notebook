#! /usr/bin/env Rscript

library(TopKLists)

args <- commandArgs(TRUE)
out.csv <- args[1]

input.csv <- "/Users/jonesa7/CTNS/results/2021-12-31-rank-aggregation/breast-input.csv"
space.csv <- "/Users/jonesa7/CTNS/results/2021-12-31-rank-aggregation/breast-space.csv"

input <- read.csv(input.csv)
space <- read.csv(space.csv)
d.w <- c(0, 1, 0, 0, 0, 1)

start <- Sys.time()
outCEMC <- CEMC(input, space, N = 2000, d.w = d.w)
runtime <- Sys.time() - start
print(runtime)

TopK <- data.frame(gene = outCEMC$TopK)
write.csv(TopK, out.csv)
