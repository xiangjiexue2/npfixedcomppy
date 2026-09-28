# Emit gold (full precision) for the DETERMINISTIC CVM/AD/CLL cases, one line
# per case, for verify_cvmadcll.py to read. CLL mix is deterministic in R with
# npt=92 (element-wise comparable); EP_CLL is deterministic with npt=87 but the
# Python fit lands on a neighbouring 88-pt point in the flat region, so only its
# ll is golded (the test gates EP_CLL on ll + invariants, not pt/pr).
library(npfixedcomp2)
v <- read.csv("C:/Users/xxjie/Documents/rebuild/npfc_data_1000.csv")[, 1]
n <- length(v)
out <- file("tests/gold_cvmadcll.txt", open = "wt")
cll <- computemixdist(tanh(v), method = "npnormcll", beta = n)
cat("CLL", sprintf("%.17g", cll$ll), length(cll$mix$pt), as.character(cll$iter), "\n", file = out)
cat("CLL_pt", paste(sprintf("%.17g", cll$mix$pt), collapse = " "), "\n", file = out)
cat("CLL_pr", paste(sprintf("%.17g", cll$mix$pr), collapse = " "), "\n", file = out)
epc <- estpi0(tanh(v), method = "npnormcll", val = 2, beta = n)
cat("EP_CLL", sprintf("%.17g", epc$ll), length(epc$mix$pt), as.character(epc$iter), "\n", file = out)
close(out)
cat("gold written\n")
