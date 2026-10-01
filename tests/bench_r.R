# R wall-time on the hot cases, to benchmark npfixedcomppy against.
# Run from tests/ (the data files live here).
library(npfixedcomp2)
v1 <- read.csv("npfc_data_1000.csv")[, 1]
v5 <- read.csv("npfc_data_5000.csv")[, 1]
pois <- read.csv("npfc_data_pois.csv")[, 1]
timeit <- function(tag, expr) {
  t0 <- Sys.time()
  r <- expr
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("%-20s %.3f s  (ll=%.6g iter=%d npt=%d)\n", tag, dt, r$ll, r$iter, length(r$mix$pt)))
  invisible(dt)
}
timeit("npnormll n=1000", computemixdist(v1, method = "npnormll"))
timeit("nptll(b=inf) n=1000", computemixdist(v1, method = "nptll"))
timeit("nptll(b=5) n=5000", computemixdist(v5, method = "nptll", beta = 5))
timeit("npnormcll n=1000", computemixdist(tanh(v1), method = "npnormcll", beta = length(v1)))
timeit("nppoisll n=1000", computemixdist(pois, method = "nppoisll"))
timeit("npnormad n=1000", computemixdist(v1, method = "npnormad"))
timeit("estpi0 norm n=1000", estpi0(v1, method = "npnormll", val = 2))
