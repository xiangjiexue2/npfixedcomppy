# R-side wall-clock for the binned ("...w") family bench (counterpart of
# bench_binned.py). Reads the same data files, times each R entry point
# (min of `rep` runs), prints seconds.
suppressWarnings(library(npfixedcomp2))

v  <- as.numeric(read.csv("npfc_data_5000.csv")[[1]])
v2 <- as.numeric(read.csv("npfc_bench_data_20000.csv")[[1]])

timeit <- function(name, expr, rep = 3) {
  # substitute before timing: a raw promise is forced once and its value is
  # cached for the later reps, so only rep 1 would actually run.
  expr <- substitute(expr)
  ts <- replicate(rep, system.time(eval(expr))[3])
  cat(sprintf("%-24s min=%9.3f s  (rep=%d)\n", name, min(ts), rep))
}

cat("--- n=5000 ---\n")
timeit("npnormllw n=5000",  computemixdist(v, method = "npnormllw"))
timeit("npnormcvmw n=5000", computemixdist(v, method = "npnormcvmw"))
timeit("npnormadw  n=5000", computemixdist(v, method = "npnormadw"))
timeit("nptllw(b=inf) n=5000", computemixdist(v, method = "nptllw"))
timeit("nptllw(b=5)  n=5000", computemixdist(v, method = "nptllw", beta = 5), rep = 1)
timeit("estpi0 llw  n=5000", estpi0(v, method = "npnormllw", val = 2))
cat("--- n=20000 ---\n")
timeit("npnormllw n=20000",  computemixdist(v2, method = "npnormllw"))
timeit("nptllw(b=inf) n=20000", computemixdist(v2, method = "nptllw"))
