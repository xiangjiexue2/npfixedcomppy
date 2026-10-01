# Cross-ll check: evaluate R's own binned non-central-t density at the
# Python-returned support points and recover the log-likelihood. If R's
# dnpdisct reproduces R's recorded ll at Python's (slightly different)
# pt/pr, then the pt/pr offset is just valley flatness -- not a gradient or
# weight bug in the Python NpTLLW port.
suppressWarnings(library(npfixedcomp2))
data <- as.numeric(read.csv("npfc_data_1000.csv")[[1]])
b <- bin(data, -3)   # the R pre-binning the wrapper does

r_llw <- computemixdist(data, method = "nptllw")
r_t5  <- computemixdist(data, method = "nptllw", beta = 5)

cat(sprintf("R nptllw  ll=%.17g\n", r_llw$ll))
cat(sprintf("R nptllw5 ll=%.17g\n", r_t5$ll))

crossll <- function(pt, pr, df) {
  d <- dnpdisct(b$v, pt, pr, df, 1e-3)
  sum(b$w * log(d)) * -1
}

# Python's returned pt/pr (dumped by dump_binned_py_pt.py; Py is
# deterministic, so these are stable).
pt_inf <- c(0, 0.14738764054147382, 1.9719683021802077, 2.9971254285363793)
pr_inf <- c(0, 0.53044356287877981, 0.45782867093958868, 0.011727766181631521)
pt_5   <- c(0, 0.13369570766345237, 1.4718256353919519)
pr_5   <- c(0, 0.43039690130753749, 0.5696030986924624)

ll_cross_inf <- crossll(pt_inf, pr_inf, Inf)
ll_cross_5   <- crossll(pt_5, pr_5, 5)
cat(sprintf("crossll(inf) = %.17g  (R recorded %.17g)  d=%.3e\n",
            ll_cross_inf, r_llw$ll, abs(ll_cross_inf - r_llw$ll)))
cat(sprintf("crossll(5)   = %.17g  (R recorded %.17g)  d=%.3e\n",
            ll_cross_5, r_t5$ll, abs(ll_cross_5 - r_t5$ll)))
