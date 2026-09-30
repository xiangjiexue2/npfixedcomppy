# Gold references for npfixedcomppy::posteriormean vs R npfixedcomp2.
#
# The posterior mean is a PURE function of (x, pt, pr, beta, family). The
# npnormll FIT itself is non-reproducible in R (OpenMP parallel reductions on
# a flat NPMLE surface shift the trajectory between runs), so gating the
# posterior mean against an independently re-run fit is unstable. We therefore
# record, per case, the EXACT fit params R used (family, beta, pt, pr) and the
# posterior means R computed from them. The Python test then feeds those exact
# params back into its own posteriormean and checks they agree to working
# precision — validating the posterior-mean code path in isolation. A separate
# end-to-end refit check covers the deterministic families (npt beta=Inf,
# nppois), whose fits ARE reproducible in R.
#
# Writes:
#   npfc_pm_fit.csv : tag, family, beta, k, pt, pr  (pt/pr semicolon-joined %.17g)
#   npfc_pm_ref.csv : tag, idx, idval, tanval       (%.17g)
#
# Cases (one fit per family):
#   npnorm  : computemixdist(d, "npnormll"); estpi0(d, "npnormll", val=2)
#   npt     : computemixdist(d, "nptll")           (beta = Inf, deterministic)
#   npnormc : computemixdist(tanh(d), "npnormcll", beta=100); x = tanh(d)
#   nppois  : computemixdist(p, "nppoisll")        (deterministic)
library(npfixedcomp2)

d <- as.numeric(read.csv("npfc_data_1000.csv")[[1]])
p <- as.numeric(read.csv("npfc_data_pois.csv")[[1]])

fits <- list(
  list(tag = "npnorm:cm",  x = d,        r = computemixdist(d, method = "npnormll")),
  list(tag = "npnorm:ep",  x = d,        r = estpi0(d, method = "npnormll", val = 2)),
  list(tag = "npt:cm",     x = d,        r = computemixdist(d, method = "nptll")),
  list(tag = "npnormc:cm", x = tanh(d),  r = computemixdist(tanh(d), method = "npnormcll", beta = 100)),
  list(tag = "nppois:cm",  x = p,        r = computemixdist(p, method = "nppoisll"))
)

fit_con <- file("npfc_pm_fit.csv", "w")
cat("tag,family,beta,k,pt,pr\n", file = fit_con)
ref_con <- file("npfc_pm_ref.csv", "w")
cat("tag,idx,idval,tanval\n", file = ref_con)

for (fc in fits) {
  tag <- fc$tag; x <- fc$x; r <- fc$r
  pt <- r$mix$pt; pr <- r$mix$pr; beta <- r$beta; fam <- r$family
  pt_str <- paste(sprintf("%.17g", pt), collapse = ";")
  pr_str <- paste(sprintf("%.17g", pr), collapse = ";")
  cat(tag, fam, sprintf("%.17g", beta), length(pt), pt_str, pr_str, sep = ",", "\n", file = fit_con)

  id  <- posteriormean(x, r)
  tan <- posteriormean(x, r, fun = tanh)
  for (i in seq_along(x)) {
    cat(tag, i - 1, sprintf("%.17g", id[i]), sprintf("%.17g", tan[i]), sep = ",", "\n", file = ref_con)
  }
  cat(sprintf("  %-12s fam=%-8s beta=%-10g k=%d ll=%.6g\n", tag, fam, beta, length(pt), r$ll))
}
close(fit_con); close(ref_con)
cat("WROTE npfc_pm_fit.csv npfc_pm_ref.csv\n")
