# References for the 4 binned ("...w") families vs R npfixedcomp2.
# Writes npfc_binned_ref.csv:
#   tag  method  ll  npt  iter  convergence  min.gradient  beta  family  flag
# and (for the deterministic LL cases) the pt/pr vectors as extra lines.
#
# The binned LL (npnormllw / nptllw) and fixed-component cases are
# DETERMINISTIC in R; CVM/AD are run-to-run non-deterministic (Eigen/LAPACK
# parallel reductions), so we gate those on the ll-recomputation invariant
# on the Python side and only record R's ll for information.
library(npfixedcomp2)

data  <- as.numeric(read.csv("npfc_data_1000.csv")[[1]])
data5k <- as.numeric(read.csv("npfc_data_5000.csv")[[1]])

scal <- function(tag, method, r, con, conpt, withpt) {
  cat(tag, method, sprintf("%.17g", r$ll), length(r$mix$pt), r$iter,
      r$convergence, sprintf("%.17g", r$min.gradient),
      sprintf("%.17g", r$beta), r$family, r$flag, sep = "\t", "\n", file = con)
  if (withpt) {
    cat(tag, "_pt", paste(r$mix$pt, collapse = " "), "\n", file = conpt)
    cat(tag, "_pr", paste(r$mix$pr, collapse = " "), "\n", file = conpt)
  }
}

con  <- file("npfc_binned_ref.csv", "w")
conpt <- file("npfc_binned_pt.txt", "w")
cat("tag\tmethod\tll\tnpt\titer\tconvergence\tmin.gradient\tbeta\tfamily\tflag\n",
    file = con)

# ---- computemixdist (4 families) ----
r_llw  <- computemixdist(data,  method = "npnormllw")
r_cvmw <- computemixdist(data,  method = "npnormcvmw")
r_adw  <- computemixdist(data,  method = "npnormadw")
r_tllw <- computemixdist(data,  method = "nptllw")
r_tllw5 <- computemixdist(data, method = "nptllw", beta = 5)

scal("LLW",  "npnormllw",  r_llw,  con, conpt, TRUE)
scal("CVMW", "npnormcvmw", r_cvmw, con, conpt, FALSE)
scal("ADW",  "npnormadw",  r_adw,  con, conpt, FALSE)
scal("TLLW", "nptllw",     r_tllw, con, conpt, TRUE)
scal("TLLW5","nptllw5",    r_tllw5,con, conpt, TRUE)

# ---- fixed-component (deterministic) ----
r_fix <- computemixdist(data, method = "npnormllw", mu0 = c(-0.5), pi0 = c(0.3))
scal("LLW_FIX", "npnormllw_fix", r_fix, con, conpt, TRUE)

# ---- estpi0 (recommended thresholds) ----
e_llw  <- estpi0(data, method = "npnormllw",  val = 2)
e_cvmw <- estpi0(data, method = "npnormcvmw", val = 0.1)
e_adw  <- estpi0(data, method = "npnormadw",  val = 1)
e_tllw <- estpi0(data, method = "nptllw",     val = 2)
scal("EP_LLW", "estpi0_npnormllw",  e_llw,  con, conpt, FALSE)
scal("EP_CVMW", "estpi0_npnormcvmw", e_cvmw, con, conpt, FALSE)
scal("EP_ADW",  "estpi0_npnormadw",  e_adw,  con, conpt, FALSE)
scal("EP_TLLW", "estpi0_nptllw",     e_tllw, con, conpt, FALSE)

close(con); close(conpt)

cat("WROTE npfc_binned_ref.csv npfc_binned_pt.txt\n")
res <- list(LLW = r_llw, CVMW = r_cvmw, ADW = r_adw, TLLW = r_tllw,
            TLLW5 = r_tllw5, FIX = r_fix,
            EP_L = e_llw, EP_C = e_cvmw, EP_A = e_adw, EP_T = e_tllw)
for (nm in names(res)) {
  r <- res[[nm]]
  cat(sprintf("%-6s ll=%.9g  npt=%d  iter=%d  conv=%d\n",
              nm, r$ll, length(r$mix$pt), r$iter, r$convergence))
}
