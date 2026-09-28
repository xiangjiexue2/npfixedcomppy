# Density + scalar references for npfixedcomppy vs R npfixedcomp2.
# Reads the CURRENT data files on disk (does NOT regenerate them) and writes:
#   npfc_scalar_ref.csv         tab-separated: tag ll iter convergence mingrad beta family flag
#   npfc_density_ref.csv        (x, cm, ep, fix) mixture density on a grid
#   npfc_density_ref_big.csv    (x, big) mixture density on a grid
#
# The density is the continuous mixing part (dnpnorm). A point mass at 0
# (from estpi0 / a fixed component at 0) has no density and is intentionally
# excluded -- we compare the overall mixture density curve.
library(npfixedcomp2)
dnpnorm <- npfixedcomp2:::dnpnorm

data   <- as.numeric(read.csv("npfc_data_1000.csv")[[1]])
data5k <- as.numeric(read.csv("npfc_data_5000.csv")[[1]])

r_cm  <- computemixdist(data, method = "npnormll")
e_ep  <- estpi0(data, method = "npnormll", val = 2)
r_f   <- computemixdist(data, mu0 = c(-0.5), pi0 = c(0.3), method = "npnormll")
r_big <- computemixdist(data5k, method = "npnormll")

scal <- function(tag, r, con) {
  cat(tag, sprintf("%.17g", r$ll), r$iter, r$convergence,
      sprintf("%.17g", r$min.gradient), sprintf("%.17g", r$beta),
      r$family, r$flag, sep = "\t", "\n", file = con)
}
con <- file("npfc_scalar_ref.csv", "w")
cat("tag\tll\titer\tconvergence\tmin.gradient\tbeta\tfamily\tflag\n", file = con)
scal("CM", r_cm, con)
scal("EP", e_ep, con)
scal("CM_FIXED", r_f, con)
scal("CM_BIG", r_big, con)
close(con)

# --- density references ---
g1 <- seq(min(data) - 5, max(data) + 5, length.out = 2001)
g2 <- seq(min(data5k) - 5, max(data5k) + 5, length.out = 2001)
d1 <- data.frame(
  x   = g1,
  cm  = dnpnorm(g1, r_cm$mix$pt,  r_cm$mix$pr,  r_cm$beta),
  ep  = dnpnorm(g1, e_ep$mix$pt,  e_ep$mix$pr,  e_ep$beta),
  fix = dnpnorm(g1, r_f$mix$pt,   r_f$mix$pr,   r_f$beta)
)
d2 <- data.frame(x = g2, big = dnpnorm(g2, r_big$mix$pt, r_big$mix$pr, r_big$beta))
write.csv(d1, "npfc_density_ref.csv", row.names = FALSE)
write.csv(d2, "npfc_density_ref_big.csv", row.names = FALSE)

cat("WROTE npfc_scalar_ref.csv npfc_density_ref.csv npfc_density_ref_big.csv\n")
cat(sprintf("CM   ll=%.17g  npt=%d\n", r_cm$ll,  length(r_cm$mix$pt)))
cat(sprintf("EP   ll=%.17g  npt=%d\n", e_ep$ll,  length(e_ep$mix$pt)))
cat(sprintf("FIX  ll=%.17g  npt=%d\n", r_f$ll,   length(r_f$mix$pt)))
cat(sprintf("BIG  ll=%.17g  npt=%d\n", r_big$ll, length(r_big$mix$pt)))
