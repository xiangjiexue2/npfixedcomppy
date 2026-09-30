# Gold references for npfixedcomppy covestEB / covestEB.cor vs R npfixedcomp2.
#
# The R `covestEB`/`covestEB.cor` (utility.R) compute, for an n x p matrix X:
#   covest = cov(X); index = diag(covest) > eps
#   (fisherdata | data) = atanh(lower-tri(cov2cor(...)))  [Fisher, col-major order]
#   r = computemixdist/estpi0(fisherdata|data, ...)
#   postmean = posteriormean(fisherdata|data, r[, fun = tanh])
#   ans = eye(p); ans[index,index] = returnlower(postmean)   # pre-projection corr
#   ans1 = correlationmatrixcpp(ans, tol = 1e-3)             # post-projection
#   varest = sqrt(diag(covest))
#   mat = ans1 * varest %o% varest ; correction.Fnorm = ||ans1 - ans||_F
#
# We re-implement the body here so we can capture the intermediates `ans` and
# `ans1` (which the real function does not return), plus the fit params. The
# Python test then: (a) feeds R's `ans` to its OWN correlationmatrixcpp and
# compares to R's `ans1` (strict — validates the C++ projection primitive),
# (b) recomputes mat/correction.Fnorm (strict), and (c) runs its own full
# pipeline and compares `ans` to R's `ans` at a loose gate (validates the
# column-major lower-triangle ordering; the fit itself is non-reproducible in
# R, so a loose gate, not a bit gate, is used here).
#
# Writes (tab-separated):
#   npfc_coveb_X.csv        the shared X (header row)
#   npfc_coveb_ans.csv      pre-projection correlation matrix (p x p)
#   npfc_coveb_ans1.csv     post-projection correlation matrix (p x p)
#   npfc_coveb_varest.csv   sample standard deviations (p)
#   npfc_coveb_fit.csv      tag family beta k pt pr  (semicolons)
#   npfc_coveb_mat.csv      final covariance estimate (p x p)
#   npfc_coveb_norm.csv     tag correction.Fnorm
library(npfixedcomp2)

set.seed(20240529)
n <- 200; p <- 24
X <- matrix(rnorm(n * p), n, p)
# induce some correlated structure so the projection has work to do
S <- matrix(0, p, p)
S[1:5, 1:5] <- 0.6
diag(S) <- 1
chol <- chol(S + diag(1e-6, p))
X <- X %*% chol
write.csv(X, "npfc_coveb_X.csv", row.names = FALSE)

# --- covestEB (Fisher) ---
covest <- cov(X)
index <- diag(covest) > .Machine$double.eps
fisherdata <- atanh(extractlower(cov2cor(covest[index, index])))
rE <- computemixdist(fisherdata, beta = sqrt(1 / (n - 3)))
postmeanE <- posteriormean(fisherdata, rE, fun = tanh)
ansE <- diag(nrow = p, ncol = p)
ansE[index, index] <- returnlower(postmeanE)
ans1E <- npfixedcomp2:::correlationmatrixcpp(ansE, tol = 1e-3)
varest <- sqrt(diag(covest))
matE <- ans1E * varest * rep(varest, rep(length(varest), length(varest)))
normE <- norm(ans1E - ansE, type = "F")

# --- covestEB.cor (raw correlations) ---
dataC <- extractlower(cov2cor(covest[index, index]))
rC <- computemixdist(dataC, beta = n, method = "npnormcll")
postmeanC <- posteriormean(dataC, rC)
ansC <- diag(nrow = p, ncol = p)
ansC[index, index] <- returnlower(postmeanC)
ans1C <- npfixedcomp2:::correlationmatrixcpp(ansC, tol = 1e-3)
matC <- ans1C * varest * rep(varest, rep(length(varest), length(varest)))
normC <- norm(ans1C - ansC, type = "F")

write.csv(ansE, "npfc_coveb_ansE.csv", row.names = FALSE)
write.csv(ans1E, "npfc_coveb_ans1E.csv", row.names = FALSE)
write.csv(ansC, "npfc_coveb_ansC.csv", row.names = FALSE)
write.csv(ans1C, "npfc_coveb_ans1C.csv", row.names = FALSE)
write.csv(matrix(varest, ncol = 1), "npfc_coveb_varest.csv", row.names = FALSE)
write.csv(matE, "npfc_coveb_matE.csv", row.names = FALSE)
write.csv(matC, "npfc_coveb_matC.csv", row.names = FALSE)

fit_con <- file("npfc_coveb_fit.csv", "w")
cat("tag,family,beta,k,pt,pr\n", file = fit_con)
for (fc in list(list(tag = "E", r = rE), list(tag = "C", r = rC))) {
  r <- fc$r
  cat(fc$tag, r$family, sprintf("%.17g", r$beta), length(r$mix$pt),
      paste(sprintf("%.17g", r$mix$pt), collapse = ";"),
      paste(sprintf("%.17g", r$mix$pr), collapse = ";"),
      sep = ",", "\n", file = fit_con)
}
close(fit_con)

norm_con <- file("npfc_coveb_norm.csv", "w")
cat("tag,norm\nE,", sprintf("%.17g", normE), "\nC,", sprintf("%.17g", normC), "\n", file = norm_con)
close(norm_con)

cat(sprintf("covestEB    : p=%d n=%d  npt(E)=%d  norm=%.6g\n", p, n, length(rE$mix$pt), normE))
cat(sprintf("covestEB.cor: p=%d n=%d  npt(C)=%d  norm=%.6g\n", p, n, length(rC$mix$pt), normC))
cat("WROTE npfc_coveb_*.csv\n")
