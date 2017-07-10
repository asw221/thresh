

## get mass univariate coefficients, standard errors, and max
## intercept assumed in first column of X
.univar.coefs <- function(X, y) {
  coefs <- array(dim = c(ncol(X), 2))
  M <- 0
  for (i in 2:ncol(X)) {
    summ <- summary(lm(y ~ X[, i]))
    coefs[i, ] <- coef(summ)[2, 1:2]
    if (abs(coefs[i, 1]) > M) {
      M <- abs(coefs[i, 1])
      coefs[1, ] <- coef(summ)[1, 1:2]
    }
  }
  list(coef = coefs[, 1], se.coef = coefs[, 2], M = M)
}



## get estimated beta's given alpha, lambda
.get.beta <- function(alpha, lambda) {
  ifelse(abs(alpha) > lambda, alpha, 0)
}



## adjust kernel scaling constants
.scale.fun <- function(x, p) {
  if (p < 0.32)
    x * 0.9
  else if (p > 0.38)
    x * 1.1
  else
    x
}
