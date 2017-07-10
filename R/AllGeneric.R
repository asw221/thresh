

## extract coefficients at a given posterior probability of inclusion level
coef.btlm <- function(object, prob.included = 0.9, ...) {
  if (prob.included < 0 || prob.included > 1)
    stop ("prob.included must be between [0, 1]")
  pprob <- colMeans(apply(object$alpha, 2, function(x) abs(x) > object$lambda))
  pprob[1] <- 1
  colMeans(object$alpha) * as.numeric(pprob > prob.included)
}


## extract residual standard deviation
sigma.btlm <- function(object, ...) {
  sqrt(object$sigmaSqHat)
}






