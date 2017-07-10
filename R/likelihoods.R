

## log-posterior likelihood of alpha, up to proportionality
## mu, tau.mu are priors on alpha, tau.sq is inverse residual variance
logposterior.alpha <- function(theta, y, X, lambda, tau.sq, mu, tau.mu) {
  beta <- .get.beta(theta, lambda)
  -0.5 * c(tau.sq * crossprod(y - X %*% beta) + tau.mu * crossprod(theta - mu))
}


## log-posterior of lambda, up to proportionality
logposterior.lambda <- function(theta, y, X, alpha, tau.sq) {
  beta <- .get.beta(alpha, theta)
  -0.5 * tau.sq * c(crossprod(y - X %*% beta))
}







