

#' @rdname btlm
btlm.adapt <- function(formula, data, subset, na.action,
                 control = list(), ...
                 ) {
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "subset", "na.action"), names(mf), 0)
  mf <- mf[c(1, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")
  y <- model.response(mf, "numeric")
  X <- model.matrix(mt, mf)
  con <- list(n.sims = 10000, burnin = 5000, n.save = 1000, block = 200,
              alpha.tau = 0.1, beta.tau = 1e-6,
              rng.seed = as.integer(Sys.time())
              )
  nmsC <- names(con)
  con[(namc <- names(control))] <- control
  if (length(noNms <- namc[!(namc %in% nmsC)]))
    warning ("unknown names in control: ", paste(noNms, collapse = ", "))
  btlm.fit(X, y, con)
}




btlm.fit <- function(X, y, control) {
  start <- .univar.coefs(X, y)
  alpha <- start$coef
  lambda <- start$M / 2
  tau.sq <- 1 / var(y)
  thin <- max(floor((control$n.sims - control$burnin) / control$n.save), 1)
  kernels <- list(
    alpha = list(K = diag(start$se.coef),
                 scale = 2.4 / sqrt(sum(.get.beta(alpha, lambda) != 0))),
    lambda = list(K = start$M / sqrt(12), scale = 1)
    )
  iter <- 0
  while (iter < control$burnin) {
    sim <- .update.block(control$block, alpha, lambda, tau.sq,
                         kernels, start$M, control$alpha.tau, control$beta.tau,
                         y, X
                         )
    alpha <- c(tail(sim$alpha, 1))
    lambda <- tail(sim$lambda, 1)
    tau.sq <- tail(sim$tau.sq, 1)
    iter <- iter + control$block
    kernels$alpha$scale <- .scale.fun(kernels$alpha$scale, sim$p.jump$alpha)
    kernels$lambda$scale <- .scale.fun(kernels$lambda$scale, sim$p.jump$lambda)
    if (sim$p.jump$alpha > 0.05 && sim$p.jump$alpha < 0.7)
      kernels$alpha$K <- .alpha.kernel(sim$alpha, sim$lambda)
    if (sim$p.jump$lambda > 0.05 && sim$p.jump$lambda < 0.7)
      kernels$lambda$K <- sqrt(var(sim$lambda))
  }
  cat ("After burnin (", control$burnin, " iterations)\nPr(jump) =\nalpha: ",
       sim$p.jump$alpha, "\nlambda: ", sim$p.jump$lambda, "\n", sep = ""
       )
  sim <- .update.block(control$n.save, alpha, lambda, tau.sq,
                       kernels, start$M, control$alpha.tau, control$beta.tau,
                       y, X, thin
                       )
  sim$alphaHat <- colMeans(sim$alpha)
  sim$lambdaHat <- mean(sim$lambda)
  sim$sigmaSqHat <- 1 / mean(sim$tau.sq)
  sim$sigmaSq <- 1 / sim$tau.sq
  sim <- sim[c("alphaHat", "lambdaHat", "sigmaSqHat", "alpha", "lambda", "sigmaSq")]
  class (sim) <- "btlm"
  return (sim)
}





.update.block <- function(nsims, alpha, lambda, tau.sq,
                          kernels, M, tau.sq.alpha, tau.sq.beta,
                          y, X, thin = 1
                          ) {
  p.jump <- list(alpha = 0, lambda = 0)
  alpha.sims <- array(NA, c(nsims, length(alpha)))
  lambda.sims <- numeric(nsims)
  tau.sq.sims <- numeric(nsims)
  K.alpha <- kernels$alpha$K * kernels$alpha$scale
  K.lambda <- kernels$lambda$K * kernels$lambda$scale
  iter <- 1
  thin.count <- 1
  while (thin.count <= nsims) {
    sim <- alpha.update(alpha, K.alpha, y, X, lambda, tau.sq, 0, 0.001)
    alpha <- sim$theta
    p.jump$alpha <- p.jump$alpha + sim$p
    sim <- lambda.update(lambda, M, K.lambda, y, X, alpha, tau.sq)
    lambda <- sim$theta
    p.jump$lambda <- p.jump$lambda + sim$p
    tau.sq <- tau.sq.update(y, X, lambda, alpha, tau.sq.alpha, tau.sq.beta)
    if (iter %% thin == 0) {
      alpha.sims[thin.count, ] <- alpha
      lambda.sims[thin.count] <- lambda
      tau.sq.sims[thin.count] <- tau.sq
      thin.count <- thin.count + 1
    }
    iter <- iter + 1
  }
  p.jump <- lapply(p.jump, function(x) x / iter)
  list (alpha = alpha.sims, lambda = lambda.sims, tau.sq = tau.sq.sims,
        p.jump = p.jump
        )
}







## inverse residual variance updated via full conditional Gibbs
tau.sq.update <- function(y, X, lambda, alpha, alpha.prior, beta.prior) {
  beta <- .get.beta(alpha, lambda)
  res <- c(y - X %*% beta)
  rgamma(1, alpha.prior + length(y) / 2, beta.prior + c(crossprod(res)) / 2)
}


## threshold parameter updating via random walk
lambda.update <- function(lambda, M, K, y, X, alpha, tau.sq) {
  proposal <- rnorm(1, lambda, K)
  p <- 0
  if (proposal < M && proposal > 0) {
    r <- exp(logposterior.lambda(proposal, y, X, alpha, tau.sq) -
             logposterior.lambda(lambda, y, X, alpha, tau.sq)
             )
    p <- min(r, 1)
    if (runif(1) < r)
      lambda <- proposal
  }
  list (theta = lambda, p = p)
}


## model coefficients updated via random walk
alpha.update <- function(alpha, K, y, X, lambda, tau.sq, mu, tau.mu) {
  proposal <- c(alpha + K %*% rnorm(length(alpha)))
  r <- exp(logposterior.alpha(proposal, y, X, lambda, tau.sq, mu, tau.mu) -
           logposterior.alpha(alpha, y, X, lambda, tau.sq, mu, tau.mu)
           )
  p <- min(r, 1)
  if (runif(1) < r)
    alpha <- proposal
  list (theta = alpha, p = p)
}



## estimate alpha jumping kernel given simulations of alpha, lambda
## intercept assumed in first column of alpha
.alpha.kernel <- function(alpha, lambda, level = 0.9) {
  Kernel <- array(0, c(ncol(alpha), ncol(alpha)))
  diag(Kernel) <- apply(alpha, 2, function(x) sqrt(var(x)))
  tryCatch ({
    include <- colMeans(apply(alpha, 2, function(x) abs(x) > lambda)) > level
    include[1] <- TRUE
    Kernel[include, include] <- t(chol(cov(alpha[, include])))
  }, error = function(e) NULL
  )
  return (Kernel)
}
