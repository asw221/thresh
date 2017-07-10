
#' @title Bayesian Thresholded Linear Models
#'
#' @description
#' Fit a bayesian linear model with thresholded priors for simultaneous
#' variable selection and point estimation.
#'
#' @param formula
#' an R formula for design specification
#'
#' @param data
#' an optional \code{data.frame} containing model variables
#'
#' @param subset
#' an optional index vector to subset input data
#'
#' @param na.action
#' an optional function given to handle missing data
#' cases
#'
#' @param control an optional list to control MCMC related parameters. See
#'   "Details"
#'
#' @details
#' The \code{control} list allows tuning of specified default parameters,
#' including: \code{n.sims} (default = 10000), the total number of MCMC
#' samples; \code{burnin} (default = 5000), the total number of samples
#' to discard from the beginning of the chain; \code{n.save} (default = 1000),
#' the total number of samples to retain from simulations; \code{alpha.tau}
#' (default = 0.1) and \code{beta.tau} (default = 1e-6), prior shape and rate
#' parameters for the Gamma-distributed residual precision parameter,
#' \eqn{\tau^2}; and \code{rng.seed}, an integer-valued random seed.
#'
#' In addition, there's another \code{control} parameter, \code{block}
#' (default = 200), which controls how often Metropolis-Hastings jumping
#' kernels are updated. Kernels are adaptively updated every \code{block}'th
#' iteration during the \code{burnin} period in order to control jumping
#' rates for more accurate and efficient simulations.
#'
#' Prior specification for the fitted model follows the heirarchy:
#'
#' \deqn{y_i \sim N(\mu_i, \tau^{-2})}{y[i] ~ N(\mu[i], 1 / \tau^2)}
#' \deqn{\mu_i = \alpha_0 + \Sum_{j = 1}^{P} X_{ij} I(|\alpha_j| > \lambda) \alpha_j}{\mu[i] = \alpha[0] + \Sum_j X[ij] * I(|\alpha[j]| > \lambda) * \alpha[j]}
#' \deqn{\alpha_j \sim N(0, 1000)}{\alpha[j] ~ N(0, 1000)}
#' \deqn{\tau^{-2} = \sigma^2 \sim Gamma(\alpha_{\tau}, \beta_{\tau})}{1 / \tau^2 = \sigma^2 ~ Gamma(\alpha[\tau], \beta[\tau])}
#' \deqn{\lambda \sim Unif(0, M)}{\lambda ~ Unif(0, M)}
#'
#' @return An object of class \code{"btlm"} listing simulation output
#' and estimated parameter values.
#'
#' @name btlm
#'

btlm <- function(formula, data, subset, na.action,
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
  fit <- .Call("bayesTLM", X, y, con$alpha.tau, con$beta.tau,
        con$n.sims, con$burnin, con$n.save, con$block, con$rng.seed,
        PACKAGE = "thresh"
        )
  class (fit) <- "btlm"
  fit
}
