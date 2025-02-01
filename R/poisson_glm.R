set.seed(123)
library(here)
Rcpp::sourceCpp(file.path(here(), "R", "irwls_poisson.cpp"))

# Generate data
n <- 100
p <- 3
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
beta_true <- c(0.5, 1, 2)
eta <- X %*% beta_true
y <- rpois(n, exp(eta))

# Fit Poisson regression using IRWLS
beta_est <- irwls_poisson(X, y)

# Compare estimated vs. true coefficients
print(beta_est)
print(beta_true)