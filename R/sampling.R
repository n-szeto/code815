set.seed(815)
library(here)
library(coda)
Rcpp::sourceCpp(file.path(here(), "R", "sampling.cpp"))

# Generate sample from mixture of Gaussians
mix.sample <- mcmc(rmixnorm(100000, -3, 3, 1, 1, 0.3, 0.7, 0, 1))
traceplot(mix.sample) # Trace plot of sample
hist(mix.sample) # Histogram of sample

# Generate sample from Beta(a,b) distribution
trunc.mix.sample <- mcmc(rtruncmixnorm(100000, -3, 3, 1, 1, 0.3, 0.7, 0, 1, -4, 4))
traceplot(trunc.mix.sample) # Trace plot of sample
hist(trunc.mix.sample) # Histogram of sample
