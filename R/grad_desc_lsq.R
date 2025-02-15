set.seed(123)
library(here)

# Generate data
n <- 100
p <- 3
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
beta_true <- c(0.5, 1, 2)
y <- X %*% beta_true + rnorm(n, sd = 0.5)

# Set initial parameters
beta_init <- rep(0, p)
lambda <- 0.1
gamma <- 0.01
max_iter <- 5000
tol <- 1e-6

# Run gradient descent optimization w/ ridge regularization
result <- gradient_descent_lsq(y, X, beta_init, lambda, gamma, max_iter, tol)

# Extract results
beta_est <- result$x
loss_values <- result$loss
diff_values <- result$diff

# Compare estimated vs. true coefficients
print(beta_est)
print(beta_true)

# Plot loss over iterations
plot(loss_values, type = "l", col = "blue", lwd = 2, xlim = c(0, 10),
     xlab = "Iteration", ylab = "Loss",
     main = "Loss Convergence in Gradient Descent")

# Plot convergence criterion over iterations
plot(diff_values, type = "l", col = "red", lwd = 2, xlim = c(0, 5),
     xlab = "Iteration", ylab = "Convergence Criterion",
     main = "Convergence Criterion in Gradient Descent")
