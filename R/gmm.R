set.seed(815)
library(MASS)
library(ggplot2)

N <- 1000000
K <- 3
D <- 2
w_true <- c(0.3, 0.5, 0.2)
mu_true <- list(c(2, 2),c(-2, -2), c(4, -4))
sigma_true <- list(matrix(c(1, 0.2, 0.2, 1), nrow = D), 
                   matrix(c(1, -0.3, -0.3, 1), nrow = D), 
                   matrix(c(1, 0, 0, 1), nrow = D))
z <- sample(1:K, size = N, replace = TRUE, prob = w_true)
X <- matrix(0, nrow = N, ncol = D)
for (i in 1:N) {
  k <- z[i]
  X[i, ] <- mvrnorm(1, mu = mu_true[[k]], Sigma = sigma_true[[k]])
}
gmm_plot <- as.data.frame(X)
gmm_plot$cluster <- z

ggplot(gmm_plot, aes(x = V1, y = V2, color = factor(cluster))) +
  geom_point(alpha = 0.7) +
  labs(title = "Gaussian Mixture Data", color = "True Cluster") +
  theme_minimal()

gmm_data <- as.matrix(X)
result <- gmm(gmm_data, K, 10000)
result$mu
result$sigma