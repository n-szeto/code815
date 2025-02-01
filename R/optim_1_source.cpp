#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

double loss_ridge(const arma::vec& y,
                  const arma::mat& A,
                  const arma::vec& x,
                  double lambda) {
  arma::vec res = y - A * x;
  
  // Ridge regression loss: (1/2) * sum(res^2) + lambda * sum(x^2)
  double loss = 0.5 * arma::accu(res % res) + lambda * arma::accu(x % x);
  return loss;
}

//' Least Squares Gradient Descent with Ridge Regularization
//'
//' @param y      A (n x 1) vector of response variables
//' @param A      A (n x p) matrix of predictor variables
//' @param x_init A (p x 1) initial vector of effect sizes
//' @param lambda Regularization parameter
//' @param alpha  Learning rate for gradient descent
//' @param max_iter Maximum number of iterations
//' @param tol    Convergence tolerance
//' @return Estimated coefficients
// [[Rcpp::export]]
Rcpp::List gradient_descent_lsq(const arma::vec& y,
                               const arma::mat& A,
                               arma::vec x, 
                               double lambda = 1, double gamma = 0.01,
                               int max_iter = 1000, double tol = 1e-6) {
 
 arma::vec gradient = A.t() * (A * x - y) + 2 * lambda * x;
 arma::vec diff_rec = arma::zeros(max_iter);  // Store difference in coefficients
 arma::vec loss_rec = arma::zeros(max_iter);  // Store loss at each iteration
 double loss = loss_ridge(y, A, x, lambda);
 double prev_loss = loss;
 double diff = std::numeric_limits<double>::infinity();
 for (int iter = 0; iter < max_iter & diff > tol; iter++) {
   
   // Update coefficients
   x = x - gamma * gradient;
   
   // Compute gradient
   gradient = A.t() * (A * x - y) + 2 * lambda * x;
   
   loss = loss_ridge(y, A, x, lambda);
   loss_rec(iter) = loss;
   diff_rec(iter) = (prev_loss - loss) / abs(prev_loss);
   diff = abs(diff_rec(iter));
   prev_loss = loss;
 }
 
 return Rcpp::List::create(Rcpp::Named("x") = x,
                           Rcpp::Named("diff") = diff_rec,
                           Rcpp::Named("loss") = loss_rec);
}