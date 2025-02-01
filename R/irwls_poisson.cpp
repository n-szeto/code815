#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

//' Iteratively Reweighted Least Squares for Poisson Regression
//'
//' @param X Design matrix (n x p)
//' @param y Response vector (n x 1)
//' @return Estimated coefficients
//' @export
// [[Rcpp::export]]
arma::vec irwls_poisson(const arma::mat& X,
                        const arma::vec& y,
                        int max_iter = 25,
                        double tol = 1e-6) {
  int p = X.n_cols;  // Number of predictors
  
  // Initialize beta coefficients
  arma::vec beta = arma::zeros(p);
  
  for (int iter = 0; iter < max_iter; iter++) {
    
    // Compute linear predictor, clamping to prevent overflow
    arma::vec eta = arma::clamp(X * beta, -10, 10);
    
    // Compute mean function: mu = exp(eta)
    arma::vec mu = arma::exp(eta);
    
    // Compute weights
    arma::vec W_diag = mu;
    arma::mat W = arma::diagmat(W_diag);
    
    // Compute working dependent response
    arma::vec z = eta + (y - mu) / mu;
    
    // Compute new beta
    arma::mat XtWX = X.t() * W * X;
    arma::vec XtWz = X.t() * W * z;
    arma::vec beta_new = arma::solve(XtWX, XtWz);
    
    // Check for convergence
    if (arma::norm(beta_new - beta, 2) < tol) break;
    
    // Update beta
    beta = beta_new;
  }
  
  return beta;
}