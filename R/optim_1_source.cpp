#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
double loss_ridge(arma::vec y, arma::mat A, arma::vec x, double lambda) {
  int n = y.n_elem;
  int p = x.n_elem;
  arma::vec res = y - A*x;
  return (1/2)*sum(res*res) + lambda * sum(x*x);
}

// [[Rcpp::export]]
Rcpp::List gradient_descent_lsq(arma::vec y, arma::mat A, arma::vec x0,
                                double lambda, double gamma, double tol,
                                int max_iter) {
  int n = y.n_elem;
  int p = A.n_cols;
  
  arma::mat AA = A.t() * A;
  arma::vec Ay = A.t() * y;
  arma::vec grad = AA * x0 - Ay;
  
  double loss = loss_ridge(y, A, x0, lambda);
  grad = grad + 2 * lambda * x0;
  
  double prevloss = loss;
  arma::vec x = x0;
  int iter = 1;
  double diff = std::numeric_limits<double>::infinity();
  arma::vec diff_rec;
  arma::vec loss_rec;
  
  while(iter < max_iter && diff > tol) {
    x = x - gamma * grad;
    grad = AA * x - Ay;
    
    loss = loss_ridge(y, A, x, lambda);
    grad = grad + 2 * lambda * x;
    
    diff_rec(iter) = (prevloss - loss) / std::abs(prevloss);
    diff = std::abs(diff_rec(iter));
    
    loss_rec(iter) = loss;
    prevloss = loss;
    iter++;
  }
  
  return Rcpp::List::create(Rcpp::Named("x") = x,
                            Rcpp::Named("diff") = diff,
                            Rcpp::Named("loss") = loss);
}