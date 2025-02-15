#include <RcppArmadillo.h>

//' Multivariate Gaussian density
//' @param x Data vector
//' @param mu Mean vector
//' @param sigma Covariance matrix
//' @return Density value
double dmvnorm(const arma::vec& x,
               const arma::vec& mu,
               const arma::mat& sigma) {
  int xdim = x.n_elem;
  arma::vec z = x - mu;
  return exp(-0.5 * as_scalar(z.t() * sigma.i() * z)) /
    sqrt(pow(2 * M_PI, xdim) * det(sigma));
}

//' Gaussian Mixture Model via EM
//' @param K Number of components
//' @param X Data matrix
//' @param theta Initial parameters
//' @return Estimated coefficients
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List gmm(const arma::mat& X,
               int K,
               int max_iter = 100,
               double tol = 1e-6) {
  
  int N = X.n_rows;
  int D = X.n_cols;
  std::vector<arma::vec> mu(K);
  std::vector<arma::mat> sigma(K);
  arma::vec w(K);
  arma::mat gamma(N, K);
  double ll = -arma::datum::inf;
  
  for (int k = 0; k < K; k++) {
    mu[k] = X.row(arma::randi<arma::uword>(arma::distr_param(0, N-1))).t();
    sigma[k] = arma::eye(D, D);
    w(k) = 1.0 / K;
  }
  
  for (int iter = 0; iter < max_iter; iter++) {
    
    double ll_old = ll;
    
    // E-step
    for (int i = 0; i < N; i++) {
      double denom = 0.0;
      for (int k = 0; k < K; k++) {
        denom += w[k] * dmvnorm(X.row(i).t(), mu[k], sigma[k]);
      }
      for (int k = 0; k < K; k++) {
        gamma(i, k) = (w[k] * dmvnorm(X.row(i).t(), mu[k], sigma[k])) / denom;
      }
    }
    
    // Log-likelihood
    ll = 0.0;
    for (int i = 0; i < N; i++) {
      double tmp = 0.0;
      for (int k = 0; k < K; k++) {
        tmp += w[k] * dmvnorm(X.row(i).t(), mu[k], sigma[k]);
      }
      ll += log(tmp);
    }
    
    if (std::abs(ll - ll_old) < tol) {
      std::cout << "EM algorithm converged after " << iter + 1 << " iterations." << std::endl;
      break;
    }
    
    // M-step
    for (int k = 0; k < K; k++) {
      double gamma_sum = arma::sum(gamma.col(k));
      w[k] = gamma_sum / N;
      
      arma::vec numerator = arma::zeros(D);
      for (int i = 0; i < N; i++) {
        numerator += gamma(i, k) * X.row(i).t();
      }
      mu[k] = numerator / gamma_sum;
      
      arma::mat cov_k = arma::zeros<arma::mat>(D, D);
      for (int i = 0; i < N; i++) {
        arma::vec diff = X.row(i).t() - mu[k];
        cov_k += gamma(i, k) * (diff * diff.t());
      }
      sigma[k] = cov_k / gamma_sum;
    }
  }
  
  Rcpp::List mu_list(K);
  Rcpp::List sigma_list(K);
  for (int k = 0; k < K; k++) {
    mu_list[k] = mu[k];
    sigma_list[k] = sigma[k];
  }
  
  return Rcpp::List::create(
    Rcpp::Named("mu") = mu_list,
    Rcpp::Named("sigma") = sigma_list,
    Rcpp::Named("weights") = w
  );
}