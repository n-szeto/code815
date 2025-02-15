#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

//' Sample from Mixture of Two Gaussians via Random Walk Metropolis
//' 
//' @param n Number of samples
//' @param mu1 Mean of first Gaussian component
//' @param mu2 Mean of second Gaussian component
//' @param sigma1 Standard deviation of first Gaussian component
//' @param sigma2 Standard deviation of second Gaussian component
//' @param w1 Weight of first Gaussian component
//' @param w2 Weight of second Gaussian component
//' @param mean_proposal Mean of proposal distribution
//' @param sigma_proposal Standard deviation of proposal distribution
//' @return Samples from mixture of two Gaussians
//' 
//' @export
// [[Rcpp::export]]
arma::vec rmixnorm(int n = 1000,
                   double mu1 = 0,
                   double mu2 = 0,
                   double sigma1 = 1,
                   double sigma2 = 1,
                   double w1 = 0.5,
                   double w2 = 0.5,
                   double mean_proposal = 0,
                   double sigma_proposal = 1) {
  
  double x_prev = arma::randn();
  double x_curr = 0;
  double p_prev = 0;
  double p_curr = 0;
  double iter = 0;
  arma::vec samples(n);
  
  while (iter < n) {
    x_curr = x_prev + (mean_proposal + sigma_proposal * arma::randn());
    p_prev = arma::normpdf(x_prev, mu1, sigma1) * w1 +
      arma::normpdf(x_prev, mu2, sigma2) * w2;
    p_curr = arma::normpdf(x_curr, mu1, sigma1) * w1 +
      arma::normpdf(x_curr, mu2, sigma2) * w2;
    if (arma::randu() < std::min(1.0, p_curr / p_prev)) {
      samples(iter) = x_curr;
      x_prev = x_curr;
      iter++;
    } else {
      continue;
    }
  }
  return samples;
}

//' Sample from Truncated Mixture of Two Gaussians via Random Walk Metropolis
//' 
//' @param n Number of samples
//' @param mu1 Mean of first Gaussian component
//' @param mu2 Mean of second Gaussian component
//' @param sigma1 Standard deviation of first Gaussian component
//' @param sigma2 Standard deviation of second Gaussian component
//' @param w1 Weight of first Gaussian component
//' @param w2 Weight of second Gaussian component
//' @param mean_proposal Mean of proposal distribution
//' @param sigma_proposal Standard deviation of proposal distribution
//' @param lower Lower bound of truncation
//' @param upper Upper bound of truncation
//' @return Samples from mixture of two Gaussians
//' 
//' @export
// [[Rcpp::export]]
arma::vec rtruncmixnorm(int n = 1000,
                        double mu1 = 0,
                        double mu2 = 0,
                        double sigma1 = 1,
                        double sigma2 = 1,
                        double w1 = 0.5,
                        double w2 = 0.5,
                        double mean_proposal = 0,
                        double sigma_proposal = 1,
                        double lower = -1,
                        double upper = 1) {
 
 double x_prev = arma::randn();
 double x_curr = 0;
 double p_1_prev = 0;
 double p_2_prev = 0;
 double p_1_curr = 0;
 double p_2_curr = 0;
 double p_prev = 0;
 double p_curr = 0;
 double iter = 0;
 arma::vec samples(n);
 
 while (iter < n) {
   x_curr = x_prev + (mean_proposal + sigma_proposal * arma::randn());
   
   if (x_curr < lower || x_curr > upper) {
     continue;
   } else if (x_prev < lower || x_prev > upper) {
     continue;
   }
   
   p_1_curr = arma::normpdf(x_curr, mu1, sigma1) /
     (arma::normcdf(upper, mu1, sigma1) - arma::normcdf(lower, mu1, sigma1));
   p_2_curr = arma::normpdf(x_curr, mu2, sigma2) /
     (arma::normcdf(upper, mu2, sigma2) - arma::normcdf(lower, mu2, sigma2));
   p_1_prev = arma::normpdf(x_prev, mu1, sigma1) /
     (arma::normcdf(upper, mu1, sigma1) - arma::normcdf(lower, mu1, sigma1));
   p_2_prev = arma::normpdf(x_prev, mu2, sigma2) /
     (arma::normcdf(upper, mu2, sigma2) - arma::normcdf(lower, mu2, sigma2));
   p_prev = p_1_prev * w1 + p_2_prev * w2;
   p_curr = p_1_curr * w1 + p_2_curr * w2;
   
   if (arma::randu() < std::min(1.0, p_curr / p_prev)) {
     samples(iter) = x_curr;
     x_prev = x_curr;
     iter++;
   } else {
     continue;
   }
 }
 return samples;
}