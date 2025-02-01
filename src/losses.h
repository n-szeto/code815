#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
double loss_ridge(const arma::vec& y,
                  const arma::mat& A,
                  const arma::vec& x,
                  double lambda);