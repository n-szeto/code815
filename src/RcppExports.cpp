// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// gmm
Rcpp::List gmm(const arma::mat& X, int K, int max_iter, double tol);
RcppExport SEXP _code815_gmm(SEXP XSEXP, SEXP KSEXP, SEXP max_iterSEXP, SEXP tolSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    rcpp_result_gen = Rcpp::wrap(gmm(X, K, max_iter, tol));
    return rcpp_result_gen;
END_RCPP
}
// irwls_poisson
arma::vec irwls_poisson(const arma::mat& X, const arma::vec& y, int max_iter, double tol);
RcppExport SEXP _code815_irwls_poisson(SEXP XSEXP, SEXP ySEXP, SEXP max_iterSEXP, SEXP tolSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    rcpp_result_gen = Rcpp::wrap(irwls_poisson(X, y, max_iter, tol));
    return rcpp_result_gen;
END_RCPP
}
// loss_ridge_samp
double loss_ridge_samp(const arma::vec& y, const arma::mat& A, const arma::vec& x, double lambda);
RcppExport SEXP _code815_loss_ridge_samp(SEXP ySEXP, SEXP ASEXP, SEXP xSEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(loss_ridge_samp(y, A, x, lambda));
    return rcpp_result_gen;
END_RCPP
}
// gradient_descent_lsq
Rcpp::List gradient_descent_lsq(const arma::vec& y, const arma::mat& A, arma::vec x, double lambda, double gamma, int max_iter, double tol);
RcppExport SEXP _code815_gradient_descent_lsq(SEXP ySEXP, SEXP ASEXP, SEXP xSEXP, SEXP lambdaSEXP, SEXP gammaSEXP, SEXP max_iterSEXP, SEXP tolSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    rcpp_result_gen = Rcpp::wrap(gradient_descent_lsq(y, A, x, lambda, gamma, max_iter, tol));
    return rcpp_result_gen;
END_RCPP
}
// rmixnorm
arma::vec rmixnorm(int n, double mu1, double mu2, double sigma1, double sigma2, double w1, double w2, double mean_proposal, double sigma_proposal);
RcppExport SEXP _code815_rmixnorm(SEXP nSEXP, SEXP mu1SEXP, SEXP mu2SEXP, SEXP sigma1SEXP, SEXP sigma2SEXP, SEXP w1SEXP, SEXP w2SEXP, SEXP mean_proposalSEXP, SEXP sigma_proposalSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type mu1(mu1SEXP);
    Rcpp::traits::input_parameter< double >::type mu2(mu2SEXP);
    Rcpp::traits::input_parameter< double >::type sigma1(sigma1SEXP);
    Rcpp::traits::input_parameter< double >::type sigma2(sigma2SEXP);
    Rcpp::traits::input_parameter< double >::type w1(w1SEXP);
    Rcpp::traits::input_parameter< double >::type w2(w2SEXP);
    Rcpp::traits::input_parameter< double >::type mean_proposal(mean_proposalSEXP);
    Rcpp::traits::input_parameter< double >::type sigma_proposal(sigma_proposalSEXP);
    rcpp_result_gen = Rcpp::wrap(rmixnorm(n, mu1, mu2, sigma1, sigma2, w1, w2, mean_proposal, sigma_proposal));
    return rcpp_result_gen;
END_RCPP
}
// rtruncmixnorm
arma::vec rtruncmixnorm(int n, double mu1, double mu2, double sigma1, double sigma2, double w1, double w2, double mean_proposal, double sigma_proposal, double lower, double upper);
RcppExport SEXP _code815_rtruncmixnorm(SEXP nSEXP, SEXP mu1SEXP, SEXP mu2SEXP, SEXP sigma1SEXP, SEXP sigma2SEXP, SEXP w1SEXP, SEXP w2SEXP, SEXP mean_proposalSEXP, SEXP sigma_proposalSEXP, SEXP lowerSEXP, SEXP upperSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type mu1(mu1SEXP);
    Rcpp::traits::input_parameter< double >::type mu2(mu2SEXP);
    Rcpp::traits::input_parameter< double >::type sigma1(sigma1SEXP);
    Rcpp::traits::input_parameter< double >::type sigma2(sigma2SEXP);
    Rcpp::traits::input_parameter< double >::type w1(w1SEXP);
    Rcpp::traits::input_parameter< double >::type w2(w2SEXP);
    Rcpp::traits::input_parameter< double >::type mean_proposal(mean_proposalSEXP);
    Rcpp::traits::input_parameter< double >::type sigma_proposal(sigma_proposalSEXP);
    Rcpp::traits::input_parameter< double >::type lower(lowerSEXP);
    Rcpp::traits::input_parameter< double >::type upper(upperSEXP);
    rcpp_result_gen = Rcpp::wrap(rtruncmixnorm(n, mu1, mu2, sigma1, sigma2, w1, w2, mean_proposal, sigma_proposal, lower, upper));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_code815_gmm", (DL_FUNC) &_code815_gmm, 4},
    {"_code815_irwls_poisson", (DL_FUNC) &_code815_irwls_poisson, 4},
    {"_code815_loss_ridge_samp", (DL_FUNC) &_code815_loss_ridge_samp, 4},
    {"_code815_gradient_descent_lsq", (DL_FUNC) &_code815_gradient_descent_lsq, 7},
    {"_code815_rmixnorm", (DL_FUNC) &_code815_rmixnorm, 9},
    {"_code815_rtruncmixnorm", (DL_FUNC) &_code815_rtruncmixnorm, 11},
    {NULL, NULL, 0}
};

RcppExport void R_init_code815(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
