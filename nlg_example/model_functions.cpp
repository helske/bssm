#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

arma::vec Z_fn(const arma::vec& alpha, const arma::vec& theta) {
  return alpha;
}
arma::mat H_fn(const arma::vec& alpha, const arma::vec& theta) {
  arma::mat chol_cov(1, 1);
  chol_cov(0, 0) = theta(0);
  return chol_cov;
}
arma::vec T_fn(const arma::vec& alpha, const arma::vec& theta) {
  return alpha;
}
arma::mat R_fn(const arma::vec& alpha, const arma::vec& theta) {
  arma::mat chol_cov(1, 1);
  chol_cov(0, 0) = theta(1);
  return chol_cov;
}
arma::mat Z_gn(const arma::vec& alpha, const arma::vec& theta) {
  return arma::mat(1, 1, arma::fill::ones);
}
arma::mat T_gn(const arma::vec& alpha, const arma::vec& theta) {
  return arma::mat(1, 1, arma::fill::ones);
}

double log_prior_pdf(const arma::vec& theta) {
  //uniform on [0, 10]
  double log_pdf = log(theta.n_elem / 10);
  if (theta(0) < 0 || theta(1) < 0) {
    log_pdf = -arma::datum::inf;
  }
  return log_pdf;
}

// [[Rcpp::export]]
Rcpp::List create_xptrs() {
  typedef arma::vec (*funcPtr_vec)(const arma::vec&, const arma::vec&);
  typedef arma::mat (*funcPtr_mat)(const arma::vec&, const arma::vec&);
  typedef double (*funcPtr_double)(const arma::vec&);
  
  return Rcpp::List::create(
    Rcpp::Named("Z_fn") = Rcpp::XPtr<funcPtr_vec>(new funcPtr_vec(&Z_fn)),
    Rcpp::Named("H_fn") = Rcpp::XPtr<funcPtr_mat>(new funcPtr_mat(&H_fn)),
    Rcpp::Named("T_fn") = Rcpp::XPtr<funcPtr_vec>(new funcPtr_vec(&T_fn)),
    Rcpp::Named("R_fn") = Rcpp::XPtr<funcPtr_mat>(new funcPtr_mat(&R_fn)),
    Rcpp::Named("Z_gn") = Rcpp::XPtr<funcPtr_mat>(new funcPtr_mat(&Z_gn)),
    Rcpp::Named("T_gn") = Rcpp::XPtr<funcPtr_mat>(new funcPtr_mat(&T_gn)),
    Rcpp::Named("log_prior_pdf") = 
      Rcpp::XPtr<funcPtr_double>(new funcPtr_double(&log_prior_pdf)));
}