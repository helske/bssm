#include <RcppArmadillo.h>
typedef arma::vec (*vec_fnPtr)(const arma::vec&, const arma::vec&);
typedef arma::mat (*mat_fnPtr)(const arma::vec&, const arma::vec&);
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec pointer_test_vec(SEXP x) {
  
  arma::vec alpha(1);
  alpha(0) = 2;
  arma::vec theta(1);
  theta(0) = 1;
  Rcpp::XPtr<vec_fnPtr> xptr(x);
  vec_fnPtr funptr = *(xptr);
  return funptr(alpha, theta);
}
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat pointer_test_mat(SEXP x) {
  
  arma::vec alpha(1);
  alpha(0) = 2;
  arma::vec theta(1);
  theta(0) = 1;
  Rcpp::XPtr<mat_fnPtr> xptr(x);
  mat_fnPtr funptr = *(xptr);
  return funptr(alpha, theta);
}
// test that the pointers work by running pointer_test(x) where x is a pointer
