#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ugg_bsm.h"
#include "ung_bsm.h"

// [[Rcpp::export]]
Rcpp::List gaussian_approx_model(const Rcpp::List& model_, arma::vec mode_estimate,
  unsigned int max_iter, double conv_tol, unsigned int model_type) {
  
  switch (model_type) {
  case 1: {
    ung_ssm model(clone(model_), 1);
    ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
    return Rcpp::List::create(Rcpp::Named("y") = approx_model.y,
      Rcpp::Named("H") = approx_model.H);
  } break;
  case 2: {
    ung_bsm model(clone(model_), 1);
    ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
    return Rcpp::List::create(Rcpp::Named("y") = approx_model.y,
      Rcpp::Named("H") = approx_model.H);
  } break;
  case 3: {
    ung_bsm model(clone(model_), 1);
    ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
    return Rcpp::List::create(Rcpp::Named("y") = approx_model.y,
      Rcpp::Named("H") = approx_model.H);
  } break;
  default: 
    return Rcpp::List::create(Rcpp::Named("y") = 0, Rcpp::Named("H") = 0);
  }
}