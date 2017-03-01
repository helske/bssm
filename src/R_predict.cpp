#include "ugg_ssm.h"
#include "ugg_bsm.h"

// [[Rcpp::export]]
Rcpp::List gaussian_predict(const Rcpp::List& model_,
  const arma::vec& probs, const bool pred_obs,
  const arma::mat theta, const arma::mat alpha, const arma::uvec& counts, 
  const unsigned int model_type) {
  
  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;
  
  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::vec y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }
  
  
  switch (model_type) {
  case 1: {
    ugg_ssm model(clone(model_), 1, 0, 0, 0, 0);
    return model.predict(probs, theta, alpha, counts, pred_obs);
  } break;
  case 2: {
    ugg_bsm model(clone(model_), 1);
    return model.predict(probs, theta, alpha, counts, pred_obs);
  } break;
  }
  return Rcpp::List::create(Rcpp::Named("error") = arma::datum::inf);
}

