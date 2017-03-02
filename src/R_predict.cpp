#include "ugg_ssm.h"
#include "ugg_bsm.h"
#include "ung_ssm.h"
#include "ung_bsm.h"
#include "ung_svm.h"

// [[Rcpp::export]]
Rcpp::List gaussian_predict(const Rcpp::List& model_,
  const arma::vec& probs, const arma::mat theta, const arma::mat alpha, 
  const arma::uvec& counts, const bool predict_obs,
  const bool predict_intervals,  const int model_type) {
  
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
    if (predict_intervals) {
      return model.predict_interval(probs, theta, alpha, counts, predict_obs);
    } else {
      return Rcpp::List::create(model.predict_sample(theta, alpha, counts, predict_obs));
    }
  } break;
  case 2: {
    ugg_bsm model(clone(model_), 1);
    if (predict_intervals) {
      return model.predict_interval(probs, theta, alpha, counts, predict_obs);
    } else {
      return Rcpp::List::create(model.predict_sample(theta, alpha, counts, predict_obs));
    }
  } break;
  }
  return Rcpp::List::create(Rcpp::Named("error") = arma::datum::inf);
}

// [[Rcpp::export]]
Rcpp::List nongaussian_predict(const Rcpp::List& model_,
  const arma::vec& probs, const arma::mat& theta, const arma::mat& alpha, 
  const arma::uvec& counts, const bool predict_obs,
  const int model_type) {
  
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
    ung_ssm model(clone(model_), 1, 0, 0, 0);
    return Rcpp::List::create(model.predict_sample(theta, alpha, counts, predict_obs));
  } break;
  case 2: {
    ung_bsm model(clone(model_), 1);
    return Rcpp::List::create(model.predict_sample(theta, alpha, counts, predict_obs));
  } break;
  case 3: {
    ung_svm model(clone(model_), 1);
    return Rcpp::List::create(model.predict_sample(theta, alpha, counts, predict_obs));
  } break;
  }
  return Rcpp::List::create(Rcpp::Named("error") = arma::datum::inf);
}

