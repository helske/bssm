#include "model_ssm_ulg.h"
#include "model_ssm_ung.h"
#include "model_ssm_mng.h"
#include "model_bsm_ng.h"
#include "model_svm.h"
#include "model_ar1_ng.h"

// [[Rcpp::export]]
Rcpp::List importance_sample_ng(const Rcpp::List model_,
  unsigned int nsim, bool use_antithetic, const unsigned int seed, 
  const int model_type) {
  
  switch (model_type) { 
  case 0: {
    ssm_mng model(model_, seed);
    model.approximate();
    arma::cube alpha = model.approx_model.simulate_states(nsim);
    model.update_scales();
    arma::vec weights = model.importance_weights(alpha) - arma::accu(model.scales);
    double maxw = weights.max();
    weights = arma::exp(weights - maxw);
    
    return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
      Rcpp::Named("weights") = weights);
  } break;
  case 1: {
    ssm_ung model(model_, seed);
    model.approximate();
    arma::cube alpha = model.approx_model.simulate_states(nsim);
    model.update_scales();
    arma::vec weights = model.importance_weights(alpha) - arma::accu(model.scales);
    double maxw = weights.max();
    weights = arma::exp(weights - maxw);
    
    return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
      Rcpp::Named("weights") = weights);
  } break;
  case 2: {
    bsm_ng model(model_, seed);
    model.approximate();
    arma::cube alpha = model.approx_model.simulate_states(nsim);
    model.update_scales();
    arma::vec weights = model.importance_weights(alpha) - arma::accu(model.scales);
    double maxw = weights.max();
    weights = arma::exp(weights - maxw);
    
    return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
      Rcpp::Named("weights") = weights);
  } break;
  case 3: {
    svm model(model_, seed);
    model.approximate();
    arma::cube alpha = model.approx_model.simulate_states(nsim);
    model.update_scales();
    arma::vec weights = model.importance_weights(alpha) - arma::accu(model.scales);
    double maxw = weights.max();
    weights = arma::exp(weights - maxw);
    
    return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
      Rcpp::Named("weights") = weights);
  } break;
  case 4: {
    ar1_ng model(model_, seed);
    model.approximate();
    arma::cube alpha = model.approx_model.simulate_states(nsim);
    model.update_scales();
    arma::vec weights = model.importance_weights(alpha) - arma::accu(model.scales);
    double maxw = weights.max();
    weights = arma::exp(weights - maxw);
    
    return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
      Rcpp::Named("weights") = weights);
  } break;
  default:
    return Rcpp::List::create(Rcpp::Named("alpha") = 0, Rcpp::Named("weights") = 0);
  }
}
