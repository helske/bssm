#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "ung_ar1.h"

// [[Rcpp::export]]
Rcpp::List importance_sample_ung(const Rcpp::List& model_, 
  unsigned int nsim_states, bool use_antithetic,
  arma::vec mode_estimate, const unsigned int max_iter, 
  const double conv_tol, const unsigned int seed, const int model_type) {
  
  switch (model_type) {
  case 1: {
    ung_ssm model(clone(model_), seed);
    ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
    arma::cube alpha = approx_model.simulate_states(nsim_states, use_antithetic);
   
    arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
    arma::vec weights(nsim_states, arma::fill::zeros);
    for (unsigned int t = 0; t < model.n; t++) {
      weights += model.log_weights(approx_model, t, alpha);
    }
    
    weights = arma::exp(weights - arma::accu(scales));
    
    return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
      Rcpp::Named("weights") = weights);
  } break;
  case 2: {
    ung_bsm model(clone(model_), seed);
    ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
    arma::cube alpha = approx_model.simulate_states(nsim_states, use_antithetic);
    
    arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
    arma::vec weights(nsim_states, arma::fill::zeros);
    for (unsigned int t = 0; t < model.n; t++) {
      weights += model.log_weights(approx_model, t, alpha);
    }
    weights = arma::exp(weights - arma::accu(scales));
    return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
      Rcpp::Named("weights") = weights);
  } break;
  case 3: {
    ung_svm model(clone(model_), seed);
    Rcpp::Rcout<<"approx:"<<std::endl;
    ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
    Rcpp::Rcout<<"simulate:"<<std::endl;
    arma::cube alpha = approx_model.simulate_states(nsim_states, use_antithetic);
    Rcpp::Rcout<<"scaling factors:"<<std::endl;
    arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
    Rcpp::Rcout<<"weights:"<<std::endl;
    arma::vec weights(nsim_states, arma::fill::zeros);
    for (unsigned int t = 0; t < model.n; t++) {
      weights += model.log_weights(approx_model, t, alpha);
    }
    Rcpp::Rcout<<"exp:"<<std::endl;
    weights = arma::exp(weights - arma::accu(scales));
    Rcpp::Rcout<<"return:"<<std::endl;
    return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
      Rcpp::Named("weights") = weights);
  } break;
  case 4: {
    ung_ar1 model(clone(model_), seed);
    ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
    arma::cube alpha = approx_model.simulate_states(nsim_states, use_antithetic);
    
    arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
    arma::vec weights(nsim_states, arma::fill::zeros);
    for (unsigned int t = 0; t < model.n; t++) {
      weights += model.log_weights(approx_model, t, alpha);
    }
    weights = arma::exp(weights - arma::accu(scales));
    return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
      Rcpp::Named("weights") = weights);
  } break;
  default: 
    return Rcpp::List::create(Rcpp::Named("alpha") = 0, Rcpp::Named("weights") = 0);
  }
}
