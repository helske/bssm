// #include "model_ssm_ulg.h"
// #include "model_ssm_ung.h"
// #include "model_bsm_ng.h"
// #include "model_svm.h"
// #include "model_ar1_ng.h"
// 
// // [[Rcpp::export]]
// Rcpp::List importance_sample_ung(const Rcpp::List model_, 
//   unsigned int nsim, bool use_antithetic,
//   arma::vec mode_estimate, const unsigned int max_iter, 
//   const double conv_tol, const unsigned int seed, const int model_type) {
//   
//   switch (model_type) {
//   case 1: {
//     ssm_ung model(model_, seed);
//     ssm_ulg approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
//     arma::cube alpha = approx_model.simulate_states(nsim, use_antithetic);
//     arma::vec scales = model.update_scales();
//     arma::vec weights = model.importance_weights(approx_model, alpha);
//     weights = arma::exp(weights - arma::accu(scales));
//     
//     return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
//       Rcpp::Named("weights") = weights);
//   } break;
//   case 2: {
//     bsm_ng model(model_, seed);
//     ssm_ulg approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
//     arma::cube alpha = approx_model.simulate_states(nsim, use_antithetic);
//     arma::vec scales = model.update_scales();
//     arma::vec weights = model.importance_weights(approx_model, alpha);
//     weights = arma::exp(weights - arma::accu(scales));
//     return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
//       Rcpp::Named("weights") = weights);
//   } break;
//   case 3: {
//     svm model(model_, seed);
//     ssm_ulg approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
//     arma::cube alpha = approx_model.simulate_states(nsim, use_antithetic);
//      arma::vec scales = model.update_scales();
//      arma::vec weights = model.importance_weights(approx_model, alpha);
//      weights = arma::exp(weights - arma::accu(scales));
//     return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
//       Rcpp::Named("weights") = weights);
//   } break;
//   case 4: {
//     ar1_ng model(model_, seed);
//     ssm_ulg approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
//     arma::cube alpha = approx_model.simulate_states(nsim, use_antithetic);
//     arma::vec scales = model.update_scales();
//     arma::vec weights = model.importance_weights(approx_model, alpha);
//     weights = arma::exp(weights - arma::accu(scales));
//     return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
//       Rcpp::Named("weights") = weights);
//   } break;
//   default: 
//     return Rcpp::List::create(Rcpp::Named("alpha") = 0, Rcpp::Named("weights") = 0);
//   }
// }
