// #include "ng_psi_filter.h"
// 
// #include "model_ssm_ung.h"
// #include "model_bsm_ng.h"
// #include "model_svm.h"
// #include "model_ar1_ng.h"
// #include "distr_consts.h"
// 
// template double compute_ung_psi_filter(ssm_ung model, const unsigned int nsim, 
//   arma::vec mode_estimate, const unsigned int max_iter, const double conv_tol,
//   arma::cube& alpha, arma::mat& weights, arma::umat& indices);
// template double compute_ung_psi_filter(bsm_ng model, const unsigned int nsim, 
//   arma::vec mode_estimate, const unsigned int max_iter, const double conv_tol,
//   arma::cube& alpha, arma::mat& weights, arma::umat& indices);
// template double compute_ung_psi_filter(svm model, const unsigned int nsim, 
//   arma::vec mode_estimate, const unsigned int max_iter, const double conv_tol,
//   arma::cube& alpha, arma::mat& weights, arma::umat& indices);
// template double compute_ung_psi_filter(ar1_ng model, const unsigned int nsim, 
//   arma::vec mode_estimate, const unsigned int max_iter, const double conv_tol,
//   arma::cube& alpha, arma::mat& weights, arma::umat& indices);
// 
// template<class T>
// double compute_ung_psi_filter(T model, const unsigned int nsim, 
//   arma::vec mode_estimate, const unsigned int max_iter, const double conv_tol,
//   arma::cube& alpha, arma::mat& weights, arma::umat& indices) {
//   
//   ssm_ulg approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
//   // compute the log-likelihood of the approximate model
//   
//   double gaussian_loglik = approx_model.log_likelihood();
//   // compute unnormalized mode-based correction terms 
//   // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
//   arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
//   // compute the constant term
//   double const_term = compute_const_term(model, approx_model); 
//   // log-likelihood approximation
//   double approx_loglik = gaussian_loglik + const_term + arma::accu(scales);
//   
//   double loglik = model.psi_filter(approx_model, approx_loglik, scales, 
//     nsim, alpha, weights, indices);
//   return loglik;
// }
// 
