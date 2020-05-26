// #ifndef NLG_AMCMC_H
// #define NLG_AMCMC_H
// 
// #include "bssm.h"
// #include "mcmc.h"
// 
// class nlg_ssm;
// 
// class nlg_amcmc: public mcmc {
//   
// public:
//   
//   // constructor
//   nlg_amcmc(const unsigned int n_iter, const unsigned int n_burnin, const unsigned int n_thin, 
//     const unsigned int n, const unsigned int m, const double target_acceptance, 
//     const double gamma, const arma::mat& S, const unsigned int output_type, 
//     const bool store_modes);
//   
//   void expand();
//   
//   void approx_mcmc(nlg_ssm model, const bool end_ram);
//   
//   void ekf_mcmc(nlg_ssm model, const bool end_ram);
//   
//   void is_correction_bsf(nlg_ssm model, const unsigned int nsim_states, 
//     const unsigned int is_type, const unsigned int n_threads);
//   
//  
//   void is_correction_psi(nlg_ssm model, const unsigned int nsim_states, 
//     const unsigned int is_type, const unsigned int n_threads);
//   
//   void state_ekf_sample(nlg_ssm model, const unsigned int n_threads, const unsigned int iekf_iter);
//   
//   void state_ekf_summary(nlg_ssm& model, arma::mat& alphahat, arma::cube& Vt, const unsigned int iekf_iter);
//     
//   arma::vec weight_storage;
//   
// private:
//   
//   void trim_storage();
//   arma::vec approx_loglik_storage;
//   arma::vec scales_storage;
//   arma::vec prior_storage;
//   const bool store_modes;
//   arma::cube mode_storage;
// };
// 
// 
// #endif
