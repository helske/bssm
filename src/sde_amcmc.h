// #ifndef SDE_AMCMC_H
// #define SDE_AMCMC_H
// 
// #include "bssm.h"
// #include "mcmc.h"
// 
// class sde_amcmc: public mcmc {
//   
// public:
//   
//   // constructor
//   sde_amcmc(const unsigned int n_iter, const unsigned int n_burnin,
//     const unsigned int n_thin, const unsigned int n,
//     const double target_acceptance, const double gamma, 
//     const arma::mat& S, const unsigned int output_type);
//   
//   void expand();
//   
//   void approx_mcmc(sde_ssm model, const bool end_ram, 
//     const unsigned int nsim, const unsigned int L_c);
//   
//   void is_correction_bsf(sde_ssm model, const unsigned int nsim, 
//     const unsigned int L_c, const unsigned int L_f, 
//     const unsigned int is_type, const unsigned int n_threads);
//   
//   arma::vec weight_storage;
//   arma::vec approx_loglik_storage;
//   arma::vec prior_storage;
// private:
//   
//   void trim_storage();
// };
// 
// 
// #endif
