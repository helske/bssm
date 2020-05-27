#include "mcmc.h"
// #include "ung_amcmc.h"
// #include "nlg_amcmc.h"

#include "model_ugg_bsm.h"
#include "model_ugg_bsm.h"
#include "model_ugg_ar1.h"
#include "model_ung_bsm.h"
#include "model_ung_svm.h"
#include "model_ung_ar1.h"
#include "model_nlg_ssm.h"
#include "model_lgg_ssm.h"
#include "model_mng_ssm.h"
//#include "summary.h"

// [[Rcpp::export]]
Rcpp::List gaussian_mcmc(const Rcpp::List& model_,
  const unsigned int type, const unsigned int n_iter, const unsigned int n_burnin,
  const unsigned int n_thin, const double gamma, const double target_acceptance,
  const arma::mat S, const unsigned int seed, const bool end_ram,
  const unsigned int n_threads, const int model_type) {
  
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
  
  mcmc mcmc_run(n_iter, n_burnin, n_thin, n, m,
    target_acceptance, gamma, S);
  
  switch (model_type) {
  case 1: {
    ugg_ssm model(Rcpp::clone(model_), seed);
    mcmc_run.mcmc_gaussian(model, end_ram);
    switch (type) {
    case 1: {
      mcmc_run.state_posterior(model, n_threads); //sample states
      return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
        Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("counts") = mcmc_run.count_storage,
        Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
        Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
    } break;
    case 2: {
      //summary
      arma::mat alphahat(m, n + 1);
      arma::cube Vt(m, m, n + 1);
      mcmc_run.state_summary(model, alphahat, Vt);
      return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("alphahat") = alphahat.t(), Rcpp::Named("Vt") = Vt,
        Rcpp::Named("counts") = mcmc_run.count_storage,
        Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
        Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
    } break;
    case 3: {
      //marginal of theta
      return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("counts") = mcmc_run.count_storage,
        Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
        Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
    } break;
    }
  }break;
  case 2: {
    ugg_bsm model(Rcpp::clone(model_), seed);
    mcmc_run.mcmc_gaussian(model, end_ram);
    switch (type) {
    case 1: {
      mcmc_run.state_posterior(model, n_threads); //sample states
      return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
        Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("counts") = mcmc_run.count_storage,
        Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
        Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
    } break;
    case 2: {
      //summary
      arma::mat alphahat(m, n + 1);
      arma::cube Vt(m, m, n + 1);
      mcmc_run.state_summary(model, alphahat, Vt);
      return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("alphahat") = alphahat.t(), Rcpp::Named("Vt") = Vt,
        Rcpp::Named("counts") = mcmc_run.count_storage,
        Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
        Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
    } break;
    case 3: {
      //marginal of theta
      return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("counts") = mcmc_run.count_storage,
        Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
        Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
    } break;
    }
  } break;
  case 3: {
    ugg_ar1 model(Rcpp::clone(model_), seed);
    mcmc_run.mcmc_gaussian(model, end_ram);
    switch (type) {
    case 1: {
      mcmc_run.state_posterior(model, n_threads); //sample states
      return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
        Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("counts") = mcmc_run.count_storage,
        Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
        Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
    } break;
    case 2: {
      //summary
      arma::mat alphahat(m, n + 1);
      arma::cube Vt(m, m, n + 1);
      mcmc_run.state_summary(model, alphahat, Vt);
      return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("alphahat") = alphahat.t(), Rcpp::Named("Vt") = Vt,
        Rcpp::Named("counts") = mcmc_run.count_storage,
        Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
        Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
    } break;
    case 3: {
      //marginal of theta
      return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("counts") = mcmc_run.count_storage,
        Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
        Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
    } break;
    }
  } break;
  }
  return Rcpp::List::create(Rcpp::Named("error") = "error");
}

// [[Rcpp::export]]
Rcpp::List general_gaussian_mcmc(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP a1, SEXP P1,
  const arma::vec& theta,
  SEXP D, SEXP C,
  SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const arma::uvec& time_varying,
  const unsigned int n_states, const unsigned int n_etas,
  const unsigned int seed, const unsigned int n_iter,
  const unsigned int n_burnin, const unsigned int n_thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const bool end_ram, const unsigned int n_threads, const unsigned int type) {

  Rcpp::XPtr<lmat_fnPtr> xpfun_Z(Z);
  Rcpp::XPtr<lmat_fnPtr> xpfun_H(H);
  Rcpp::XPtr<lmat_fnPtr> xpfun_T(T);
  Rcpp::XPtr<lmat_fnPtr> xpfun_R(R);
  Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
  Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
  Rcpp::XPtr<lvec_fnPtr> xpfun_D(D);
  Rcpp::XPtr<lvec_fnPtr> xpfun_C(C);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);

  lgg_ssm model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_a1, *xpfun_P1,
    *xpfun_D, *xpfun_C, theta, *xpfun_prior, known_params, known_tv_params,
    time_varying, n_states, n_etas, seed);

  mcmc mcmc_run(n_iter, n_burnin, n_thin,
    model.n, model.m, target_acceptance, gamma, S, type);

  // mcmc_run.mcmc_gaussian(model, end_ram);
  // if(type == 1) mcmc_run.state_posterior(model, n_threads);

  if(type == 1) {
    return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } else {
    return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  }
  return Rcpp::List::create(Rcpp::Named("theta") = 1);
}
// [[Rcpp::export]]
Rcpp::List nongaussian_pm_mcmc(const Rcpp::List& model_,
  const unsigned int type,
  const unsigned int nsim_states, const unsigned int n_iter,
  const unsigned int n_burnin, const unsigned int n_thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const unsigned int seed, const bool end_ram, const unsigned int n_threads,
  const unsigned int simulation_method, const int model_type) {

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

  mcmc mcmc_run(n_iter, n_burnin, n_thin, n, m,
    target_acceptance, gamma, S, type);

  switch (model_type) {
  case 1: {
    ung_ssm model(Rcpp::clone(model_), seed);
    mcmc_run.pm_mcmc(model, simulation_method, nsim_states, end_ram);
  } break;
  case 2: {
    ung_bsm model(Rcpp::clone(model_), seed);
    mcmc_run.pm_mcmc(model, simulation_method, nsim_states, end_ram);
  } break;
  case 3: {
    ung_svm model(Rcpp::clone(model_), seed);
    mcmc_run.pm_mcmc(model, simulation_method, nsim_states, end_ram);
  } break;
  case 4: {
    ung_ar1 model(Rcpp::clone(model_), seed);
    mcmc_run.pm_mcmc(model, simulation_method, nsim_states, end_ram);
  }
  }
  switch (type) {
  case 1: {
    return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 2: {
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 3: {
    return Rcpp::List::create(
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  }

  return Rcpp::List::create(Rcpp::Named("error") = "error");
}


// [[Rcpp::export]]
Rcpp::List nongaussian_da_mcmc(const Rcpp::List& model_,
  const unsigned int type,
  const unsigned int nsim_states, const unsigned int n_iter,
  const unsigned int n_burnin, const unsigned int n_thin, const double gamma,
  const double target_acceptance, const arma::mat S, const unsigned int seed,
  const bool end_ram, const unsigned int n_threads,
  const unsigned int simulation_method, const int model_type) {

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

  mcmc mcmc_run(n_iter, n_burnin, n_thin, n, m, target_acceptance, gamma, S, type);

  switch (model_type) {
  case 1: {
    ung_ssm model(Rcpp::clone(model_), seed);
    mcmc_run.da_mcmc(model, simulation_method, nsim_states, end_ram);
  } break;
  case 2: {
    ung_bsm model(Rcpp::clone(model_), seed);
    mcmc_run.da_mcmc(model, simulation_method, nsim_states, end_ram);
  } break;
  case 3: {
    ung_svm model(Rcpp::clone(model_), seed);
    mcmc_run.da_mcmc(model, simulation_method, nsim_states, end_ram);
  } break;
  case 4: {
    ung_ar1 model(Rcpp::clone(model_), seed);
    mcmc_run.da_mcmc(model, simulation_method, nsim_states, end_ram);
  } break;
  }

  switch (type) {
  case 1: {
    return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 2: {
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 3: {
    return Rcpp::List::create(
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  }

  return Rcpp::List::create(Rcpp::Named("error") = "error");
}


// // 
// // // [[Rcpp::export]]
// // Rcpp::List nongaussian_is_mcmc(const Rcpp::List& model_,
// //   const unsigned int type,
// //   const unsigned int nsim_states, const unsigned int n_iter,
// //   const unsigned int n_burnin, const unsigned int n_thin, const  double gamma,
// //   const double target_acceptance, const arma::mat S, const unsigned int seed,
// //   const bool end_ram, const unsigned int n_threads, const bool local_approx,
// //   const arma::vec initial_mode, const unsigned int max_iter, const double conv_tol,
// //   const unsigned int simulation_method, const unsigned int is_type, 
// //   const int model_type) {
// //   
// //   arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
// //   unsigned int m = a1.n_elem;
// //   unsigned int n;
// //   
// //   if(model_type > 0) {
// //     arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
// //     n = y.n_elem;
// //   } else {
// //     arma::vec y = Rcpp::as<arma::mat>(model_["y"]);
// //     n = y.n_rows;
// //   }
// //   
// //   ung_amcmc mcmc_run(n_iter, n_burnin, n_thin, n, m,
// //     target_acceptance, gamma, S, type, simulation_method != 2);
// //   if (nsim_states <= 1) {
// //     mcmc_run.alpha_storage.zeros();
// //     mcmc_run.weight_storage.ones();
// //     mcmc_run.posterior_storage.zeros();
// //   }
// //   switch (model_type) {
// //   case 1: {
// //     ung_ssm model(Rcpp::clone(model_), seed);
// //     mcmc_run.approx_mcmc(model, end_ram);
// //     if(nsim_states > 1) {
// //       if(is_type == 3) {
// //         mcmc_run.expand();
// //       }
// //       switch (simulation_method) {
// //       case 1:
// //         mcmc_run.is_correction_psi(model, nsim_states, is_type, n_threads);
// //         break;
// //       case 2:
// //         mcmc_run.is_correction_bsf(model, nsim_states, is_type, n_threads);
// //         break;
// //       case 3:
// //         mcmc_run.is_correction_spdk(model, nsim_states, is_type, n_threads);
// //         break;
// //       }
// //     } else {
// //       if(nsim_states == 1) mcmc_run.approx_state_posterior(model, n_threads);
// //     }
// //   } break;
// //   case 2: {
// //     ung_bsm model(Rcpp::clone(model_), seed);
// //     mcmc_run.approx_mcmc(model, end_ram);
// //     if(nsim_states > 1) {
// //       if(is_type == 3) {
// //         mcmc_run.expand();
// //       }
// //       switch (simulation_method) {
// //       case 1:
// //         mcmc_run.is_correction_psi(model, nsim_states, is_type, n_threads);
// //         break;
// //       case 2:
// //         mcmc_run.is_correction_bsf(model, nsim_states, is_type, n_threads);
// //         break;
// //       case 3:
// //         mcmc_run.is_correction_spdk(model, nsim_states, is_type, n_threads);
// //         break;
// //       }
// //     } else {
// //       if(nsim_states == 1) mcmc_run.approx_state_posterior(model, n_threads);
// //     }
// //   } break;
// //   case 3: {
// //     ung_svm model(Rcpp::clone(model_), seed);
// //     mcmc_run.approx_mcmc(model, end_ram);
// //     if(nsim_states > 1) {
// //       if(is_type == 3) {
// //         mcmc_run.expand();
// //       }
// //       switch (simulation_method) {
// //       case 1:
// //         mcmc_run.is_correction_psi(model, nsim_states, is_type, n_threads);
// //         break;
// //       case 2:
// //         mcmc_run.is_correction_bsf(model, nsim_states, is_type, n_threads);
// //         break;
// //       case 3:
// //         mcmc_run.is_correction_spdk(model, nsim_states, is_type, n_threads);
// //         break;
// //       }
// //     } else {
// //       if(nsim_states == 1) mcmc_run.approx_state_posterior(model, n_threads);
// //     }
// //   } break;  
// //   case 4: {
// //     ung_ar1 model(Rcpp::clone(model_), seed);
// //     mcmc_run.approx_mcmc(model, end_ram);
// //     if(nsim_states > 1) {
// //       if(is_type == 3) {
// //         mcmc_run.expand();
// //       }
// //       switch (simulation_method) {
// //       case 1:
// //         mcmc_run.is_correction_psi(model, nsim_states, is_type, n_threads);
// //         break;
// //       case 2:
// //         mcmc_run.is_correction_bsf(model, nsim_states, is_type, n_threads);
// //         break;
// //       case 3:
// //         mcmc_run.is_correction_spdk(model, nsim_states, is_type, n_threads);
// //         break;
// //       }
// //     } else {
// //       if(nsim_states == 1) mcmc_run.approx_state_posterior(model, n_threads);
// //     }
// //   } break;
// //   }
// //   
// //   switch (type) { 
// //   case 1: {
// //     return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
// //       Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
// //       Rcpp::Named("weights") = mcmc_run.weight_storage,
// //       Rcpp::Named("counts") = mcmc_run.count_storage,
// //       Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
// //       Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
// //   } break;
// //   case 2: {
// //     return Rcpp::List::create(
// //       Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
// //       Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
// //       Rcpp::Named("weights") = mcmc_run.weight_storage,
// //       Rcpp::Named("counts") = mcmc_run.count_storage,
// //       Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
// //       Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
// //   } break;
// //   case 3: {
// //     return Rcpp::List::create(
// //       Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
// //       Rcpp::Named("weights") = mcmc_run.weight_storage,
// //       Rcpp::Named("counts") = mcmc_run.count_storage,
// //       Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
// //       Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
// //   } break;
// //   }
// //   
// //   return Rcpp::List::create(Rcpp::Named("error") = "error");
// // }
// 
// // [[Rcpp::export]]
// Rcpp::List nonlinear_pm_mcmc(const arma::mat& y, SEXP Z, SEXP H,
//   SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
//   const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
//   const arma::mat& known_tv_params, const arma::uvec& time_varying,
//   const unsigned int n_states, const unsigned int n_etas,
//   const unsigned int seed, const unsigned int nsim_states, const unsigned int n_iter,
//   const unsigned int n_burnin, const unsigned int n_thin,
//   const double gamma, const double target_acceptance, const arma::mat S,
//   const bool end_ram, const unsigned int n_threads,
//   const unsigned int max_iter, const double conv_tol,
//   const unsigned int simulation_method, const unsigned int iekf_iter,
//   const unsigned int type) {
//   
//   
//   Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
//   Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
//   Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
//   Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
//   Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
//   Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
//   Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
//   Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
//   Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);
//   
//   nlg_ssm model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg, 
//     *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
//     time_varying, seed);
//   
//   mcmc mcmc_run(n_iter, n_burnin, n_thin, model.n,
//     model.m, target_acceptance, gamma, S, type);
//   mcmc_run.pm_mcmc(model, simulation_method, nsim_states, end_ram);
//   
//   switch (type) { 
//   case 1: {
//     return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
//       Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
//       Rcpp::Named("counts") = mcmc_run.count_storage,
//       Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
//       Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
//   } break;
//   case 2: {
//     return Rcpp::List::create(
//       Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
//       Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
//       Rcpp::Named("counts") = mcmc_run.count_storage,
//       Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
//       Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
//   } break;
//   case 3: {
//     return Rcpp::List::create(
//       Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
//       Rcpp::Named("counts") = mcmc_run.count_storage,
//       Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
//       Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
//   } break;
//   }
//   
//   return Rcpp::List::create(Rcpp::Named("error") = "error");
// }
// // [[Rcpp::export]]
// Rcpp::List nonlinear_da_mcmc(const arma::mat& y, SEXP Z, SEXP H,
//   SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
//   const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
//   const arma::mat& known_tv_params, const arma::uvec& time_varying,
//   const unsigned int n_states, const unsigned int n_etas,
//   const unsigned int seed, const unsigned int nsim_states, const unsigned int n_iter,
//   const unsigned int n_burnin, const unsigned int n_thin,
//   const double gamma, const double target_acceptance, const arma::mat S,
//   const bool end_ram, const unsigned int n_threads,
//   const unsigned int max_iter, const double conv_tol,
//   const unsigned int simulation_method, const unsigned int iekf_iter,
//   const unsigned int type) {
//   
//   
//   Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
//   Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
//   Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
//   Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
//   Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
//   Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
//   Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
//   Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
//   Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);
//   
//   nlg_ssm model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg, 
//     *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
//     time_varying, seed);
//   
//   mcmc mcmc_run(n_iter, n_burnin, n_thin, model.n,
//     model.m, target_acceptance, gamma, S, type);
//   mcmc_run.da_mcmc(model, simulation_method, nsim_states, end_ram);
// 
//   switch (type) { 
//   case 1: {
//     return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
//       Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
//       Rcpp::Named("counts") = mcmc_run.count_storage,
//       Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
//       Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
//   } break;
//   case 2: {
//     return Rcpp::List::create(
//       Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
//       Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
//       Rcpp::Named("counts") = mcmc_run.count_storage,
//       Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
//       Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
//   } break;
//   case 3: {
//     return Rcpp::List::create(
//       Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
//       Rcpp::Named("counts") = mcmc_run.count_storage,
//       Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
//       Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
//   } break;
//   }
//   
//   return Rcpp::List::create(Rcpp::Named("error") = "error");
// }
// 
// // // [[Rcpp::export]]
// // Rcpp::List nonlinear_ekf_mcmc(const arma::mat& y, SEXP Z, SEXP H,
// //   SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
// //   const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
// //   const arma::mat& known_tv_params, const arma::uvec& time_varying,
// //   const unsigned int n_states, const unsigned int n_etas,
// //   const unsigned int seed, const unsigned int n_iter,
// //   const unsigned int n_burnin, const unsigned int n_thin,
// //   const double gamma, const double target_acceptance, const arma::mat S,
// //   const bool end_ram, const unsigned int n_threads, 
// //   const unsigned int iekf_iter, const unsigned int type) {
// //   
// //   
// //   Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
// //   Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
// //   Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
// //   Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
// //   Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
// //   Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
// //   Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
// //   Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
// //   Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);
// //   
// //   nlg_ssm model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg, 
// //     *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
// //     time_varying, seed);
// //   
// //   nlg_amcmc mcmc_run(n_iter, n_burnin, n_thin, model.n,
// //     model.m, target_acceptance, gamma, S, type, false);
// //   
// //   mcmc_run.ekf_mcmc(model, end_ram, iekf_iter);
// //   
// //   if (type == 2) {
// //     
// //     arma::mat alphahat(model.m, model.n + 1);
// //     arma::cube Vt(model.m, model.m, model.n + 1);
// //     mcmc_run.state_ekf_summary(model, alphahat, Vt, iekf_iter);
// //     
// //     return Rcpp::List::create(Rcpp::Named("alphahat") = alphahat.t(), Rcpp::Named("Vt") = Vt,
// //       Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
// //       Rcpp::Named("counts") = mcmc_run.count_storage,
// //       Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
// //       Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
// //   } else {
// //     
// //     if (type == 1) {
// //       mcmc_run.state_ekf_sample(model, n_threads, iekf_iter);
// //       return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
// //         Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
// //         Rcpp::Named("counts") = mcmc_run.count_storage,
// //         Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
// //         Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
// //     } else {
// //       return Rcpp::List::create(
// //         Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
// //         Rcpp::Named("counts") = mcmc_run.count_storage,
// //         Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
// //         Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
// //     }
// //   }
// // }
// // 
// // // [[Rcpp::export]]
// // Rcpp::List nonlinear_is_mcmc(const arma::mat& y, SEXP Z, SEXP H,
// //   SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
// //   const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
// //   const arma::mat& known_tv_params, const arma::uvec& time_varying,
// //   const unsigned int n_states, const unsigned int n_etas,
// //   const unsigned int seed, const unsigned int nsim_states, const unsigned int n_iter,
// //   const unsigned int n_burnin, const unsigned int n_thin,
// //   const double gamma, const double target_acceptance, const arma::mat S,
// //   const bool end_ram, const unsigned int n_threads, const unsigned int is_type,
// //   const unsigned int simulation_method, const unsigned int max_iter,
// //   const double conv_tol, const unsigned int iekf_iter,
// //   const unsigned int type) {
// //   
// //   
// //   Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
// //   Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
// //   Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
// //   Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
// //   Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
// //   Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
// //   Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
// //   Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
// //   Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);
// //   
// //   nlg_ssm model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg, 
// //     *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
// //     time_varying, seed);
// //   
// //   nlg_amcmc mcmc_run(n_iter, n_burnin, n_thin, model.n,
// //     model.m, target_acceptance, gamma, S, type, simulation_method == 1);
// //   
// //   mcmc_run.approx_mcmc(model, max_iter, conv_tol, end_ram, iekf_iter);
// //   if(nsim_states > 0) {
// //     if (is_type == 3) {
// //       mcmc_run.expand();
// //     }
// //     if (simulation_method == 1) {
// //       mcmc_run.is_correction_psi(model, nsim_states, is_type, n_threads);
// //     } else {
// //       mcmc_run.is_correction_bsf(model, nsim_states, is_type, n_threads);
// //     }
// //   } else {
// //     mcmc_run.alpha_storage.zeros();
// //     mcmc_run.weight_storage.ones();
// //   }
// //   return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
// //     Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
// //     Rcpp::Named("weights") = mcmc_run.weight_storage,
// //     Rcpp::Named("counts") = mcmc_run.count_storage,
// //     Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
// //     Rcpp::Named("S") = mcmc_run.S,
// //     Rcpp::Named("posterior") = mcmc_run.posterior_storage);
// // }

