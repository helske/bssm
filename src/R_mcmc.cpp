#include "mcmc.h"
#include "approx_mcmc.h"

#include "model_ssm_mlg.h"
#include "model_ssm_mng.h"
#include "model_bsm_lg.h"
#include "model_bsm_ng.h"
#include "model_ar1_lg.h"
#include "model_ar1_ng.h"
#include "model_svm.h"
#include "model_ssm_nlg.h"

// [[Rcpp::export]]
Rcpp::List gaussian_mcmc(const Rcpp::List model_,
  const unsigned int output_type, const unsigned int iter, const unsigned int burnin,
  const unsigned int thin, const double gamma, const double target_acceptance,
  const arma::mat S, const unsigned int seed, const bool end_ram,
  const unsigned int n_threads, const int model_type, const bool verbose) {
  
  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;
  
  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::mat y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }
  mcmc mcmc_run(iter, burnin, thin, n, m,
    target_acceptance, gamma, S, output_type, verbose);
  
  switch (model_type) {
  case 0: {
    ssm_mlg model(model_, seed);
    mcmc_run.mcmc_gaussian(model, end_ram, 
      model_["update_fn"], model_["prior_fn"]);
    
    switch (output_type) {
    case 1: {
      mcmc_run.state_posterior(model, n_threads, model_["update_fn"]); //sample states
      
      return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
        Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("counts") = mcmc_run.count_storage,
        Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
        Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
    } break;
    case 2: {
      //summary
      mcmc_run.state_summary(model, model_["update_fn"]);
      return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
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
  case 1: {
    ssm_ulg model(model_, seed);
    mcmc_run.mcmc_gaussian(model, end_ram, 
      model_["update_fn"], model_["prior_fn"]);
    
    switch (output_type) {
    case 1: {
      
      mcmc_run.state_posterior(model, n_threads, model_["update_fn"]); //sample states
      
      return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
        Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("counts") = mcmc_run.count_storage,
        Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
        Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
    } break;
    case 2: {
      //summary
      mcmc_run.state_summary(model, model_["update_fn"]);
      return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
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
  case 2: {
    bsm_lg model(model_, seed);
    mcmc_run.mcmc_gaussian(model, end_ram);
    switch (output_type) {
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
      mcmc_run.state_summary(model);
      return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
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
    ar1_lg model(model_, seed);
    mcmc_run.mcmc_gaussian(model, end_ram);
    switch (output_type) {
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
      mcmc_run.state_summary(model);
      return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
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
Rcpp::List nongaussian_pm_mcmc(const Rcpp::List model_,
  const unsigned int output_type,
  const unsigned int nsim, const unsigned int iter,
  const unsigned int burnin, const unsigned int thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const unsigned int seed, const bool end_ram, const unsigned int n_threads,
  const unsigned int sampling_method, const unsigned int model_type, 
  const bool verbose) {
  
  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;
  
  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::mat y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }
  
  mcmc mcmc_run(iter, burnin, thin, n, m,
    target_acceptance, gamma, S, output_type, verbose);
  
  switch (model_type) {
  case 0: {
    ssm_mng model(model_, seed);
    mcmc_run.pm_mcmc(model, sampling_method, nsim, end_ram, 
      model_["update_fn"], model_["prior_fn"]);
  } break;
  case 1: {
    ssm_ung model(model_, seed);
    mcmc_run.pm_mcmc(model, sampling_method, nsim, end_ram, 
      model_["update_fn"], model_["prior_fn"]);
  } break;
  case 2: {
    bsm_ng model(model_, seed);
    mcmc_run.pm_mcmc(model, sampling_method, nsim, end_ram);
  } break;
  case 3: {
    svm model(model_, seed);
    mcmc_run.pm_mcmc(model, sampling_method, nsim, end_ram);
  } break;
  case 4: {
    ar1_ng model(model_, seed);
    mcmc_run.pm_mcmc(model, sampling_method, nsim, end_ram);
  }
  }
  switch (output_type) {
  case 1: {
    return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,
      Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 2: {
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,
      Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 3: {
    return Rcpp::List::create(
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  
      Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  }
  
  return Rcpp::List::create(Rcpp::Named("error") = "error");
}


// [[Rcpp::export]]
Rcpp::List nongaussian_da_mcmc(const Rcpp::List model_,
  const unsigned int output_type,
  const unsigned int nsim, const unsigned int iter,
  const unsigned int burnin, const unsigned int thin, const double gamma,
  const double target_acceptance, const arma::mat S, const unsigned int seed,
  const bool end_ram, const unsigned int n_threads,
  const unsigned int sampling_method, const int model_type, 
  const bool verbose) {
  
  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;
  
  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::mat y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }
  mcmc mcmc_run(iter, burnin, thin, n, m, target_acceptance, gamma, S, 
    output_type, verbose);
  
  switch (model_type) {
  case 0: {
    ssm_mng model(model_, seed);
    mcmc_run.da_mcmc(model, sampling_method, nsim, end_ram, 
      model_["update_fn"], model_["prior_fn"]);
  } break;
  case 1: {
    ssm_ung model(model_, seed);
    mcmc_run.da_mcmc(model, sampling_method, nsim, end_ram, 
      model_["update_fn"], model_["prior_fn"]);
  } break;
  case 2: {
    bsm_ng model(model_, seed);
    mcmc_run.da_mcmc(model, sampling_method, nsim, end_ram);
  } break;
  case 3: {
    svm model(model_, seed);
    mcmc_run.da_mcmc(model, sampling_method, nsim, end_ram);
  } break;
  case 4: {
    ar1_ng model(model_, seed);
    mcmc_run.da_mcmc(model, sampling_method, nsim, end_ram);
  } break;
  }
  
  switch (output_type) {
  case 1: {
    return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  
      Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 2: {
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  
      Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 3: {
    return Rcpp::List::create(
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  
      Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  }
  
  return Rcpp::List::create(Rcpp::Named("error") = "error");
}



// [[Rcpp::export]]
Rcpp::List nongaussian_is_mcmc(const Rcpp::List model_,
  const unsigned int output_type,
  const unsigned int nsim, const unsigned int iter,
  const unsigned int burnin, const unsigned int thin, const  double gamma,
  const double target_acceptance, const arma::mat S, const unsigned int seed,
  const bool end_ram, const unsigned int n_threads,
  const unsigned int sampling_method, const unsigned int is_type,
  const int model_type, const bool approx, const bool verbose) {
  
  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;
  unsigned int p;
  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
    p = 1;
  } else {
    arma::mat y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
    p = y.n_cols;
  }
  approx_mcmc mcmc_run(iter, burnin, thin, n, m, p,
    target_acceptance, gamma, S, output_type, true, verbose);
  
  if (nsim <= 1) {
    mcmc_run.alpha_storage.zeros();
    mcmc_run.weight_storage.ones();
    mcmc_run.posterior_storage.zeros();
  }
  
  switch (model_type) {
  case 0: {
    ssm_mng model(model_, seed);
    mcmc_run.amcmc(model, 1, end_ram,
      model_["update_fn"], model_["prior_fn"]);
    if(approx) {
      if(output_type == 1) {
        mcmc_run.approx_state_posterior(model, n_threads, model_["update_fn"]);
        
      } else {
        if(output_type == 2) {
          mcmc_run.approx_state_summary(model, model_["update_fn"]);
        }
      }
    } else {
      if(is_type == 3) {
        mcmc_run.expand();
      }
      switch (sampling_method) {
      case 1:
        mcmc_run.is_correction_psi(model, nsim, is_type, n_threads, model_["update_fn"]);
        break;
      case 2:
        mcmc_run.is_correction_bsf(model, nsim, is_type, n_threads, model_["update_fn"]);
        break;
      case 3:
        mcmc_run.is_correction_spdk(model, nsim, is_type, n_threads, model_["update_fn"]);
        break;
      }
      
    } 
  } break;
  case 1: {
    ssm_ung model(model_, seed);
    mcmc_run.amcmc(model, 1, end_ram, model_["update_fn"], model_["prior_fn"]);
    if(approx) {
      if(output_type == 1) {
        mcmc_run.approx_state_posterior(model, n_threads, model_["update_fn"]);
      } else {
        if(output_type == 2) {
          mcmc_run.approx_state_summary(model, model_["update_fn"]);
        }
      }
    } else {
      if(is_type == 3) {
        mcmc_run.expand();
      }
      
      switch (sampling_method) {
      case 1:
        mcmc_run.is_correction_psi(model, nsim, is_type, n_threads, model_["update_fn"]);
        break;
      case 2:
        mcmc_run.is_correction_bsf(model, nsim, is_type, n_threads, model_["update_fn"]);
        break;
      case 3:
        mcmc_run.is_correction_spdk(model, nsim, is_type, n_threads, model_["update_fn"]);
        break;
      }
      
    } 
  } break;
  case 2: {
    
    bsm_ng model(model_, seed);
    mcmc_run.amcmc(model, 1, end_ram);
    if(approx) {
      if(output_type == 1) {
        mcmc_run.approx_state_posterior(model, n_threads);
      } else {
        if(output_type == 2) {
          mcmc_run.approx_state_summary(model);
        }
      }
    } else {
      if(is_type == 3) {
        
        mcmc_run.expand();
      }
      switch (sampling_method) {
      case 1:
        mcmc_run.is_correction_psi(model, nsim, is_type, n_threads);
        break;
      case 2:
        mcmc_run.is_correction_bsf(model, nsim, is_type, n_threads);
        break;
      case 3:
        mcmc_run.is_correction_spdk(model, nsim, is_type, n_threads);
        break;
      }
    } 
  } break;
  case 3: {
    svm model(model_, seed);
    mcmc_run.amcmc(model, 1, end_ram);
    if(approx) {
      if(output_type == 1) {
        mcmc_run.approx_state_posterior(model, n_threads);
      } else {
        if(output_type == 2) {
          mcmc_run.approx_state_summary(model);
        }
      }
    } else {
      if(is_type == 3) {
        mcmc_run.expand();
      }
      switch (sampling_method) {
      case 1:
        mcmc_run.is_correction_psi(model, nsim, is_type, n_threads);
        break;
      case 2:
        mcmc_run.is_correction_bsf(model, nsim, is_type, n_threads);
        break;
      case 3:
        mcmc_run.is_correction_spdk(model, nsim, is_type, n_threads);
        break;
      }
    } 
  } break;
  case 4: {
    ar1_ng model(model_, seed);
    mcmc_run.amcmc(model, 1, end_ram);
    if(approx) {
      if(output_type == 1) {
        mcmc_run.approx_state_posterior(model, n_threads);
      } else {
        if(output_type == 2) {
          mcmc_run.approx_state_summary(model);
        }
      }
    } else {
      if(is_type == 3) {
        mcmc_run.expand();
      }
      switch (sampling_method) {
      case 1:
        mcmc_run.is_correction_psi(model, nsim, is_type, n_threads);
        break;
      case 2:
        mcmc_run.is_correction_bsf(model, nsim, is_type, n_threads);
        break;
      case 3:
        mcmc_run.is_correction_spdk(model, nsim, is_type, n_threads);
        break;
      }
    } 
  } break;
  }
  
  switch (output_type) {
  case 1: {
    return Rcpp::List::create(
      Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S, 
      Rcpp::Named("posterior") = mcmc_run.posterior_storage,
      Rcpp::Named("modes") = mcmc_run.mode_storage);
  } break;
  case 2: {
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S, 
      Rcpp::Named("posterior") = mcmc_run.posterior_storage,
      Rcpp::Named("modes") = mcmc_run.mode_storage);
  } break;
  case 3: {
    return Rcpp::List::create(
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S, 
      Rcpp::Named("posterior") = mcmc_run.posterior_storage,
      Rcpp::Named("modes") = mcmc_run.mode_storage);
  } break;
  }
  
  return Rcpp::List::create(Rcpp::Named("error") = "error");
}

// [[Rcpp::export]]
Rcpp::List nonlinear_pm_mcmc(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const arma::uvec& time_varying,
  const unsigned int n_states, const unsigned int n_etas,
  const unsigned int seed, const unsigned int nsim, const unsigned int iter,
  const unsigned int burnin, const unsigned int thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const bool end_ram, const unsigned int n_threads,
  const unsigned int max_iter, const double conv_tol,
  const unsigned int sampling_method, const unsigned int iekf_iter,
  const unsigned int output_type, const bool verbose) {
  
  
  Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
  Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
  Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
  Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
  Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
  Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);
  
  ssm_nlg model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg,
    *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);
  
  mcmc mcmc_run(iter, burnin, thin, model.n,
    model.m, target_acceptance, gamma, S, output_type, verbose);
  mcmc_run.pm_mcmc(model, sampling_method, nsim, end_ram);
  
  switch (output_type) {
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
Rcpp::List nonlinear_da_mcmc(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const arma::uvec& time_varying,
  const unsigned int n_states, const unsigned int n_etas,
  const unsigned int seed, const unsigned int nsim, const unsigned int iter,
  const unsigned int burnin, const unsigned int thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const bool end_ram, const unsigned int n_threads,
  const unsigned int max_iter, const double conv_tol,
  const unsigned int sampling_method, const unsigned int iekf_iter,
  const unsigned int output_type, const bool verbose) {
  
  
  Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
  Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
  Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
  Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
  Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
  Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);
  
  ssm_nlg model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg,
    *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);
  
  mcmc mcmc_run(iter, burnin, thin, model.n,
    model.m, target_acceptance, gamma, S, output_type, verbose);
  mcmc_run.da_mcmc(model, sampling_method, nsim, end_ram);
  
  switch (output_type) {
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
Rcpp::List nonlinear_ekf_mcmc(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const arma::uvec& time_varying,
  const unsigned int n_states, const unsigned int n_etas,
  const unsigned int seed, const unsigned int iter,
  const unsigned int burnin, const unsigned int thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const bool end_ram, const unsigned int n_threads,
  const unsigned int iekf_iter, const unsigned int output_type, 
  const bool verbose) {
  
  
  Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
  Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
  Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
  Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
  Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
  Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);
  
  ssm_nlg model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg,
    *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);
  
  approx_mcmc mcmc_run(iter, burnin, thin, model.n,
    model.m, model.m, target_acceptance, gamma, S, output_type, false, verbose);
  
  mcmc_run.ekf_mcmc(model, end_ram);
  
  if (output_type == 2) {
    
    mcmc_run.ekf_state_summary(model);
    
    return Rcpp::List::create(Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), 
      Rcpp::Named("Vt") = mcmc_run.Vt,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } else {
    if (output_type == 1) {
      
      mcmc_run.ekf_state_sample(model, n_threads);
      
      return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
        Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("counts") = mcmc_run.count_storage,
        Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
        Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
    } else {
      
      return Rcpp::List::create(
        Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
        Rcpp::Named("counts") = mcmc_run.count_storage,
        Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
        Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
    }
  }
}

// [[Rcpp::export]]
Rcpp::List nonlinear_is_mcmc(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const arma::uvec& time_varying,
  const unsigned int n_states, const unsigned int n_etas,
  const unsigned int seed, const unsigned int nsim, const unsigned int iter,
  const unsigned int burnin, const unsigned int thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const bool end_ram, const unsigned int n_threads, const unsigned int is_type,
  const unsigned int sampling_method, const unsigned int max_iter,
  const double conv_tol, const unsigned int iekf_iter,
  const unsigned int output_type,
  const bool approx, const bool verbose) {
  
  Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
  Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
  Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
  Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
  Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
  Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);

  ssm_nlg model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg,
    *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed, iekf_iter, max_iter, conv_tol);

  approx_mcmc mcmc_run(iter, burnin, thin, model.n,
    model.m, model.m, target_acceptance, gamma, S, output_type, true, verbose);

  if (nsim <= 1) {
    mcmc_run.alpha_storage.zeros();
    mcmc_run.weight_storage.ones();
    mcmc_run.posterior_storage.zeros();
  }
  
  mcmc_run.amcmc(model, sampling_method, end_ram);

  if(approx) {
    if(output_type == 1) {
      mcmc_run.approx_state_posterior(model, n_threads);
    } else {
      if(output_type == 2) {
        mcmc_run.approx_state_summary(model);
      }
    }
  } else {
    if (is_type == 3) {
      mcmc_run.expand();
    }
    if (sampling_method == 1) {
      mcmc_run.is_correction_psi(model, nsim, is_type, n_threads);
    } else {
      mcmc_run.is_correction_bsf(model, nsim, is_type, n_threads);
    }
  } 
  
  switch (output_type) {
  case 1: {
    return Rcpp::List::create(
      Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S, 
      Rcpp::Named("posterior") = mcmc_run.posterior_storage,
      Rcpp::Named("modes") = mcmc_run.mode_storage);
  } break;
  case 2: {
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S, 
      Rcpp::Named("posterior") = mcmc_run.posterior_storage,
      Rcpp::Named("modes") = mcmc_run.mode_storage);
  } break;
  case 3: {
    return Rcpp::List::create(
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S, 
      Rcpp::Named("posterior") = mcmc_run.posterior_storage,
      Rcpp::Named("modes") = mcmc_run.mode_storage);
  } break;
  }
  
  return Rcpp::List::create(Rcpp::Named("error") = "error");
}
