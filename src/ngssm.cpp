#include "ngssm.h"


//general constructor
ngssm::ngssm(arma::vec y, arma::mat Z, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, arma::vec phi, arma::mat xreg,
  arma::vec beta, unsigned int distribution, unsigned int seed) :
  gssm(y, Z, arma::vec(y.n_elem), T, R, a1, P1, xreg, beta, seed),
  phi(phi), distribution(distribution), ng_y(y), max_iter(100), conv_tol(1.0e-8) {
}

//general constructor with parameter indices
ngssm::ngssm(arma::vec y, arma::mat Z, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, arma::vec phi, arma::mat xreg,
  arma::vec beta, unsigned int distribution, arma::uvec Z_ind,
  arma::uvec T_ind, arma::uvec R_ind, unsigned int seed) :
  gssm(y, Z, arma::vec(y.n_elem), T, R, a1, P1, xreg, beta, Z_ind,
    arma::umat(0,0), T_ind, R_ind, seed), phi(phi),
    distribution(distribution), ng_y(y), max_iter(100), conv_tol(1.0e-8) {
}


double ngssm::proposal(const arma::vec& theta, const arma::vec& theta_prop) {
  return 0.0;
}

// find approximating Gaussian model
double ngssm::approx(arma::vec& signal, unsigned int max_iter, double conv_tol) {
  
  // signal_t = Z_t * alpha_t
  arma::mat Kt(m, n, arma::fill::zeros);
  arma::vec Ft(n, arma::fill::zeros);
  // log[p(signal)] + log[p(y | signal)]
  double ll = logp_signal(signal, Kt, Ft) + logp_y(signal);
  
  unsigned int i = 0;
  while(i < max_iter) {
    // compute new guess of signal
    arma::vec signal_new = approx_iter(signal);
    //log[p(signal)] + log[p(y | signal)]
    double ll_new = precomp_logp_signal(signal_new, Kt, Ft) +
      logp_y(signal_new);
    double diff = std::abs(ll_new - ll)/(0.1 + std::abs(ll_new));
    // Rcout<<arma::mean(arma::square(signal-signal_new))<<std::endl;
    // Rcout<<"diff "<<diff<<std::endl;
    signal = signal_new;
    if(!std::isfinite(ll_new) || signal.has_nan()){
      return -arma::datum::inf;
    }
    if (diff < conv_tol) {
      break;
    } else {
      ll = ll_new;
    }
    
    i++;
  }
  ll = 0.0;
  // log[g(pseudo_y | signal)]
  if (distribution != 0) {
    for (unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(ng_y(t))) {
        ll += R::dnorm(y(t), signal(t) + xbeta(t), H(t), 1);
      }
    }
  } else {
    for (unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(ng_y(t))) {
        ll += R::dnorm(y(t), signal(t), H(t), 1);
      }
    }
  }
  // log[p(y | signal)] - log[g(pseudo_y | signal)]
  return logp_y(signal) - ll;
}

// compute new values of pseudo y and H given the signal
// and the new signal using Kalman smoothing
arma::vec ngssm::approx_iter(arma::vec& signal) {
  
  // new pseudo y and H
  switch(distribution) {
  case 1  :
    HH = (1.0 / (exp(signal + xbeta) % phi));
    y = ng_y % HH + signal + xbeta - 1.0;
    break;
  case 2  :
    HH = pow(1.0 + exp(signal + xbeta), 2) / (phi % exp(signal + xbeta));
    y = ng_y % HH + signal + xbeta - 1.0 - exp(signal + xbeta);
    break;
  case 3  :
    HH = 1.0 / phi + 1.0 / exp(signal + xbeta);
    y = signal + xbeta + ng_y / exp(signal + xbeta) - 1.0;
    break;
  }
  
  // new signal
  
  arma::mat alpha = fast_smoother(true);
  arma::vec signal_new(n);
  
  for (unsigned int t = 0; t < n; t++) {
    signal_new(t) = arma::as_scalar(Z.col(Ztv * t).t() * alpha.col(t));
  }
  H = sqrt(HH);
  
  return signal_new;
}


// compute log[p(signal)] for the first time and store Kt and Ft which does not depend
// on the signal (observation vector)
double ngssm::logp_signal(arma::vec& signal, arma::mat& Kt, arma::vec& Ft) {
  
  double logLik = 0.0;
  
  arma::vec at = a1;
  arma::mat Pt = P1;
  
  for (unsigned int t = 0; t < n; t++) {
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv));
    if (Ft(t) > arma::datum::eps) { // can be zero if P1 is zero
      double v = arma::as_scalar(signal(t) - Z.col(t * Ztv).t() * at);
      Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
      at = T.slice(t * Ttv) * (at + Kt.col(t) * v);
      Pt = arma::symmatu(T.slice(t * Ttv) * (Pt - Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
      logLik -= 0.5 * (LOG2PI + log(Ft(t)) + v * v / Ft(t));
    } else {
      at = T.slice(t * Ttv) * at;
      Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  
  return logLik;
}

// fast computation of log[p(signal)] using the precomputed Kt and Ft
double ngssm::precomp_logp_signal(arma::vec& signal, const arma::mat& Kt, const arma::vec& Ft) {
  
  
  double logLik = 0.0;
  
  arma::vec at = a1;
  
  for (unsigned int t = 0; t < n; t++) {
    if (Ft(t) > arma::datum::eps) {
      double v = arma::as_scalar(signal(t) - Z.col(t * Ztv).t() * at);
      at = T.slice(t * Ttv) * (at + Kt.col(t) * v);
      logLik -= 0.5 * (LOG2PI + log(Ft(t)) + v * v / Ft(t));
    } else {
      at = T.slice(t * Ttv) * at;
    }
  }
  
  return logLik;
}

// log[p(y | signal)]
double ngssm::logp_y(arma::vec& signal) {
  
  double logp = 0.0;
  
  switch(distribution) {
  case 1  :
    for (unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(ng_y(t))) {
        logp += R::dpois(ng_y(t), phi(t) * exp(signal(t) + xbeta(t)), 1);
      }
    }
    break;
  case 2  :
    for (unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(ng_y(t))) {
        double exptmp = exp(signal(t) + xbeta(t));
        logp += R::dbinom(ng_y(t), phi(t),  exptmp / (1.0 + exptmp), 1);
      }
    }
    break;
  case 3  :
    for (unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(ng_y(t))) {
        logp += R::dnbinom_mu(ng_y(t), phi(t), exp(signal(t) + xbeta(t)), 1);
      }
    }
    break;
  }
  return logp;
}


// update system matrices given theta
void ngssm::update_model(arma::vec theta) {
  
  // !! add phi when adding other distributions !!
  //
  if (Z_ind.n_elem > 0) {
    Z.elem(Z_ind) = theta.subvec(0, Z_ind.n_elem - 1);
  }
  if (T_ind.n_elem > 0) {
    T.elem(T_ind) = theta.subvec(Z_ind.n_elem + H_ind.n_elem,
      Z_ind.n_elem + T_ind.n_elem - 1);
  }
  if (R_ind.n_elem > 0) {
    R.elem(R_ind) = theta.subvec(Z_ind.n_elem + H_ind.n_elem + T_ind.n_elem,
      Z_ind.n_elem + T_ind.n_elem + R_ind.n_elem - 1);
  }
  
  if (R_ind.n_elem  > 0) {
    compute_RR();
  }
  
  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols - (distribution == 3),
      theta.n_elem - 1 - (distribution == 3));
    compute_xbeta();
  }
  if(distribution == 3) {
    phi.fill(exp(theta(theta.n_elem - 1)));
  }
}

// pick up theta from system matrices
arma::vec ngssm::get_theta(void) {
  
  // !! add phi when adding other distributions !!
  arma::vec theta(Z_ind.n_elem + T_ind.n_elem + R_ind.n_elem + (distribution == 3));
  
  if (Z_ind.n_elem > 0) {
    theta.subvec(0, Z_ind.n_elem - 1) = Z.elem(Z_ind);
  }
  if (T_ind.n_elem > 0) {
    theta.subvec(Z_ind.n_elem + H_ind.n_elem,
      Z_ind.n_elem + T_ind.n_elem - 1) = T.elem(T_ind);
  }
  if (R_ind.n_elem > 0) {
    theta.subvec(Z_ind.n_elem + H_ind.n_elem + T_ind.n_elem,
      Z_ind.n_elem + T_ind.n_elem + R_ind.n_elem - 1) =
        R.elem(R_ind);
  }
  
  if(xreg.n_cols > 0) {
    theta.subvec(theta.n_elem - xreg.n_cols - (distribution == 3),
      theta.n_elem - 1 - (distribution == 3)) = beta;
  }
  
  if(distribution == 3) {
    theta(theta.n_elem - 1) = log(phi(0));
  }
  return theta;
}




arma::mat ngssm::predict2(arma::vec theta_lwr,
  arma::vec theta_upr, unsigned int n_iter, unsigned int nsim_states,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat S, unsigned int n_ahead,
  unsigned int interval, arma::vec init_signal) {
  
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  
  arma::mat pred_store(n_ahead, nsim_states * n_samples);
  
  unsigned int npar = theta_lwr.n_elem;
  arma::vec theta = get_theta();
  arma::vec signal = init_signal;
  double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
  double ll = ll_approx + log_likelihood(distribution != 0);
  
  arma::cube alpha_pred(m, n_ahead, nsim_states);
  double ll_w = 0;
  if (nsim_states > 1) {
    arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
    arma::vec weights = exp(importance_weights(alpha) - scaling_factor(signal));
    ll_w = log(sum(weights) / nsim_states);
    // sample from p(alpha | y)
    std::discrete_distribution<> sample(weights.begin(), weights.end());
    for (unsigned int ii = 0; ii < nsim_states; ii++) {
      alpha_pred.slice(ii) = alpha.slice(sample(engine)).cols(n - n_ahead, n - 1);
    }
  } else {
    alpha_pred = sim_smoother(nsim_states, distribution != 0).tube(0, n - n_ahead, m - 1,  n - 1);
  }
  
  unsigned int j = 0;
  
  if (n_burnin == 0){
    for (unsigned int ii = 0; ii < nsim_states; ii++) {
      for (unsigned int t = 0; t < n_ahead; t++) {
        pred_store(t, ii) = arma::as_scalar(Z.col(Ztv * (n - n_ahead + t)).t() * alpha_pred.slice(ii).col(t));
      }
    }
    if(xreg.n_cols > 0) {
      for (unsigned int ii = 0; ii < nsim_states; ii++) {
        pred_store.col(ii) +=  xbeta.subvec(n - n_ahead, n - 1);
      }
    }
    
    if (interval == 1) {
      switch(distribution) {
      case 1  :
        for (unsigned int ii = 0; ii < nsim_states; ii++) {
          pred_store.col(ii) = exp(pred_store.col(ii)) % phi.subvec(n - n_ahead, n - 1);
        }
        break;
      case 2  :
        for (unsigned int ii = 0; ii < nsim_states; ii++) {
          pred_store.col(ii) = exp(pred_store.col(ii)) / (1.0 + exp(pred_store.col(ii)));
        }
        break;
      case 3  :
        for (unsigned int ii = 0; ii < nsim_states; ii++) {
          pred_store.col(ii) = exp(pred_store.col(ii));
        }
        break;
      }
    } else {
      switch(distribution) {
      case 1  :
        for (unsigned int ii = 0; ii < nsim_states; ii++) {
          for (unsigned int t = 0; t < n_ahead; t++) {
            pred_store(t, ii) = R::rpois(exp(pred_store(t, ii)) * phi(n - n_ahead + t));
          }
        }
        break;
      case 2  :
        for (unsigned int ii = 0; ii < nsim_states; ii++) {
          for (unsigned int t = 0; t < n_ahead; t++) {
            pred_store(t, ii) = R::rbinom(phi(n - n_ahead + t), exp(pred_store(t, ii)) / (1.0 + exp(pred_store(t, ii))));
          }
        }
        break;
      case 3  :
        for (unsigned int ii = 0; ii < nsim_states; ii++) {
          for (unsigned int t = 0; t < n_ahead; t++) {
            pred_store(t, ii) = R::rnbinom(phi(n - n_ahead + t), exp(pred_store(t, ii)));
          }
        }
        break;
      }
    }
    j++;
  }
  
  double accept_prob = 0;
  double ll_prop = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    // sample from standard normal distribution
    //arma::vec u = rnorm(npar);
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;
    
    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute approximate log-likelihood with proposed theta
      signal = init_signal;
      ll_approx = approx(signal, max_iter, conv_tol);
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + q));
    } else accept_prob = 0;
    
    //accept
    if (inrange && (unif(engine) < accept_prob)) {
      if (nsim_states > 1) {
        arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
        arma::vec weights = exp(importance_weights(alpha) - scaling_factor(signal));
        double ll_w_prop = log(sum(weights) / nsim_states);
        double pp = std::min(1.0, exp(ll_w_prop - ll_w));
        //accept_prob *= pp;
        
        if (unif(engine) < pp) {
          ll = ll_prop;
          ll_w = ll_w_prop;
          theta = theta_prop;
          std::discrete_distribution<> sample(weights.begin(), weights.end());
          for (unsigned int ii = 0; ii < nsim_states; ii++) {
            alpha_pred.slice(ii) = alpha.slice(sample(engine)).cols(n - n_ahead, n - 1);
          }
        }
      } else {
        ll = ll_prop;
        theta = theta_prop;
        alpha_pred = sim_smoother(nsim_states, distribution != 0).tube(0, n - n_ahead, m - 1,  n - 1);
      }
    }
    if ((i >= n_burnin) && (i % n_thin == 0)) {
      update_model(theta);
      for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
        for (unsigned int t = 0; t < n_ahead; t++) {
          pred_store(t, ii) = arma::as_scalar(Z.col(Ztv * (n - n_ahead + t)).t() *
            alpha_pred.slice(ii - j * nsim_states).col(t));
        }
      }
      if(xreg.n_cols > 0) {
        for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
          pred_store.col(ii) += xbeta.subvec(n - n_ahead, n - 1);
        }
      }
      if (interval == 1) {
        switch(distribution) {
        case 1  :
          for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
            pred_store.col(ii) = exp(pred_store.col(ii)) % phi.subvec(n - n_ahead, n - 1);
          }
          break;
        case 2  :
          for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
            pred_store.col(ii) = exp(pred_store.col(ii)) / (1.0 + exp(pred_store.col(ii)));
          }
          break;
        case 3  :
          for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
            pred_store.col(ii) = exp(pred_store.col(ii));
          }
          break;
        }
      } else {
        switch(distribution) {
        case 1  :
          for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
            for (unsigned int t = 0; t < n_ahead; t++) {
              pred_store(t, ii) = R::rpois(exp(pred_store(t, ii)) * phi(n - n_ahead + t));
            }
          }
          break;
        case 2  :
          for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
            for (unsigned int t = 0; t < n_ahead; t++) {
              pred_store(t, ii) = R::rbinom(phi(n - n_ahead + t), exp(pred_store(t, ii)) / (1.0 + exp(pred_store(t, ii))));
            }
          }
          break;
        case 3  :
          for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
            for (unsigned int t = 0; t < n_ahead; t++) {
              pred_store(t, ii) = R::rnbinom(phi(n - n_ahead + t), exp(pred_store(t, ii)));
            }
          }
          break;
        }
      }
      j++;
    }
    
    adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    
  }
  
  return pred_store;
  
}
//compute log-weights
arma::vec ngssm::importance_weights(const arma::cube& alphasim) {
  
  arma::vec weights(alphasim.n_slices, arma::fill::zeros);
  
  switch(distribution) {
  case 0  :
    for (unsigned int i = 0; i < alphasim.n_slices; i++) {
      for (unsigned int t = 0; t < n; t++) {
        if (arma::is_finite(ng_y(t))) {
          double simsignal = alphasim(0, t, i);
          weights(i) += -0.5 * (simsignal +
            pow((ng_y(t) - xbeta(t)) / phi(t), 2) * exp(-simsignal)) +
            0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
        }
      }
    }
    break;
  case 1  :
    for (unsigned int i = 0; i < alphasim.n_slices; i++) {
      for (unsigned int t = 0; t < n; t++) {
        if (arma::is_finite(ng_y(t))) {
          double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
            alphasim.slice(i).col(t) + xbeta(t));
          weights(i) += ng_y(t) * simsignal  - phi(t) * exp(simsignal) +
            0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
        }
      }
    }
    break;
  case 2  :
    for (unsigned int i = 0; i < alphasim.n_slices; i++) {
      for (unsigned int t = 0; t < n; t++) {
        if (arma::is_finite(ng_y(t))) {
          double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
            alphasim.slice(i).col(t) + xbeta(t));
          weights(i) += ng_y(t) * simsignal - phi(t) * log(1 + exp(simsignal)) +
            0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
        }
      }
    }
    break;
  case 3  :
    for (unsigned int i = 0; i < alphasim.n_slices; i++) {
      for (unsigned int t = 0; t < n; t++) {
        if (arma::is_finite(ng_y(t))) {
          double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
            alphasim.slice(i).col(t) + xbeta(t));
          weights(i) += ng_y(t) * simsignal - (ng_y(t) + phi(t)) * log(phi(t) + exp(simsignal)) +
            0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
        }
      }
    }
    break;
  }
  
  return weights;
}

//compute log[p(y|alphahat)/g(y|alphahat)] without constants
double ngssm::scaling_factor(const arma::vec& signal) {
  
  double ll_approx_u = 0.0;
  switch(distribution) {
  case 0  :
    for (unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(ng_y(t))) {
        ll_approx_u += -0.5 * (signal(t) +
          pow((ng_y(t) - xbeta(t)) / phi(t), 2) * exp(-signal(t))) +
          0.5 * pow(y(t) - signal(t), 2) / HH(t);
      }
    }
    break;
  case 1  :
    for (unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(ng_y(t))) {
        ll_approx_u += ng_y(t) * (signal(t) + xbeta(t)) - phi(t) * exp(signal(t) + xbeta(t)) +
          0.5 * pow(y(t) - signal(t) - xbeta(t), 2) / HH(t);
      }
    }
    break;
  case 2  :
    for (unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(ng_y(t))) {
        ll_approx_u += ng_y(t) * (signal(t) + xbeta(t)) - phi(t) * log(1 + exp(signal(t) + xbeta(t))) +
          0.5 * std::pow(y(t) - signal(t) - xbeta(t), 2) / HH(t);
      }
    }
    break;
  case 3  :
    for (unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(ng_y(t))) {
        ll_approx_u += ng_y(t) * (signal(t) + xbeta(t)) -
          (ng_y(t) + phi(t)) * log(phi(t) + exp(signal(t) + xbeta(t))) +
          0.5 * std::pow(y(t) - signal(t) - xbeta(t), 2) / HH(t);
      }
    }
    break;
  }
  return ll_approx_u;
}


double ngssm::mcmc_approx(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
  const arma::vec init_signal, arma::mat& theta_store, arma::vec& ll_store,
  arma::mat& y_store, arma::mat& H_store, arma::vec& ll_approx_u_store,
  bool end_ram, bool adapt_approx) {
  
  unsigned int npar = theta_lwr.n_elem;
  
  unsigned int n_samples = ll_approx_u_store.n_elem;
  
  double acceptance_rate = 0.0;
  arma::vec theta = get_theta();
  arma::vec signal = init_signal;
  double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
  double ll = ll_approx + log_likelihood(distribution != 0);
  if (!std::isfinite(ll)) {
    Rcpp::stop("Non-finite log-likelihood from initial values. ");
  }
  unsigned int j = 0;
  
  double ll_approx_u = scaling_factor(signal);
  
  if (n_burnin == 0) {
    theta_store.col(0) = theta;
    ll_store(0) = ll;
    y_store.col(0) = y;
    H_store.col(0) = H;
    ll_approx_u_store(0) = ll_approx_u;
    acceptance_rate++;
    j++;
  }
  
  double accept_prob = 0;
  double ll_prop = 0;
  arma::vec y_tmp(n);
  arma::vec H_tmp(n);
  
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    
    y_tmp = y;
    H_tmp = H;
    // sample from standard normal distribution
    // arma::vec u = rnorm(npar);
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;
    
    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute approximate log-likelihood with proposed theta
      if (adapt_approx) {
        signal = init_signal;
        ll_approx = approx(signal, max_iter, conv_tol);
      }
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      if(!std::isfinite(ll_prop)) {
        accept_prob = 0;
      } else {
        double q = proposal(theta, theta_prop);
        accept_prob = std::min(1.0, exp(ll_prop - ll + q));
      }
      
    } else accept_prob = 0;
    
    if (inrange && (unif(engine) < accept_prob)) {
      if (i >= n_burnin) {
        acceptance_rate++;
      }
      ll = ll_prop;
      theta = theta_prop;
      if (adapt_approx) {
        ll_approx_u = scaling_factor(signal);
      }
    } else {
      y = y_tmp;
      H = H_tmp;
    }
    
    //store
    if ((i >= n_burnin) && (i % n_thin == 0) && j < n_samples) {
      ll_store(j) = ll;
      theta_store.col(j) = theta;
      y_store.col(j) = y;
      H_store.col(j) = H;
      ll_approx_u_store(j) = ll_approx_u;
      j++;
    }
    if (!end_ram || i < n_burnin) {
      adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    }
  }
  return acceptance_rate / (n_iter - n_burnin);
  
}


double ngssm::mcmc_approx2(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
  const arma::vec init_signal, arma::mat& theta_store, arma::vec& ll_store,
  arma::mat& y_store, arma::mat& H_store, arma::vec& ll_approx_u_store,
  arma::uvec& counts, bool end_ram, bool adapt_approx) {
  
  unsigned int npar = theta_lwr.n_elem;
  
  double acceptance_rate = 0.0;
  arma::vec theta = get_theta();
  arma::vec signal = init_signal;
  double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
  double ll = ll_approx + log_likelihood(distribution != 0);
  if (!std::isfinite(ll)) {
    Rcpp::stop("Non-finite log-likelihood from initial values. ");
  }
  
  double accept_prob = 0;
  double ll_prop = 0;
  
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for(unsigned int i = 0; i < n_burnin; i++) {
    
    // sample from standard normal distribution
    // arma::vec u = rnorm(npar);
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;
    
    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute approximate log-likelihood with proposed theta
      if (adapt_approx) {
        signal = init_signal;
        ll_approx = approx(signal, max_iter, conv_tol);
      }
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      if(!std::isfinite(ll_prop)) {
        accept_prob = 0;
      } else {
        double q = proposal(theta, theta_prop);
        accept_prob = std::min(1.0, exp(ll_prop - ll + q));
      }
    } else accept_prob = 0;
    
    
    if (inrange && (unif(engine) < accept_prob)) {
      ll = ll_prop;
      theta = theta_prop;
    }
    adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    
  }
  
  
  
  update_model(theta);
  theta_store.col(0) = theta;
  ll_store(0) = ll;
  if (adapt_approx) {
    // compute approximate log-likelihood with proposed theta
    signal = init_signal;
    ll_approx = approx(signal, max_iter, conv_tol);
  }
  y_store.col(0) = y;
  H_store.col(0) = H;
  double ll_approx_u = scaling_factor(signal);
  ll_approx_u_store(0) = ll_approx_u;
  counts(0) = 1;
  unsigned int n_unique = 0;
  arma::vec y_tmp(n);
  arma::vec H_tmp(n);
  
  for (unsigned int i = n_burnin  + 1; i < n_iter; i++) {
    y_tmp = y;
    H_tmp = H;
    // sample from standard normal distribution
    // arma::vec u = rnorm(npar);
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;
    
    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute approximate log-likelihood with proposed theta
      if (adapt_approx) {
        signal = init_signal;
        ll_approx = approx(signal, max_iter, conv_tol);
      }
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + q));
    } else accept_prob = 0;
    
    
    if (inrange && (unif(engine) < accept_prob)) {
      ll = ll_prop;
      theta = theta_prop;
      if (adapt_approx) {
        ll_approx_u = scaling_factor(signal);
      }
      n_unique++;
      acceptance_rate++;
      counts(n_unique) = 1;
      ll_store(n_unique) = ll;
      theta_store.col(n_unique) = theta;
      y_store.col(n_unique) = y;
      H_store.col(n_unique) = H;
      ll_approx_u_store(n_unique) = ll_approx_u;
      
    } else {
      y = y_tmp;
      H = H_tmp;
      counts(n_unique) = counts(n_unique) + 1;
    }
    
    if (!end_ram) {
      adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    }
    
  }
  theta_store.resize(npar, n_unique + 1);
  ll_store.resize(n_unique + 1);
  counts.resize(n_unique + 1);
  y_store.resize(n, n_unique + 1);
  H_store.resize(n, n_unique + 1);
  ll_approx_u_store.resize(n_unique + 1);
  
  return acceptance_rate / (n_iter - n_burnin);
  
}



List ngssm::mcmc_full(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat S,
  const arma::vec init_signal, bool end_ram, bool adapt_approx) {
  
  unsigned int npar = theta_lwr.n_elem;
  
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  // in order to save space,
  // we always store just one sample of alphas per iteration
  arma::cube alpha_store(m, n, n_samples);
  arma::vec ll_store(n_samples);
  double acceptance_rate = 0.0;
  
  arma::vec theta = get_theta();
  arma::vec signal = init_signal;
  double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
  double ll = ll_approx + log_likelihood(distribution != 0);
  if (!std::isfinite(ll)) {
    Rcpp::stop("Non-finite log-likelihood from initial values. ");
  }
  arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
  unsigned int ind = 0;
  unsigned int ind_prop = 0;
  double ll_approx_u = scaling_factor(signal);
  if (nsim_states > 1) {
    arma::vec weights = exp(importance_weights(alpha) - ll_approx_u);
    std::discrete_distribution<> sample(weights.begin(), weights.end());
    ind = sample(engine);
    ll += log(sum(weights) / nsim_states);
  }
  
  unsigned int j = 0;
  
  if (n_burnin == 0) {
    theta_store.col(0) = theta;
    ll_store(0) = ll;
    alpha_store.slice(0) = alpha.slice(ind);
    acceptance_rate++;
    j++;
  }
  
  double accept_prob = 0;
  double ll_prop = 0;
  arma::cube alpha_prop = alpha;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    // sample from standard normal distribution
    //arma::vec u = rnorm(npar);
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;
    
    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute approximate log-likelihood with proposed theta
      if (adapt_approx) {
        signal = init_signal;
        ll_approx = approx(signal, max_iter, conv_tol);
        ll_approx_u = scaling_factor(signal);
      }
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      
      // if nsim_states = 1, target hat_p(theta, alpha | y)
      if (std::isfinite(ll_prop) && nsim_states > 1) {
        alpha_prop = sim_smoother(nsim_states, distribution != 0);
        arma::vec weights = exp(importance_weights(alpha_prop) - ll_approx_u);
        ll_prop += log(sum(weights) / nsim_states);
        std::discrete_distribution<> sample(weights.begin(), weights.end());
        ind_prop = sample(engine);
      }
      if(!std::isfinite(ll_prop)) {
        accept_prob = 0;
      } else {
        double q = proposal(theta, theta_prop);
        accept_prob = std::min(1.0, exp(ll_prop - ll + q));
      }
    } else {
      accept_prob = 0.0;
    }
    if (inrange && unif(engine) < accept_prob) {
      if (i >= n_burnin) {
        acceptance_rate++;
      }
      ll = ll_prop;
      theta = theta_prop;
      if (nsim_states == 1){
        // this is just a sample from the approximating model
        alpha = sim_smoother(1, distribution != 0);
      } else {
        alpha = alpha_prop;
        ind = ind_prop;
      }
      
    }
    
    //store
    if ((i >= n_burnin) && (i % n_thin == 0) && j < n_samples) {
      ll_store(j) = ll;
      theta_store.col(j) = theta;
      alpha_store.slice(j) = alpha.slice(ind);
      j++;
    }
    
    if (!end_ram || i < n_burnin) {
      adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    }
    
  }
  
  arma::inplace_trans(theta_store);
  return List::create(Named("alpha") = alpha_store,
    Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate / (n_iter - n_burnin),
    Named("S") = S,  Named("logLik") = ll_store);
}

// delayed acceptance, always targets p(theta, alpha | y)
List ngssm::mcmc_da(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat S,
  const arma::vec init_signal, bool end_ram, bool adapt_approx) {
  
  unsigned int npar = theta_lwr.n_elem;
  
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  // in order to save space,
  // we always store just one sample of alphas per iteration
  arma::cube alpha_store(m, n, n_samples);
  arma::vec ll_store(n_samples);
  double acceptance_rate = 0.0;
  arma::vec theta = get_theta();
  arma::vec signal = init_signal;
  
  double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
  double ll = ll_approx + log_likelihood(distribution != 0);
  if (!std::isfinite(ll)) {
    Rcpp::stop("Non-finite log-likelihood from initial values. ");
  }
  double ll_approx_u = scaling_factor(signal);
  arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
  arma::vec weights = exp(importance_weights(alpha) - ll_approx_u);
  double ll_w = log(sum(weights) / nsim_states);
  
  std::discrete_distribution<> sample(weights.begin(), weights.end());
  
  unsigned int j = 0;
  unsigned int ind = sample(engine);
  
  if (n_burnin == 0) {
    theta_store.col(0) = theta;
    alpha_store.slice(0) = alpha.slice(ind);
    ll_store(0) = ll + ll_w;
    acceptance_rate++;
    j++;
  }
  
  double accept_prob = 0;
  double ll_prop = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    
    // sample from standard normal distribution
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // arma::vec u = normal(npar);
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;
    
    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute approximate log-likelihood with proposed theta
      if (adapt_approx) {
        signal = init_signal;
        ll_approx = approx(signal, max_iter, conv_tol);
        ll_approx_u = scaling_factor(signal);
      }
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      if(!std::isfinite(ll_prop)) {
        accept_prob = 0;
      } else {
        double q = proposal(theta, theta_prop);
        accept_prob = std::min(1.0, exp(ll_prop - ll + q));
      }
    } else accept_prob = 0;
    
    
    // initial acceptance based on hat_p(theta, alpha | y)
    if (inrange && (unif(engine) < accept_prob)) {
      // simulate states
      arma::cube alpha_prop = sim_smoother(nsim_states, distribution != 0);
      arma::vec weights = exp(importance_weights(alpha_prop) - ll_approx_u);
      double ll_w_prop = log(sum(weights) / nsim_states);
      // delayed acceptance ratio
      double pp = 0;
      if(std::isfinite(ll_w_prop)) {
        pp = std::min(1.0, exp(ll_w_prop - ll_w));
      }
      if (unif(engine) < pp) {
        if (i >= n_burnin) {
          acceptance_rate++;
        }
        ll = ll_prop;
        ll_w = ll_w_prop;
        theta = theta_prop;
        std::discrete_distribution<> sample(weights.begin(), weights.end());
        ind = sample(engine);
        alpha = alpha_prop;
      }
    }
    
    //store
    if ((i >= n_burnin) && (i % n_thin == 0) && j < n_samples) {
      ll_store(j) = ll + ll_w;
      theta_store.col(j) = theta;
      alpha_store.slice(j) = alpha.slice(ind);
      j++;
    }
    
    if (!end_ram || i < n_burnin) {
      adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    }
    
  }
  arma::inplace_trans(theta_store);
  return List::create(Named("alpha") = alpha_store,
    Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate / (n_iter - n_burnin),
    Named("S") = S,  Named("logLik") = ll_store);
  
}

List ngssm::mcmc_param(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat S,
  const arma::vec init_signal, bool end_ram, bool adapt_approx) {
  
  unsigned int npar = theta_lwr.n_elem;
  
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::vec ll_store(n_samples);
  double acceptance_rate = 0.0;
  
  arma::vec theta = get_theta();
  arma::vec signal = init_signal;
  double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
  double ll = ll_approx + log_likelihood(distribution != 0);
  if (!std::isfinite(ll)) {
    Rcpp::stop("Non-finite log-likelihood from initial values. ");
  }
  double ll_approx_u = scaling_factor(signal);
  
  if (nsim_states > 1) {
    arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
    arma::vec weights = exp(importance_weights(alpha) - ll_approx_u);
    ll += log(sum(weights) / nsim_states);
  }
  
  unsigned int j = 0;
  
  if (n_burnin == 0) {
    theta_store.col(0) = theta;
    ll_store(0) = ll;
    acceptance_rate++;
    j++;
  }
  
  double accept_prob = 0;
  double ll_prop = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    // sample from standard normal distribution
    //arma::vec u = rnorm(npar);
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;
    
    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute approximate log-likelihood with proposed theta
      if (adapt_approx) {
        signal = init_signal;
        ll_approx = approx(signal, max_iter, conv_tol);
        ll_approx_u = scaling_factor(signal);
      }
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      
      // if nsim_states = 1, target hat_p(theta, alpha | y)
      if (std::isfinite(ll_prop) && nsim_states > 1) {
        arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
        arma::vec weights = exp(importance_weights(alpha) - ll_approx_u);
        ll_prop += log(sum(weights) / nsim_states);
      }
      if(!std::isfinite(ll_prop)) {
        accept_prob = 0;
      } else {
        double q = proposal(theta, theta_prop);
        accept_prob = std::min(1.0, exp(ll_prop - ll + q));
      }
    } else {
      accept_prob = 0.0;
    }
    if (inrange && unif(engine) < accept_prob) {
      if (i >= n_burnin) {
        acceptance_rate++;
      }
      ll = ll_prop;
      theta = theta_prop;
      
    }
    
    //store
    if ((i >= n_burnin) && (i % n_thin == 0) && j < n_samples) {
      ll_store(j) = ll;
      theta_store.col(j) = theta;
      j++;
    }
    
    if (!end_ram || i < n_burnin) {
      adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    }
    
  }
  
  arma::inplace_trans(theta_store);
  return List::create(
    Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate / (n_iter - n_burnin),
    Named("S") = S,  Named("logLik") = ll_store);
}

// delayed acceptance, always targets p(theta, alpha | y)
List ngssm::mcmc_da_param(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat S,
  const arma::vec init_signal, bool end_ram, bool adapt_approx) {
  
  unsigned int npar = theta_lwr.n_elem;
  
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  
  arma::vec ll_store(n_samples);
  double acceptance_rate = 0.0;
  arma::vec theta = get_theta();
  arma::vec signal = init_signal;
  
  double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
  double ll = ll_approx + log_likelihood(distribution != 0);
  if (!std::isfinite(ll)) {
    Rcpp::stop("Non-finite log-likelihood from initial values. ");
  }
  
  double ll_approx_u = scaling_factor(signal);
  arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
  arma::vec weights = exp(importance_weights(alpha) - ll_approx_u);
  double ll_w = log(sum(weights) / nsim_states);
  
  
  unsigned int j = 0;
  
  if (n_burnin == 0) {
    theta_store.col(0) = theta;
    ll_store(0) = ll + ll_w;
    acceptance_rate++;
    j++;
  }
  
  double accept_prob = 0;
  double ll_prop = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    
    // sample from standard normal distribution
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // arma::vec u = normal(npar);
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;
    
    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute approximate log-likelihood with proposed theta
      if (adapt_approx) {
        signal = init_signal;
        ll_approx = approx(signal, max_iter, conv_tol);
        ll_approx_u = scaling_factor(signal);
      }
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      if(!std::isfinite(ll_prop)) {
        accept_prob = 0;
      } else {
        double q = proposal(theta, theta_prop);
        accept_prob = std::min(1.0, exp(ll_prop - ll + q));
      }
    } else accept_prob = 0;
    
    
    // initial acceptance based on hat_p(theta, alpha | y)
    if (inrange && (unif(engine) < accept_prob)) {
      // simulate states
      arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
      arma::vec weights = exp(importance_weights(alpha) - ll_approx_u);
      double ll_w_prop = log(sum(weights) / nsim_states);
      // delayed acceptance ratio
      double pp = 0;
      if(std::isfinite(ll_w_prop)) {
        pp = std::min(1.0, exp(ll_w_prop - ll_w));
      }
      if (unif(engine) < pp) {
        if (i >= n_burnin) {
          acceptance_rate++;
        }
        ll = ll_prop;
        ll_w = ll_w_prop;
        theta = theta_prop;
      }
    }
    
    //store
    if ((i >= n_burnin) && (i % n_thin == 0) && j < n_samples) {
      ll_store(j) = ll + ll_w;
      theta_store.col(j) = theta;
      j++;
    }
    
    if (!end_ram || i < n_burnin) {
      adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    }
    
  }
  arma::inplace_trans(theta_store);
  return List::create(
    Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate / (n_iter - n_burnin),
    Named("S") = S,  Named("logLik") = ll_store);
  
}


List ngssm::mcmc_summary(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat S,
  const arma::vec init_signal, bool end_ram, bool adapt_approx) {
  
  unsigned int npar = theta_lwr.n_elem;
  
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  
  arma::vec ll_store(n_samples);
  double acceptance_rate = 0.0;
  
  arma::vec theta = get_theta();
  arma::vec signal = init_signal;
  double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
  double ll = ll_approx + log_likelihood(distribution != 0);
  if (!std::isfinite(ll)) {
    Rcpp::stop("Non-finite log-likelihood from initial values. ");
  }
  double ll_approx_u = scaling_factor(signal);
  arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
  
  
  arma::mat alphahat(m, n, arma::fill::zeros);
  arma::cube Vt(m, m, n, arma::fill::zeros);
  arma::cube Valpha(m, m, n, arma::fill::zeros);
  arma::mat mu(1, n, arma::fill::zeros);
  arma::cube Vmu(1, 1, n, arma::fill::zeros);
  arma::cube Vmu2(1, 1, n, arma::fill::zeros);
  arma::vec weights(nsim_states, arma::fill::ones);
  
  if (nsim_states > 1) {
    weights = exp(importance_weights(alpha) - ll_approx_u);
    ll += log(sum(weights) / nsim_states);
  }
  
  unsigned int j = 0;
  
  if (n_burnin == 0) {
    arma::mat alphahat_i(m, n);
    arma::cube Vt_i(m, m, n);
    running_weighted_summary(alpha, alphahat_i, Vt_i, weights);
    Vt += (Vt_i - Vt) / (j + 1);
    running_summary(alphahat_i, alphahat, Valpha, j);
    
    arma::mat mu_i(1, n);
    arma::cube Vmu_i(1, 1, n);
    running_weighted_summary(invlink(alpha), mu_i, Vmu_i, weights);
    Vmu += (Vmu_i - Vmu) / (j + 1);
    running_summary(mu_i, mu, Vmu2, j);
    
    theta_store.col(0) = theta;
    ll_store(0) = ll;
    acceptance_rate++;
    j++;
  }
  
  double accept_prob = 0;
  double ll_prop = 0;
  arma::cube alpha_prop = alpha;
  arma::vec weights_prop = weights;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    // sample from standard normal distribution
    //arma::vec u = rnorm(npar);
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;
    
    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute approximate log-likelihood with proposed theta
      if (adapt_approx) {
        signal = init_signal;
        ll_approx = approx(signal, max_iter, conv_tol);
        ll_approx_u = scaling_factor(signal);
      }
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      
      // if nsim_states = 1, target hat_p(theta, alpha | y)
      if (std::isfinite(ll_prop) && nsim_states > 1) {
        alpha_prop = sim_smoother(nsim_states, distribution != 0);
        weights_prop = exp(importance_weights(alpha_prop) - ll_approx_u);
        ll_prop += log(sum(weights_prop) / nsim_states);
      }
      if(!std::isfinite(ll_prop)) {
        accept_prob = 0;
      } else {
        double q = proposal(theta, theta_prop);
        accept_prob = std::min(1.0, exp(ll_prop - ll + q));
      }
    } else {
      accept_prob = 0.0;
    }
    if (inrange && unif(engine) < accept_prob) {
      if (i >= n_burnin) {
        acceptance_rate++;
      }
      ll = ll_prop;
      theta = theta_prop;
      if (nsim_states == 1){
        // this is just a sample from the approximating model
        alpha = sim_smoother(1, distribution != 0);
      } else {
        alpha = alpha_prop;
        weights = weights_prop;
      }
      
    }
    
    if ((i >= n_burnin) && (i % n_thin == 0) && j < n_samples) {
      update_model(theta);
      arma::mat alphahat_i(m, n);
      arma::cube Vt_i(m, m, n);
      running_weighted_summary(alpha, alphahat_i, Vt_i, weights);
      Vt += (Vt_i - Vt) / (j + 1);
      running_summary(alphahat_i, alphahat, Valpha, j);
      
      arma::mat mu_i(1, n);
      arma::cube Vmu_i(1, 1, n);
      running_weighted_summary(invlink(alpha), mu_i, Vmu_i, weights);
      Vmu += (Vmu_i - Vmu) / (j + 1);
      running_summary(mu_i, mu, Vmu2, j);
      
      ll_store(j) = ll;
      theta_store.col(j) = theta;
      
      j++;
    }
    
    if (!end_ram || i < n_burnin) {
      adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    }
    
  }
  
  arma::inplace_trans(mu);
  arma::inplace_trans(alphahat);
  arma::inplace_trans(theta_store);
  return List::create(Named("alphahat") = alphahat, Named("Vt") =  Vt + Valpha,
    Named("muhat") = mu, Named("Vmu") =  Vmu + Vmu2,
    Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate / (n_iter - n_burnin),
    Named("S") = S,  Named("logLik") = ll_store);
}

// delayed acceptance, always targets p(theta, alpha | y)
List ngssm::mcmc_da_summary(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat S,
  const arma::vec init_signal, bool end_ram, bool adapt_approx) {
  
  unsigned int npar = theta_lwr.n_elem;
  
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  
  arma::vec ll_store(n_samples);
  double acceptance_rate = 0.0;
  arma::vec theta = get_theta();
  arma::vec signal = init_signal;
  
  double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
  double ll = ll_approx + log_likelihood(distribution != 0);
  if (!std::isfinite(ll)) {
    Rcpp::stop("Non-finite log-likelihood from initial values. ");
  }
  double ll_approx_u = scaling_factor(signal);
  
  arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
  arma::vec weights = exp(importance_weights(alpha) - ll_approx_u);
  double ll_w = log(sum(weights) / nsim_states);
  
  arma::mat alphahat(m, n, arma::fill::zeros);
  arma::cube Vt(m, m, n, arma::fill::zeros);
  arma::cube Valpha(m, m, n, arma::fill::zeros);
  arma::mat mu(1, n, arma::fill::zeros);
  arma::cube Vmu(1, 1, n, arma::fill::zeros);
  arma::cube Vmu2(1, 1, n, arma::fill::zeros);
  
  unsigned int j = 0;
  
  if (n_burnin == 0) {
    arma::mat alphahat_i(m, n);
    arma::cube Vt_i(m, m, n);
    running_weighted_summary(alpha, alphahat_i, Vt_i, weights);
    Vt += (Vt_i - Vt) / (j + 1);
    running_summary(alphahat_i, alphahat, Valpha, j);
    
    arma::mat mu_i(1, n);
    arma::cube Vmu_i(1, 1, n);
    running_weighted_summary(invlink(alpha), mu_i, Vmu_i, weights);
    Vmu += (Vmu_i - Vmu) / (j + 1);
    running_summary(mu_i, mu, Vmu2, j);
    
    theta_store.col(0) = theta;
    ll_store(0) = ll + ll_w;
    acceptance_rate++;
    j++;
  }
  
  double accept_prob = 0;
  double ll_prop = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    
    // sample from standard normal distribution
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // arma::vec u = normal(npar);
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;
    
    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute approximate log-likelihood with proposed theta
      if (adapt_approx) {
        signal = init_signal;
        ll_approx = approx(signal, max_iter, conv_tol);
        ll_approx_u = scaling_factor(signal);
      }
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      if(!std::isfinite(ll_prop)) {
        accept_prob = 0;
      } else {
        double q = proposal(theta, theta_prop);
        accept_prob = std::min(1.0, exp(ll_prop - ll + q));
      }
    } else accept_prob = 0;
    
    
    // initial acceptance based on hat_p(theta, alpha | y)
    if (inrange && (unif(engine) < accept_prob)) {
      // simulate states
      arma::cube alpha_prop = sim_smoother(nsim_states, distribution != 0);
      arma::vec weights_prop = exp(importance_weights(alpha_prop) - ll_approx_u);
      double ll_w_prop = log(sum(weights_prop) / nsim_states);
      // delayed acceptance ratio
      double pp = 0;
      if(std::isfinite(ll_w_prop)) {
        pp = std::min(1.0, exp(ll_w_prop - ll_w));
      }
      // accept_prob *= pp; // only count the initial acceptance
      // Rcout<<"ll_approx_u "<<ll_approx_u<<std::endl;
      // Rcout<<"ll_w_prop - ll_w "<<ll_w_prop - ll_w<<std::endl;
      if (unif(engine) < pp) {
        if (i >= n_burnin) {
          acceptance_rate++;
        }
        ll = ll_prop;
        ll_w = ll_w_prop;
        theta = theta_prop;
        alpha = alpha_prop;
        weights = weights_prop;
      }
    }
    
    //store
    if ((i >= n_burnin) && (i % n_thin == 0) && j < n_samples) {
      update_model(theta);
      arma::mat alphahat_i(m, n);
      arma::cube Vt_i(m, m, n);
      running_weighted_summary(alpha, alphahat_i, Vt_i, weights);
      Vt += (Vt_i - Vt) / (j + 1);
      running_summary(alphahat_i, alphahat, Valpha, j);
      
      arma::mat mu_i(1, n);
      arma::cube Vmu_i(1, 1, n);
      running_weighted_summary(invlink(alpha), mu_i, Vmu_i, weights);
      Vmu += (Vmu_i - Vmu) / (j + 1);
      running_summary(mu_i, mu, Vmu2, j);
      
      ll_store(j) = ll + ll_w;
      theta_store.col(j) = theta;
      j++;
    }
    
    if (!end_ram || i < n_burnin) {
      adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    }
    
  }
  arma::inplace_trans(mu);
  arma::inplace_trans(alphahat);
  arma::inplace_trans(theta_store);
  return List::create(Named("alphahat") = alphahat,
    Named("Vt") =  Vt + Valpha,
    Named("muhat") = mu, Named("Vmu") =  Vmu + Vmu2,
    Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate / (n_iter - n_burnin),
    Named("S") = S,  Named("logLik") = ll_store);
  
}

arma::cube ngssm::invlink(const arma::cube& alpha) {
  
  unsigned int nsim = alpha.n_slices;
  arma::cube y_mean(1, n, nsim);
  switch(distribution) {
  case 0  :
    if(xreg.n_cols > 0) {
      for (unsigned int i = 0; i < nsim; i++) {
        for (unsigned int t = 0; t < n; t++) {
          y_mean(0, t, i) = xbeta(t);
        }
      }
    } else {
      y_mean.zeros();
    }
    break;
  case 1  :
    if(xreg.n_cols > 0) {
      for (unsigned int i = 0; i < nsim; i++) {
        for (unsigned int t = 0; t < n; t++) {
          y_mean(0, t, i) = arma::as_scalar(
            exp(xbeta(t) + Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
        }
      }
    } else {
      for (unsigned int i = 0; i < nsim; i++) {
        for (unsigned int t = 0; t < n; t++) {
          y_mean(0, t, i) = arma::as_scalar(
            exp(Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
        }
      }
    }
    break;
  case 2  :
    if(xreg.n_cols > 0) {
      for (unsigned int i = 0; i < nsim; i++) {
        for (unsigned int t = 0; t < n; t++) {
          double tmp = arma::as_scalar(exp(xbeta(t) + Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
          y_mean(0, t, i) = tmp / (1.0 + tmp);
        }
      }
    } else {
      for (unsigned int i = 0; i < nsim; i++) {
        for (unsigned int t = 0; t < n; t++) {
          double tmp = arma::as_scalar(exp(Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
          y_mean(0, t, i) = tmp / (1.0 + tmp);
        }
      }
    }
    break;
  case 3  :
    if(xreg.n_cols > 0) {
      for (unsigned int i = 0; i < nsim; i++) {
        for (unsigned int t = 0; t < n; t++) {
          y_mean(0, t, i) = arma::as_scalar(exp(xbeta(t) + Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
        }
      }
    } else {
      for (unsigned int i = 0; i < nsim; i++) {
        for (unsigned int t = 0; t < n; t++) {
          y_mean(0, t, i) = arma::as_scalar(exp(Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
        }
      }
    }
    break;
  }
  return y_mean;
}



//compute log-weights
arma::vec ngssm::pyt(const unsigned int t, const arma::cube& alphasim) {
  
  arma::vec V(alphasim.n_slices);
  
  if (arma::is_finite(ng_y(t))) {
    switch(distribution) {
    case 0  :
      for (unsigned int i = 0; i < alphasim.n_slices; i++) {
        V(i) = -0.5 * (LOG2PI + 2.0 * log(phi(0)) + alphasim(0, t, i) + 
          pow((ng_y(t) - xbeta(t))/phi(0), 2) * exp(-alphasim(0, t, i)));
        
      }
      break;
    case 1  :
      for (unsigned int i = 0; i < alphasim.n_slices; i++) {
        double exptmp = exp(arma::as_scalar(Z.col(t * Ztv).t() *
          alphasim.slice(i).col(t) + xbeta(t)));
        V(i) = R::dpois(ng_y(t), phi(t) * exptmp, 1);
      }
      break;
    case 2  :
      for (unsigned int i = 0; i < alphasim.n_slices; i++) {
        double exptmp = exp(arma::as_scalar(Z.col(t * Ztv).t() *
          alphasim.slice(i).col(t) + xbeta(t)));
        V(i) = R::dbinom(ng_y(t), phi(t),  exptmp / (1.0 + exptmp), 1);
      }
      break;
    case 3  :
      for (unsigned int i = 0; i < alphasim.n_slices; i++) {
        double exptmp = exp(arma::as_scalar(Z.col(t * Ztv).t() *
          alphasim.slice(i).col(t) + xbeta(t)));
        V(i) = R::dnbinom_mu(ng_y(t), phi(t), exptmp, 1);
      }
      break;
    }
  } else {
    V.fill(-arma::datum::inf);
  }
  
  return V;
}
//bootstrap filter
double ngssm::bootstrap_filter(unsigned int nsim, arma::cube& alphasim, arma::vec& V) {
  
  arma::uvec nonzero = arma::find(P1.diag() > 0);
  arma::mat L_P1(m, m, arma::fill::zeros);
  if (nonzero.n_elem > 0) {
    L_P1.submat(nonzero, nonzero) =
      arma::chol(P1.submat(nonzero, nonzero), "lower");
  }
  
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alphasim.slice(i).col(0) = a1 + L_P1 * um;
  }
  double w = 0;
  if (arma::is_finite(ng_y(0))) {
    V = pyt(0, alphasim);
    double maxV = V.max();
    V = exp(V - maxV);
    w = maxV + log(sum(V)) - log(nsim);
    V /= sum(V);
  } else {
    V.ones();
  }
  
  arma::mat ind(n, nsim);
  for (unsigned int t = 0; t < (n - 1); t++) {
    std::discrete_distribution<> sample(V.begin(), V.end());
    
    arma::mat alphatmp(m, nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      ind(t, i) = sample(engine);
      alphatmp.col(i) = alphasim.slice(ind(t, i)).col(t);
    }
    
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec uk(k);
      for(unsigned int j = 0; j < k; j++) {
        uk(j) = normal(engine);
      }
      alphasim.slice(i).col(t + 1) = T.slice(t * Ttv) * alphatmp.col(i) + 
        R.slice(t * Rtv) * uk;
    }
    if (arma::is_finite(ng_y(t + 1))) {
      V = pyt(t + 1, alphasim);
      double maxV = V.max();
      V = exp(V - maxV);
      w += maxV + log(sum(V)) - log(nsim);
      V /= sum(V);
    } else {
      V.ones();
    }
  }
  
  for (int t = n - 2; t >= 0; t--) {
    arma::mat alphatmp = alphasim.tube(arma::span::all, arma::span(t));
    for (unsigned int i = 0; i < nsim; i++) {
      alphasim.slice(i).col(t) = alphatmp.col(ind(t,i));
    }
  }
  return w;
}


//bootstrap filter with initial value simulation from approximating model
double ngssm::gap_filter0(unsigned int nsim, arma::cube& alphasim, arma::vec& V, arma::vec& init_signal) {
  
  arma::mat alphahat(m, n);
  arma::cube Vt(m, m, n);
  
  double ll_approx = approx(init_signal, 100, 1e-8);
  smoother(alphahat, Vt, distribution != 0);
  
  arma::uvec nonzero = arma::find(Vt.slice(0).diag() > 0);
  arma::mat L(m, m, arma::fill::zeros);
  if (nonzero.n_elem > 0) {
    L.submat(nonzero, nonzero) =
      arma::chol(Vt.slice(0).submat(nonzero, nonzero), "lower");
  }
  
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alphasim.slice(i).col(0) = alphahat.col(0) + L * um;
  }
  nonzero = arma::find(P1.diag() > 0);
  arma::mat L_P1(m, m, arma::fill::zeros);
  if (nonzero.n_elem > 0) {
    L_P1.submat(nonzero, nonzero) =
      arma::chol(P1.submat(nonzero, nonzero), "lower");
  }
  
  double w = 0;
  if (arma::is_finite(ng_y(0))) {
    V = pyt(0, alphasim) + dmvnorm1(alphasim.tube(arma::span::all, arma::span(0)),
      a1, L_P1, true, true)
    - dmvnorm1(alphasim.tube(arma::span::all, arma::span(0)),
      alphahat.col(0), L, true, true);
    double maxV = V.max();
    V = exp(V - maxV);
    w = maxV + log(sum(V)) - log(nsim);
    V /= sum(V);
  } else {
    V.ones();
  }
  arma::mat ind(n, nsim);
  for (unsigned int t = 0; t < (n - 1); t++) {
    std::discrete_distribution<> sample(V.begin(), V.end());
    
    arma::mat alphatmp(m, nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      ind(t, i) = sample(engine);
      alphatmp.col(i) = alphasim.slice(ind(t, i)).col(t);
    }
    
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec uk(k);
      for(unsigned int j = 0; j < k; j++) {
        uk(j) = normal(engine);
      }
      alphasim.slice(i).col(t + 1) = T.slice(t * Ttv) * alphatmp.col(i) + 
        R.slice(t * Rtv) * uk;
    }
    if (arma::is_finite(ng_y(t + 1))) {
      V = pyt(t + 1, alphasim);
      double maxV = V.max();
      V = exp(V - maxV);
      w += maxV + log(sum(V)) - log(nsim);
      V /= sum(V);
    } else {
      V.ones();
    }
  }
  
  for (int t = n - 2; t >= 0; t--) {
    arma::mat alphatmp = alphasim.tube(arma::span::all, arma::span(t));
    for (unsigned int i = 0; i < nsim; i++) {
      alphasim.slice(i).col(t) = alphatmp.col(ind(t,i));
    }
  }
  return w;
}

// 
//bootstrap filter with initial value simulation from approximating model
// double ngssm::gap_filter(unsigned int nsim, arma::cube& alphasim, arma::vec& V, arma::vec& init_signal) {
// 
//   arma::mat alphahat(m, n);
//   arma::cube Vt(m, m, n);
// 
//   double ll_approx = approx(init_signal, 100, 1e-8);
//   smoother(alphahat, Vt, distribution != 0);
// 
//   arma::mat L(m, m);
//   L = arma::chol(Vt.slice(0), "lower");
// 
//   std::normal_distribution<> normal(0.0, 1.0);
// 
//   for (unsigned int i = 0; i < nsim; i++) {
//     arma::vec um(m);
//     for(unsigned int j = 0; j < m; j++) {
//       um(j) = normal(engine);
//     }
//     alphasim.slice(i).col(0) = alphahat.col(0) + L * um;
//   }
//   arma::uvec nonzero = arma::find(P1.diag() > 0);
//   arma::mat L_P1(m, m, arma::fill::zeros);
//   if (nonzero.n_elem > 0) {
//     L_P1.submat(nonzero, nonzero) =
//       arma::chol(P1.submat(nonzero, nonzero), "lower");
//   }
//   double w = 0;
//   if (arma::is_finite(ng_y(0))) {
//     V = pyt(0, alphasim) + dmvnorm1(alphasim.tube(arma::span::all, arma::span(0)),
//       a1, L_P1, true, true)
//     - dmvnorm1(alphasim.tube(arma::span::all, arma::span(0)),
//       alphahat.col(0), L, true, true);
//     double maxV = V.max();
//     V = exp(V - maxV);
//     w = maxV + log(sum(V)) - log(nsim);
//     V /= sum(V);
//   } else {
//     V.ones();
//   }
//   for (unsigned int t = 0; t < (n - 1); t++) {
//     std::discrete_distribution<> sample(V.begin(), V.end());
//     arma::cube alphatmp(m, n, nsim);
//     for (unsigned int i = 0; i < nsim; i++) {
//       alphatmp(arma::span::all, arma::span(0, t), arma::span(i)) =
//         alphasim(arma::span::all, arma::span(0, t), arma::span(sample(engine)));
//     }
//     alphasim(arma::span::all, arma::span(0, t), arma::span::all) =
//       alphatmp(arma::span::all, arma::span(0, t), arma::span::all);
// 
//     L = arma::chol(Vt.slice(t + 1), "lower");
// 
//     for (unsigned int i = 0; i < nsim; i++) {
//       arma::vec um(m);
//       for(unsigned int j = 0; j < m; j++) {
//         um(j) = normal(engine);
//       }
//       alphasim.slice(i).col(t + 1) = alphahat.col(t + 1) +
//         L * um;
//     }
//     if (arma::is_finite(ng_y(t + 1))) {
//       V = pyt(t + 1, alphasim) + dmvnorm2(alphasim.tube(arma::span::all, arma::span(t + 1)),
//         alphasim.tube(arma::span::all, arma::span(t)),
//         R.slice(Rtv * t), true, true, T.slice(Ttv * t))
//       - dmvnorm1(alphasim.tube(arma::span::all, arma::span(t + 1)),
//         alphahat.col(t + 1), L, true, true);
//       double maxV = V.max();
//       V = exp(V - maxV);
//       w += maxV + log(sum(V)) - log(nsim);
//       V /= sum(V);
//     } else {
//       V.ones();
//     }
//   }
// 
//   return w;
// }
// 

double ngssm::gap_filter(unsigned int nsim, arma::cube& alphasim, arma::vec& V, arma::vec& init_signal) {
  
  arma::mat alphahat(m, n);
  arma::cube Vt(m, m, n);
  
  double ll_approx = approx(init_signal, 100, 1e-8);
  smoother(alphahat, Vt, distribution != 0);
  
  arma::mat L(m, m);
  L = arma::chol(Vt.slice(0), "lower");
  
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alphasim.slice(i).col(0) = alphahat.col(0) + L * um;
  }
  arma::uvec nonzero = arma::find(P1.diag() > 0);
  arma::mat L_P1(m, m, arma::fill::zeros);
  if (nonzero.n_elem > 0) {
    L_P1.submat(nonzero, nonzero) =
      arma::chol(P1.submat(nonzero, nonzero), "lower");
  }
  double w = 0;
  if (arma::is_finite(ng_y(0))) {
    V = pyt(0, alphasim) + dmvnorm1(alphasim.tube(arma::span::all, arma::span(0)),
      a1, L_P1, true, true)
    - dmvnorm1(alphasim.tube(arma::span::all, arma::span(0)),
      alphahat.col(0), L, true, true);
    double maxV = V.max();
    V = exp(V - maxV);
    w = maxV + log(sum(V)) - log(nsim);
    V /= sum(V);
  } else {
    V.ones();
  }
  for (unsigned int t = 0; t < (n - 1); t++) {
    std::discrete_distribution<> sample(V.begin(), V.end());
    arma::cube alphatmp(m, n, nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp(arma::span::all, arma::span(0, t), arma::span(i)) =
        alphasim(arma::span::all, arma::span(0, t), arma::span(sample(engine)));
    }
    alphasim(arma::span::all, arma::span(0, t), arma::span::all) =
      alphatmp(arma::span::all, arma::span(0, t), arma::span::all);
    
    L = arma::chol(Vt.slice(t + 1), "lower");
    
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      alphasim.slice(i).col(t + 1) = alphahat.col(t + 1) +
        L * um;
    }
    if (arma::is_finite(ng_y(t + 1))) {
      V = pyt(t + 1, alphasim) + dmvnorm2(alphasim.tube(arma::span::all, arma::span(t + 1)),
        alphasim.tube(arma::span::all, arma::span(t)),
        R.slice(Rtv * t), true, true, T.slice(Ttv * t))
      - dmvnorm1(alphasim.tube(arma::span::all, arma::span(t + 1)),
        alphahat.col(t + 1), L, true, true);
      double maxV = V.max();
      V = exp(V - maxV);
      w += maxV + log(sum(V)) - log(nsim);
      V /= sum(V);
    } else {
      V.ones();
    }
  }
  
  return w;
}



// delayed acceptance, always targets p(theta, alpha | y)
List ngssm::mcmc_da_bsf(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat S,
  const arma::vec init_signal, bool end_ram, bool adapt_approx) {
  
  unsigned int npar = theta_lwr.n_elem;
  
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  // in order to save space,
  // we always store just one sample of alphas per iteration
  arma::cube alpha_store(m, n, n_samples);
  arma::vec ll_store(n_samples);
  double acceptance_rate = 0.0;
  arma::vec theta = get_theta();
  arma::vec signal = init_signal;
  
  double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
  double ll = ll_approx + log_likelihood(distribution != 0);
  if (!std::isfinite(ll)) {
    Rcpp::stop("Non-finite log-likelihood from initial values. ");
  }
  arma::cube alpha(m, n, nsim_states);
  arma::vec weights(nsim_states);
  double ll_w = bootstrap_filter(nsim_states, alpha, weights);
  
  std::discrete_distribution<> sample(weights.begin(), weights.end());
  
  unsigned int j = 0;
  unsigned int ind = sample(engine);
  
  if (n_burnin == 0) {
    theta_store.col(0) = theta;
    alpha_store.slice(0) = alpha.slice(ind);
    ll_store(0) = ll_w;
    acceptance_rate++;
    j++;
  }
  
  double accept_prob = 0;
  double ll_prop = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    
    // sample from standard normal distribution
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // arma::vec u = normal(npar);
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;
    
    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute approximate log-likelihood with proposed theta
      if (adapt_approx) {
        signal = init_signal;
        ll_approx = approx(signal, max_iter, conv_tol);
      }
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      if(!std::isfinite(ll_prop)) {
        accept_prob = 0;
      } else {
        double q = proposal(theta, theta_prop);
        accept_prob = std::min(1.0, exp(ll_prop - ll + q));
      }
    } else accept_prob = 0;
    
    
    // initial acceptance based on hat_p(theta, alpha | y)
    if (inrange && (unif(engine) < accept_prob)) {
      // simulate states
      arma::cube alpha_prop(m, n, nsim_states);
      double ll_w_prop = bootstrap_filter(nsim_states, alpha_prop, weights);
    Rcout<<ll_w_prop<<std::endl;
      // delayed acceptance ratio
      double pp = 0;
      if(std::isfinite(ll_w_prop)) {
        pp = std::min(1.0, exp(ll_w_prop - ll_w + ll - ll_prop));
      }
      if (unif(engine) < pp) {
        if (i >= n_burnin) {
          acceptance_rate++;
        }
        ll = ll_prop;
        ll_w = ll_w_prop;
        theta = theta_prop;
        std::discrete_distribution<> sample(weights.begin(), weights.end());
        ind = sample(engine);
        alpha = alpha_prop;
      }
    }
    
    //store
    if ((i >= n_burnin) && (i % n_thin == 0) && j < n_samples) {
      ll_store(j) = ll_w;
      theta_store.col(j) = theta;
      alpha_store.slice(j) = alpha.slice(ind);
      j++;
    }
    
    if (!end_ram || i < n_burnin) {
      adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    }
    
  }
  arma::inplace_trans(theta_store);
  return List::create(Named("alpha") = alpha_store,
    Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate / (n_iter - n_burnin),
    Named("S") = S,  Named("logLik") = ll_store);
  
}

