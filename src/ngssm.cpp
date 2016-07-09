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
  // approximation does not need to be accurate
  // as we are correcting the approximation with importance sampling
  //unsigned int max_iter = 1000;
  //double conv_tol = 1e-8;
  //
  unsigned int i = 0;
  while(i < max_iter) {
    // compute new guess of signal
    arma::vec signal_new = approx_iter(signal);
    //log[p(signal)] + log[p(y | signal)]
    double ll_new = precomp_logp_signal(signal_new, Kt, Ft) +
      logp_y(signal_new);
    double diff = std::abs(ll_new - ll)/(0.1 + std::abs(ll_new));
    signal = signal_new;

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
  arma::vec signal_tmp = signal;

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



List ngssm::mcmc_full(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat S,
  const arma::vec init_signal) {

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
  arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
  unsigned int ind = 0;
  unsigned int ind_prop = 0;



  if (nsim_states > 1) {
    arma::vec weights = exp(importance_weights(alpha, signal));
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
      signal = init_signal;
      ll_approx = approx(signal, max_iter, conv_tol);
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later

      // if nsim_states = 1, target hat_p(theta, alpha | y)
      if (nsim_states > 1) {
        alpha_prop = sim_smoother(nsim_states, distribution != 0);
        arma::vec weights = exp(importance_weights(alpha_prop, signal));
        ll_prop += log(sum(weights) / nsim_states);
        std::discrete_distribution<> sample(weights.begin(), weights.end());
        ind_prop = sample(engine);
      }
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + q));
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

    double change = accept_prob - target_acceptance;
    u = S * u / arma::norm(u) * sqrt(std::min(1.0, npar * pow(i, -gamma)) *
      std::abs(change));

    if(change > 0) {
      S = cholupdate(S, u);

    } else {
      if(change < 0){
        //update S unless numerical problems occur
        arma::mat Stmp = choldowndate(S, u);
        arma::uvec cond = arma::find(arma::diagvec(Stmp) < 0);
        if (cond.n_elem == 0) {
          S = Stmp;
        }
      }
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
  const arma::vec init_signal) {

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
  arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
  arma::vec weights = exp(importance_weights(alpha, signal));
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
      signal = init_signal;
      ll_approx = approx(signal, max_iter, conv_tol);
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + q));
    } else accept_prob = 0;


    // initial acceptance based on hat_p(theta, alpha | y)
    if (inrange && (unif(engine) < accept_prob)) {
      // simulate states
      arma::cube alpha_prop = sim_smoother(nsim_states, distribution != 0);
      arma::vec weights = exp(importance_weights(alpha_prop, signal));
      double ll_w_prop = log(sum(weights) / nsim_states);
      // delayed acceptance ratio
      double pp = std::min(1.0, exp(ll_w_prop - ll_w));
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

    double change = accept_prob - target_acceptance;
    u = S * u / arma::norm(u) * sqrt(std::min(1.0, npar * pow(i, -gamma)) *
      std::abs(change));


    if(change > 0) {
      S = cholupdate(S, u);

    } else {
      if(change < 0){
        //update S unless numerical problems occur
        arma::mat Stmp = choldowndate(S, u);
        arma::uvec cond = arma::find(arma::diagvec(Stmp) < 0);
        if (cond.n_elem == 0) {
          S = Stmp;
        }
      }
    }

  }
  arma::inplace_trans(theta_store);
  return List::create(Named("alpha") = alpha_store,
    Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate / (n_iter - n_burnin),
    Named("S") = S,  Named("logLik") = ll_store);

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
    arma::vec weights = exp(importance_weights(alpha, signal));
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
        arma::vec weights = exp(importance_weights(alpha, signal));
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

    double change = accept_prob - target_acceptance;
    u = S * u * sqrt(std::min(1.0, npar * pow(i, -gamma)) * std::abs(change)) /
      arma::norm(u);

    if(change > 0) {
      S = cholupdate(S, u);

    } else {
      if(change < 0){
        //update S unless numerical problems occur
        arma::mat Stmp = choldowndate(S, u);
        arma::uvec cond = arma::find(arma::diagvec(Stmp) < 0);
        if (cond.n_elem == 0) {
          S = Stmp;
        }
      }
    }

  }

  return pred_store;

}
//compute log-weights normalized with log[p(y|alphahat)/g(y|alphahat)]
arma::vec ngssm::importance_weights(const arma::cube& alphasim, const arma::vec& signal) {

  arma::vec weights(alphasim.n_slices, arma::fill::zeros);

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
  switch(distribution) {
  case 0  :
    for (unsigned int i = 0; i < alphasim.n_slices; i++) {
      for (unsigned int t = 0; t < n; t++) {
        if (arma::is_finite(ng_y(t))) {
          double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
            alphasim.slice(i).col(t));
          weights(i) += -0.5 * (simsignal +
            pow((ng_y(t) - xbeta(t)) / phi(t), 2) * exp(-simsignal)) +
            0.5 * pow(y(t) - simsignal, 2) / HH(t);
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

  return weights - ll_approx_u;
}

//compute log-weights without normalization
arma::vec ngssm::importance_weights2(const arma::cube& alphasim) {

  arma::vec weights(alphasim.n_slices, arma::fill::zeros);

  switch(distribution) {
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

double ngssm::mcmc_approx(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
  const arma::vec init_signal, arma::mat& theta_store, arma::vec& ll_store,
  arma::mat& y_store, arma::mat& H_store, arma::vec& ll_approx_u_store) {

  unsigned int npar = theta_lwr.n_elem;

  unsigned int n_samples = ll_approx_u_store.n_elem;

  double acceptance_rate = 0.0;
  arma::vec theta = get_theta();
  arma::vec signal = init_signal;
  double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
  double ll = ll_approx + log_likelihood(distribution != 0);

  unsigned int j = 0;

  double ll_approx_u = 0.0;
  for (unsigned int t = 0; t < n; t++) {
    if (arma::is_finite(ng_y(t))) {
      ll_approx_u += ng_y(t) * (signal(t) + xbeta(t)) -
        phi(t) * exp(signal(t) + xbeta(t)) + 0.5 * pow(y(t) - signal(t) - xbeta(t), 2) / HH(t);
    }
  }

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
      signal = init_signal;
      ll_approx = approx(signal, max_iter, conv_tol);
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + q));
    } else accept_prob = 0;


    if (inrange && (unif(engine) < accept_prob)) {
      if (i >= n_burnin) {
        acceptance_rate++;
      }
      ll = ll_prop;
      theta = theta_prop;
      ll_approx_u = 0.0;
      for (unsigned int t = 0; t < n; t++) {
        if (arma::is_finite(ng_y(t))) {
          ll_approx_u += ng_y(t) * (signal(t) + xbeta(t))-
            phi(t) * exp(signal(t) + xbeta(t)) + 0.5 * pow(y(t) - signal(t) - xbeta(t), 2) / HH(t);
        }
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

    double change = accept_prob - target_acceptance;
    u = S * u / arma::norm(u) * sqrt(std::min(1.0, npar * pow(i, -gamma)) *
      std::abs(change));


    if(change > 0) {
      S = cholupdate(S, u);

    } else {
      if(change < 0){
        //update S unless numerical problems occur
        arma::mat Stmp = choldowndate(S, u);
        arma::uvec cond = arma::find(arma::diagvec(Stmp) < 0);
        if (cond.n_elem == 0) {
          S = Stmp;
        }
      }
    }

  }
  return acceptance_rate / (n_iter - n_burnin);

}


double ngssm::mcmc_approx2(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
  const arma::vec init_signal, arma::mat& theta_store, arma::vec& ll_store,
  arma::mat& y_store, arma::mat& H_store, arma::vec& ll_approx_u_store, arma::uvec& counts) {

  unsigned int npar = theta_lwr.n_elem;

  unsigned int n_samples = ll_approx_u_store.n_elem;

  double acceptance_rate = 0.0;
  arma::vec theta = get_theta();
  arma::vec signal = init_signal;
  double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
  double ll = ll_approx + log_likelihood(distribution != 0);


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
      signal = init_signal;
      ll_approx = approx(signal, max_iter, conv_tol);
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + q));
    } else accept_prob = 0;


    if (inrange && (unif(engine) < accept_prob)) {
      ll = ll_prop;
      theta = theta_prop;
    }

    double change = accept_prob - target_acceptance;
    u = S * u / arma::norm(u) * sqrt(std::min(1.0, npar * pow(i, -gamma)) *
      std::abs(change));

    if(change > 0) {
      S = cholupdate(S, u);

    } else {
      if(change < 0){
        //update S unless numerical problems occur
        arma::mat Stmp = choldowndate(S, u);
        arma::uvec cond = arma::find(arma::diagvec(Stmp) < 0);
        if (cond.n_elem == 0) {
          S = Stmp;
        }
      }
    }
  }



  update_model(theta);
  theta_store.col(0) = theta;
  ll_store(0) = ll;
  // compute approximate log-likelihood with proposed theta
  signal = init_signal;
  ll_approx = approx(signal, max_iter, conv_tol);

  y_store.col(0) = y;
  H_store.col(0) = H;

  double ll_approx_u = 0.0;
  for (unsigned int t = 0; t < n; t++) {
    if (arma::is_finite(ng_y(t))) {
      ll_approx_u += ng_y(t) * (signal(t) + xbeta(t)) - phi(t) *
        exp(signal(t) + xbeta(t)) + 0.5 * pow(y(t) - signal(t) - xbeta(t), 2) / HH(t);
    }
  }
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
      signal = init_signal;
      ll_approx = approx(signal, max_iter, conv_tol);
      ll_prop = ll_approx + log_likelihood(distribution != 0);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + q));
    } else accept_prob = 0;


    if (inrange && (unif(engine) < accept_prob)) {
      ll = ll_prop;
      theta = theta_prop;
      ll_approx_u = 0.0;
      for (unsigned int t = 0; t < n; t++) {
        if (arma::is_finite(ng_y(t))) {
          ll_approx_u += ng_y(t) * (signal(t) + xbeta(t)) - phi(t) *
            exp(signal(t) + xbeta(t)) + 0.5 * pow(y(t) - signal(t) - xbeta(t), 2) / HH(t);
        }
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


    double change = accept_prob - target_acceptance;
    u = S * u / arma::norm(u) * sqrt(std::min(1.0, npar * pow(i, -gamma)) *
      std::abs(change));


    if(change > 0) {
      S = cholupdate(S, u);

    } else {
      if(change < 0){
        //update S unless numerical problems occur
        arma::mat Stmp = choldowndate(S, u);
        arma::uvec cond = arma::find(arma::diagvec(Stmp) < 0);
        if (cond.n_elem == 0) {
          S = Stmp;
        }
      }
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
