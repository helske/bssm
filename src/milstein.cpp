#include "milstein_functions.h"

// Functions for the Milstein scheme

// Calculates the terminal values of the Milstein discretisations of a
// SDE in [0,t] using 2^L levels
// In:
//  x0     -- Starting point
//  mu     -- Drift function
//  diffusion  -- Volatility function
//  ddiffusion -- Derivative of volatility
//  L      -- Discretisation level (2^L)
//  t      -- Terminal time
// Out: list with entries
//  X      -- Terminal value
//  seed   -- Seed of RNG in the beginning of the function

double milstein(const double x0, const unsigned int L, const double t,
  const arma::vec& theta,
  funcPtr drift, funcPtr diffusion, funcPtr ddiffusion,
  bool positive, sitmo::prng_engine& eng) {

  int n = std::pow(2, L);
  double dt = t / n;

  arma::vec dB(n);
  std::normal_distribution<> normal(0.0, std::sqrt(dt));

  for (unsigned int i=0; i < n; i++){
    dB(i) = normal(eng);
  }

  return milstein_worker(x0, dB, dt, n, theta,
    drift, diffusion, ddiffusion, positive);
}

// A worker which uses simulated Brownian differences
double milstein_worker(double x, arma::vec& dB, double dt, int n,
  const arma::vec& theta, funcPtr drift, funcPtr diffusion,
  funcPtr ddiffusion, bool positive) {

  for(unsigned int k = 0; k < n; k++) {
    x += drift(x, theta) * dt + diffusion(x, theta) * dB(k) +
      0.5 * diffusion(x, theta) * ddiffusion(x, theta) * (dB(k) * dB(k) - dt);
    if(positive) x = std::abs(x);
  }
  return x;
}


// Simulate a realisation of the differences of a Brownian bridge on [0,t] with a
// uniform mesh of n points
arma::vec brownian_bridge(const double t, const double sd, const unsigned int n,
  const double X_t, sitmo::prng_engine& eng) {

  double dt = t / n;
  arma::vec dB(n);
  std::normal_distribution<> normal(0.0, sd);

  for (unsigned int i=0; i < n; i++){
    dB(i) = normal(eng);
  }

  double B_t = arma::accu(dB);
  return(dB - dt / t * (B_t - X_t));
}

// Milstein scheme using coarse mesh filled with Brownian bridge
// At the start of the joint filtering, the coarse engine eng_c 
// should be restarted with the original seed used in coarce filter
// in essense we end up simulating lot's of extra increments (2^L_c) but 
// we save even more (?) in memory as we don't need to store all the increments

// note that this approach does not work with thinning and/or parallelisation
// in those cases we need to discard some states in the engine...
double milstein_joint(const double x0,
  const unsigned int L_c, const unsigned int L_f, const double t,
  const arma::vec& theta,
  funcPtr drift, funcPtr diffusion, funcPtr ddiffusion,
  bool positive, sitmo::prng_engine& eng_c, sitmo::prng_engine& eng_f) {

  int n_c = std::pow(2, L_c);
  double dt_c = t / n_c;
  int n_f = std::pow(2, L_f);
  double dt_f = t / n_f;
  int n_d = std::pow(2, L_f - L_c);

  // Coarse-level path, with fixed seed
  
  std::normal_distribution<> normal(0.0, std::sqrt(dt_c));
  arma::vec dB_c(n_c);
  for (unsigned int i=0; i < n_c; i++){
    dB_c(i) = normal(eng_c);
  }

  // fine-level path, independent engine
  arma::mat dB_f_(n_d, n_c);
  for (unsigned int k = 0; k < n_c; k++){
    dB_f_.col(k) = brownian_bridge(dt_c, dt_f, n_d, dB_c(k), eng_f);
  }
  arma::vec dB_f = arma::vectorise(dB_f_);

  // Calculate the actual path of the finer mesh:
  return milstein_worker(x0, dB_f, dt_f, n_f,
    theta, drift, diffusion, ddiffusion, positive);
}
