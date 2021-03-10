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
  fnPtr drift, fnPtr diffusion, fnPtr ddiffusion,
  bool positive, sitmo::prng_engine& eng) {

  unsigned int n = std::pow(2, L);
  double dt = t / n;

  arma::vec dB(n);
  std::normal_distribution<> normal(0.0, std::sqrt(dt));

  for (unsigned int  i=0; i < n; i++){
    dB(i) = normal(eng);
  }

  return milstein_worker(x0, dB, dt, n, theta,
    drift, diffusion, ddiffusion, positive);
}

// A worker which uses simulated Brownian differences
double milstein_worker(double x, arma::vec& dB, double dt, unsigned int n,
  const arma::vec& theta, fnPtr drift, fnPtr diffusion,
  fnPtr ddiffusion, bool positive) {

  for(unsigned int k = 0; k < n; k++) {
    x += drift(x, theta) * dt + diffusion(x, theta) * dB(k) +
      0.5 * diffusion(x, theta) * ddiffusion(x, theta) * (dB(k) * dB(k) - dt);
    if(positive) x = std::abs(x);
  }
  return x;
}
