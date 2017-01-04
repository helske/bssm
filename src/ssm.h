#ifndef SSM_H
#define SSM_H

#include <RcppArmadillo.h>

class ssm {
  
public:
  virtual void set_theta(arma::vec) = 0;
  virtual arma::vec get_theta() = 0;
  virtual double log_likelihood(unsigned int, unsigned int) = 0;
  virtual arma::cube simulate_states(unsigned int) = 0;
  virtual double log_prior_pdf(const arma::vec&, const arma::uvec&, const arma::mat&);
  virtual double proposal(const arma::vec& theta, const arma::vec& theta_proposal) {
    return 0.0;
  }
};


#endif
