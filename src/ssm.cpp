#include <RcppArmadillo.h>
#include "ssm.h"

//log-prior_pdf
// type 0 = uniform distribution
// type 1 = half-normal
// type 2 = normal
//
double ssm::log_prior_pdf(const arma::vec& theta, const arma::uvec& distributions,
  const arma::mat& parameters) {
  
  double q = 0.0;
  for(unsigned int i = 0; i < theta.n_elem; i++) {
    switch(distributions(i)) {
    case 0  :
      q += R::dunif(theta(i), parameters(0, i), parameters(1, i), 1);
      break;
    case 1  :
      if (theta(i) < 0) {
        return -arma::datum::inf;
      } else {
        q += log(2.0) + R::dnorm(theta(i), 0, parameters(0, i), 1);
      }
      break;
    case 2  :
      q += R::dnorm(theta(i), parameters(0, i), parameters(1, i), 1);
      break;
    }
  }
  return q;
}


