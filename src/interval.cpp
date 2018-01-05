#include <boost/function.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/tools/roots.hpp>
#include "bssm.h"

struct objective_gaussian {
  objective_gaussian(const arma::vec& means, const arma::vec& sds, double prob) : 
  means(means), sds(sds), prob(prob) {}
  
  double operator()(double b) const {
    return Rcpp::sum(Rcpp::pnorm(Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap((b - means)/sds))))/means.n_elem - prob;
  }
  
private:
  arma::vec means;
  arma::vec sds;
  double prob;
};

// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppArmadillo)]]
arma::mat intervals(arma::mat& means, const arma::mat& sds, const arma::vec& probs, unsigned int n_ahead) {
 
  boost::math::tools::eps_tolerance<double> tol;
  
  arma::mat intv(n_ahead, probs.n_elem);
  
  for (unsigned int i = 0; i < n_ahead; i++) {
    double lower = means.col(i).min() - 2 * sds.col(i).max();
    double upper = means.col(i).max() + 2 * sds.col(i).max();
    for (unsigned int j = 0; j < probs.n_elem; j++) {
      boost::uintmax_t max_iter = 1000;
      objective_gaussian f(means.col(i), sds.col(i), probs(j));
      std::pair<double, double> r =
        boost::math::tools::bisect(f, lower, upper, tol, max_iter);
      if (!tol(r.first, r.second) || (max_iter >= 1000)) {
        max_iter = 10000;
        r =  boost::math::tools::bisect(f, 1000 * lower, 1000 * upper, tol, max_iter);
      }
      intv(i, j) = r.first + (r.second - r.first) / 2.0;
      lower = intv(i, j);
    }
    
  }
  return intv;
}
