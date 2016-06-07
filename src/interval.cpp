#include <boost/function.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/tools/roots.hpp>
#include <RcppArmadillo.h>
using namespace Rcpp;


struct objective_gaussian {
  objective_gaussian(arma::vec means, arma::vec sds, double prob) : means(means), sds(sds), prob(prob) {}

  double operator()(double b) const {
    return sum(pnorm(as<NumericVector>(wrap((b - means)/sds))))/means.n_elem - prob;
  }

private:
  arma::vec means;
  arma::vec sds;
  double prob;
};

// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat intervals(arma::mat& means, arma::mat& sds, arma::vec& probs, unsigned int n_ahead) {

  int digits = std::numeric_limits<double>::digits;
  boost::math::tools::eps_tolerance<double> tol(digits);

  arma::mat intv(n_ahead, probs.n_elem);

  for (unsigned int j = 0; j < probs.n_elem; j++) {
    double guess = 0.0;
    for (unsigned int k = 0; k < means.n_rows; k++) {
      guess += R::qnorm(probs(j), means(k, 0), sds(k, 0), 1, 0);
    }
    guess /= means.n_rows;
    for (unsigned int i = 0; i < n_ahead; i++) {
      boost::uintmax_t maxit = 100;
      objective_gaussian f(means.col(i), sds.col(i), probs(j));
      std::pair<double, double> r = boost::math::tools::bracket_and_solve_root(f, guess, 2.0, true, tol, maxit);

      intv(i, j) = r.first + (r.second - r.first) / 2.0;
      guess = intv(i, j);
    }
  }

  return intv;
}
