#include <boost/function.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/tools/roots.hpp>
#include <RcppArmadillo.h>

struct objective_gaussian {
  objective_gaussian(const arma::vec& means, const arma::vec& sds, double prob) : means(means), sds(sds), prob(prob) {}

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

  int digits = std::numeric_limits<double>::digits;
  boost::math::tools::eps_tolerance<double> tol(digits);

  arma::mat intv(n_ahead, probs.n_elem);

  for (unsigned int i = 0; i < n_ahead; i++) {
    for (unsigned int j = 0; j < probs.n_elem; j++) {
      double guess = 0.0;
      for (unsigned int k = 0; k < means.n_rows; k++) {
        guess += R::qnorm(probs(j), means(k, i), sds(k, i), 1, 0);
      }
      guess /= means.n_rows;
      boost::uintmax_t maxit = 1000;
      objective_gaussian f(means.col(i), sds.col(i), probs(j));
      std::pair<double, double> r =
        boost::math::tools::bracket_and_solve_root(f, guess, 2.0, true, tol, maxit);
      if(maxit >= 1000) {
        r = boost::math::tools::bracket_and_solve_root(f, -guess, 2.0, true, tol, maxit);
      }
      intv(i, j) = r.first + (r.second - r.first) / 2.0;
      guess = intv(i, j);
    }
  }
  return intv;
}
