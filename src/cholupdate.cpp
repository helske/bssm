#include <RcppArmadillo.h>
using namespace Rcpp;


// [[Rcpp::export]]
arma::mat cholupdate(arma::mat L, arma::vec u) {
  unsigned int n = u.n_elem - 1;
  for (unsigned int i = 0; i < n; i++) {
    double r = sqrt(pow(L(i,i), 2) + pow(u(i), 2));
    double c = r / L(i, i);
    double s = u(i) / L(i, i);
    L(i, i) = r;
    L(arma::span(i + 1, n), i) =
      (L(arma::span(i + 1, n), i) + s * u.rows(i + 1, n)) / c;
    u.rows(i + 1, n) = c * u.rows(i + 1, n) -
      s * L(arma::span(i + 1, n), i);
  }
  L(n, n) = sqrt(pow(L(n, n), 2) + pow(u(n), 2));
  return L;
}
// [[Rcpp::export]]
arma::mat choldowndate(arma::mat L, arma::vec u) {
  unsigned int n = u.n_elem - 1;
  for (unsigned int i = 0; i < n; i++) {
    double r = sqrt(pow(L(i,i), 2) - pow(u(i), 2));
    double c = r / L(i, i);
    double s = u(i) / L(i, i);
    L(i, i) = r;
    L(arma::span(i + 1, n), i) =
      (L(arma::span(i + 1, n), i) - s * u.rows(i + 1, n)) / c;
    u.rows(i + 1, n) = c * u.rows(i + 1, n) -
      s * L(arma::span(i + 1, n), i);
  }
  L(n, n) = sqrt(pow(L(n, n), 2) - pow(u(n), 2));
  return L;
}


void adjust_S(arma::mat& S, arma::vec& u, double current, double target, unsigned int n, double gamma) {
  
  double change = current - target;
  u = S * u / arma::norm(u) * sqrt(std::min(1.0, u.n_elem * pow(n, -gamma)) *
    std::abs(change));
  
  if(change > 0) {
    S = cholupdate(S, u);
    
  } else {
    if(change < 0){
      //downdate S unless numerical problems occur
      arma::mat Stmp = choldowndate(S, u);
      arma::uvec cond = arma::find(arma::diagvec(Stmp) < 0);
      if (cond.n_elem == 0) {
        S = Stmp;
      } else {
        Rcout<<"numerical issues in S."<<std::endl;
      }
    }
  }
}