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

/*** R
k<-3
A <- crossprod(matrix(rnorm(k ^ 2), k, k))
L1 <- t(chol(A))
u <- rnorm(k)
library("microbenchmark")
f <- function(L1, u) t(chol(L1%*%t(L1) + u%*%t(u)))
f2 <- function(L1, u) t(chol(L1%*%t(L1) - u%*%t(u)))
bssm:::cholupdate(L1, u)
f(L1,u)
bssm:::choldowndate(L1, u)
f2(L1,u)
*/
