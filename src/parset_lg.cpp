#include "parset_lg.h"
#include "model_ssm_ulg.h"
#include "model_ssm_mlg.h"

parset_ulg::parset_ulg(const ssm_ulg& model, const arma::mat& theta, const Rcpp::Function update_fn) {
  
  Rcpp::List model_list = update_fn(Rcpp::NumericVector(model.theta.begin(), model.theta.end()));
  
  n = theta.n_cols;
  est_H = model_list.containsElementNamed("H");
  est_Z = model_list.containsElementNamed("Z");
  est_T = model_list.containsElementNamed("T");
  est_R = model_list.containsElementNamed("R");
  est_a1 = model_list.containsElementNamed("a1");
  est_P1 = model_list.containsElementNamed("P1");
  est_D = model_list.containsElementNamed("D");
  est_C = model_list.containsElementNamed("C");
  est_beta = model_list.containsElementNamed("beta");
  
  
  H = arma::mat(model.H.n_elem, n * est_H);
  Z = arma::cube(model.Z.n_rows, model.Z.n_cols, n * est_Z);
  T = arma::field<arma::cube>(n * est_T);
  R = arma::field<arma::cube>(n * est_R);
  a1 = arma::mat(model.a1.n_elem,  n * est_a1);
  P1 = arma::cube(model.P1.n_rows, model.P1.n_cols, n * est_P1);
  D = arma::mat(model.D.n_elem, n * est_D);
  C = arma::cube(model.C.n_rows, model.C.n_cols, n * est_C);
  beta = arma::mat(model.beta.n_elem, n * est_beta);
  
  for(unsigned int i = 0; i < n; i++) {
    Rcpp::NumericVector theta0(theta.col(i).begin(), theta.col(i).end());
    model_list = update_fn(theta0);
    
    if (est_H) {
      H.col(i) = Rcpp::as<arma::vec>(model_list["H"]);
    }
    
    if (est_Z) {
      Z.slice(i) = Rcpp::as<arma::mat>(model_list["Z"]);
    }
    
    if (est_T) {
      // need to create intermediate object in order to avoid memory issues
      arma::cube tcube = Rcpp::as<arma::cube>(model_list["T"]);
      T(i) = tcube;
    }
    
    if (est_R) {
      // need to create intermediate object in order to avoid memory issues
      arma::cube rcube = Rcpp::as<arma::cube>(model_list["R"]);
      R(i) = rcube;
    }
    if (est_a1) {
      a1.col(i) = Rcpp::as<arma::vec>(model_list["a1"]);
    }
    if (est_P1) {
      P1.slice(i) = Rcpp::as<arma::mat>(model_list["P1"]);
    }
    if (est_D) {
      D.col(i) = Rcpp::as<arma::vec>(model_list["D"]);
    }
    if (est_C) {
      C.slice(i) = Rcpp::as<arma::mat>(model_list["C"]);
    }
    
    if (est_beta) {
      beta.col(i) = Rcpp::as<arma::vec>(model_list["beta"]);
    }
  }
}


void parset_ulg::update(ssm_ulg& model, const unsigned int i) {
  
  if (est_H) {
    model.H = H.col(i);
    model.compute_HH();
  }
  if (est_Z) {
    model.Z = Z.slice(i);
  }
  if (est_T) {
    model.T = T(i);
  }
  
  if (est_R) {
    model.R = R(i);
    model.compute_RR();
  }
  if (est_a1) {
    model.a1 = a1.col(i);
  }
  if (est_P1) {
    model.P1 = P1.slice(i);
  }
  if (est_D) {
    model.D = D.col(i);
  }
  if (est_C) {
    model.C = C.slice(i);
  }
  
  if (est_beta) {
    model.beta = beta.col(i);
    model.compute_xbeta();
  }
  
}


parset_mlg::parset_mlg(const ssm_mlg& model, const arma::mat& theta, const Rcpp::Function update_fn) {
  
  Rcpp::List model_list = update_fn(Rcpp::NumericVector(model.theta.begin(), model.theta.end()));
  
  n = theta.n_cols;
  est_H = model_list.containsElementNamed("H");
  est_Z = model_list.containsElementNamed("Z");
  est_T = model_list.containsElementNamed("T");
  est_R = model_list.containsElementNamed("R");
  est_a1 = model_list.containsElementNamed("a1");
  est_P1 = model_list.containsElementNamed("P1");
  est_D = model_list.containsElementNamed("D");
  est_C = model_list.containsElementNamed("C");
  
  
  H = arma::field<arma::cube>(n * est_H);
  Z = arma::field<arma::cube>(n * est_Z);
  T = arma::field<arma::cube>(n * est_T);
  R = arma::field<arma::cube>(n * est_R);
  a1 = arma::mat(model.a1.n_elem,  n * est_a1);
  P1 = arma::cube(model.P1.n_rows, model.P1.n_cols, n * est_P1);
  D = arma::cube(model.D.n_rows, model.D.n_cols, n * est_D);
  C = arma::cube(model.C.n_rows, model.C.n_cols, n * est_C);
  
  for(unsigned int i = 0; i < n; i++) {
    
    Rcpp::NumericVector theta0(theta.col(i).begin(), theta.col(i).end());
    model_list = update_fn(theta0);
    
    if (est_H) {
      // need to create intermediate object in order to avoid memory issues
      arma::cube hcube = Rcpp::as<arma::cube>(model_list["H"]);
      H(i) = hcube;
    }
    if (est_Z) {
      // need to create intermediate object in order to avoid memory issues
      arma::cube zcube = Rcpp::as<arma::cube>(model_list["Z"]);
      Z(i) = zcube;
    }
    
    if (est_T) {
      // need to create intermediate object in order to avoid memory issues
      arma::cube tcube = Rcpp::as<arma::cube>(model_list["T"]);
      T(i) = tcube;
    }
    
    if (est_R) {
      // need to create intermediate object in order to avoid memory issues
      arma::cube rcube = Rcpp::as<arma::cube>(model_list["R"]);
      R(i) = rcube;
    }
    if (est_a1) {
      a1.col(i) = Rcpp::as<arma::vec>(model_list["a1"]);
    }
    if (est_P1) {
      P1.slice(i) = Rcpp::as<arma::mat>(model_list["P1"]);
    }
    if (est_D) {
      D.slice(i) = Rcpp::as<arma::mat>(model_list["D"]);
    }
    if (est_C) {
      C.slice(i) = Rcpp::as<arma::mat>(model_list["C"]);
    }
  }
}


void parset_mlg::update(ssm_mlg& model, const unsigned int i) {
  
  if (est_H) {
    model.H = H(i);
    model.compute_HH();
  }
  if (est_Z) {
    model.Z = Z(i);
  }
  if (est_T) {
    model.T = T(i);
  }
  if (est_R) {
    model.R = R(i);
    model.compute_RR();
  }
  if (est_a1) {
    model.a1 = a1.col(i);
  }
  if (est_P1) {
    model.P1 = P1.slice(i);
  }
  if (est_D) {
    model.D = D.slice(i);
  }
  if (est_C) {
    model.C = C.slice(i);
  }
  
}

