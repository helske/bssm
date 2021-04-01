#include "parset_gsv.h"
#include "model_ssm_gsv.h"

parset_gsv::parset_gsv(const ssm_gsv& model, const arma::mat& theta, const Rcpp::Function update_fn) {
  
  Rcpp::List model_list = update_fn(Rcpp::NumericVector(model.theta.begin(), model.theta.end()));
  
  n = theta.n_cols;
  est_Z_mu = model_list.containsElementNamed("Z_mu");
  est_T_mu = model_list.containsElementNamed("T_mu");
  est_R_mu = model_list.containsElementNamed("R_mu");
  est_a1_mu = model_list.containsElementNamed("a1_mu");
  est_P1_mu = model_list.containsElementNamed("P1_mu");
  est_D_mu = model_list.containsElementNamed("D_mu");
  est_C_mu = model_list.containsElementNamed("C_mu");
  
  est_Z_sv = model_list.containsElementNamed("Z_sv");
  est_T_sv = model_list.containsElementNamed("T_sv");
  est_R_sv = model_list.containsElementNamed("R_sv");
  est_a1_sv = model_list.containsElementNamed("a1_sv");
  est_P1_sv = model_list.containsElementNamed("P1_sv");
  est_D_sv = model_list.containsElementNamed("D_sv");
  est_C_sv = model_list.containsElementNamed("C_sv");
  
  Z_mu = arma::cube(model.Z_mu.n_rows, model.Z_mu.n_cols, n * est_Z_mu);
  T_mu = arma::field<arma::cube>(n * est_T_mu);
  R_mu = arma::field<arma::cube>(n * est_R_mu);
  a1_mu = arma::mat(model.a1_mu.n_elem,  n * est_a1_mu);
  P1_mu = arma::cube(model.P1_mu.n_rows, model.P1_mu.n_cols, n * est_P1_mu);
  D_mu = arma::mat(model.D_mu.n_elem, n * est_D_mu);
  C_mu = arma::cube(model.C_mu.n_rows, model.C_mu.n_cols, n * est_C_mu);
  
  Z_sv = arma::cube(model.Z_sv.n_rows, model.Z_sv.n_cols, n * est_Z_sv);
  T_sv = arma::field<arma::cube>(n * est_T_sv);
  R_sv = arma::field<arma::cube>(n * est_R_sv);
  a1_sv = arma::mat(model.a1_sv.n_elem,  n * est_a1_sv);
  P1_sv = arma::cube(model.P1_sv.n_rows, model.P1_sv.n_cols, n * est_P1_sv);
  D_sv = arma::mat(model.D_sv.n_elem, n * est_D_sv);
  C_sv = arma::cube(model.C_sv.n_rows, model.C_sv.n_cols, n * est_C_sv);
  
  for(unsigned int i = 0; i < n; i++) {
    Rcpp::NumericVector theta0(theta.col(i).begin(), theta.col(i).end());
    model_list = update_fn(theta0);
  
    if (est_Z_mu) {
      Z_mu.slice(i) = Rcpp::as<arma::mat>(model_list["Z_mu"]);
    }
    if (est_T_mu) {
      // need to create intermediate object in order to avoid memory issues
      arma::cube tcube = Rcpp::as<arma::cube>(model_list["T_mu"]);
      T_mu(i) = tcube;
    }
    if (est_R_mu) {
      // need to create intermediate object in order to avoid memory issues
      arma::cube rcube = Rcpp::as<arma::cube>(model_list["R_mu"]);
      R_mu(i) = rcube;
    }
    if (est_a1_mu) {
      a1_mu.col(i) = Rcpp::as<arma::vec>(model_list["a1_mu"]);
    }
    if (est_P1_mu) {
      P1_mu.slice(i) = Rcpp::as<arma::mat>(model_list["P1_mu"]);
    }
    if (est_D_mu) {
      D_mu.col(i) = Rcpp::as<arma::vec>(model_list["D_mu"]);
    }
    if (est_C_mu) {
      C_mu.slice(i) = Rcpp::as<arma::mat>(model_list["C_mu"]);
    }

    if (est_Z_sv) {
      Z_sv.slice(i) = Rcpp::as<arma::mat>(model_list["Z_sv"]);
    }
    if (est_T_sv) {
      // need to create intermediate object in order to avoid memory issues
      arma::cube tcube = Rcpp::as<arma::cube>(model_list["T_sv"]);
      T_sv(i) = tcube;
    }
    if (est_R_sv) {
      // need to create intermediate object in order to avoid memory issues
      arma::cube rcube = Rcpp::as<arma::cube>(model_list["R_sv"]);
      R_sv(i) = rcube;
    }
    if (est_a1_sv) {
      a1_sv.col(i) = Rcpp::as<arma::vec>(model_list["a1_sv"]);
    }
    if (est_P1_sv) {
      P1_sv.slice(i) = Rcpp::as<arma::mat>(model_list["P1_sv"]);
    }
    if (est_D_sv) {
      D_sv.col(i) = Rcpp::as<arma::vec>(model_list["D_sv"]);
    }
    if (est_C_sv) {
      C_sv.slice(i) = Rcpp::as<arma::mat>(model_list["C_sv"]);
    }
  }
}


void parset_gsv::update(ssm_gsv& model, const unsigned int i) {
  
  bool update_R = false;
  if (est_Z_mu) {
    model.Z_mu = Z_mu.slice(i);
  }
  if (est_T_mu) {
    model.T_mu = T_mu(i);
  }
  if (est_R_mu) {
    update_R = true;
    model.R_mu = R_mu(i);
  }
  if (est_a1_mu) {
    model.a1_mu = a1_mu.col(i);
  }
  if (est_P1_mu) {
    model.P1_mu = P1_mu.slice(i);
  }
  if (est_D_mu) {
    model.D_mu = D_mu.col(i);
  }
  if (est_C_mu) {
    model.C_mu = C_mu.slice(i);
  }
  if (est_Z_sv) {
    model.Z_sv = Z_sv.slice(i);
  }
  if (est_T_sv) {
    model.T_sv = T_sv(i);
  }
  if (est_R_sv) {
    update_R = true;
    model.R_sv = R_sv(i);
  }
  if (est_a1_sv) {
    model.a1_sv = a1_sv.col(i);
  }
  if (est_P1_sv) {
    model.P1_sv = P1_sv.slice(i);
  }
  if (est_D_sv) {
    model.D_sv = D_sv.col(i);
  }
  if (est_C_sv) {
    model.C_sv = C_sv.slice(i);
  }
  if(update_R) model.compute_RR();
  
  // approximation does not match theta anymore (keep as -1 if so)
  if (model.approx_state > 0) model.approx_state = 0;
}

