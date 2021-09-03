#' Example C++ Codes for Non-Linear Models
#'
#' @param example Name of the example model. 
#' Run \code{nlg_example_models("abc")} to get the names of possible models.
#' @param return_code If TRUE, will not compile the model but only returns the 
#' corresponding code.
#'
#' @return Returns pointers to the C++ snippets defining the model, or in case 
#' of \code{return_code = TRUE}, returns the example code without compiling.
#' @export
nlg_example_models <- function(example, return_code = FALSE) {
  
  example <- match.arg(example, c("linear_gaussian", "sin_exp"))
  code <- switch(example,
    "linear_gaussian" = {
      '
      #include <RcppArmadillo.h>
      // [[Rcpp::depends(RcppArmadillo)]]
      // [[Rcpp::interfaces(r, cpp)]]
      
      // [[Rcpp::export]]
      arma::vec a1_fn(const arma::vec& theta, const arma::vec& known_params) {
        
        arma::vec a1(1);
        a1(0) = 0;
        return a1;
      }
      // [[Rcpp::export]]
      arma::mat P1_fn(const arma::vec& theta, const arma::vec& known_params) {
        
        arma::mat P1(1, 1);
        P1(0,0) = 1;
        return P1;
      }
      
      // [[Rcpp::export]]
      arma::mat H_fn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        arma::mat H(1,1);
        H(0, 0) = exp(theta(0));
        return H;
      }
      
      // Function for the Cholesky of state level covariance
      // [[Rcpp::export]]
      arma::mat R_fn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        arma::mat R(1, 1);
        R(0, 0) = 1;
        return R;
      }
      
      
      // Z function
      // [[Rcpp::export]]
      arma::vec Z_fn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        return alpha;
      }
      // Jacobian of Z function
      // [[Rcpp::export]]
      arma::mat Z_gn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        arma::mat Z_gn(1, 1);
        Z_gn(0, 0) = 1.0;
        return Z_gn;
      }
      
      // T function
      // [[Rcpp::export]]
      arma::vec T_fn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        return alpha;
      }
      
      // Jacobian of T function
      // [[Rcpp::export]]
      arma::mat T_gn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        arma::mat Tg(1, 1);
        Tg(0, 0) = 1.0;
        return Tg;
      }
      
      // log-prior pdf for theta
      // [[Rcpp::export]]
      double log_prior_pdf(const arma::vec& theta) {
        return R::dnorm(exp(theta(0)), 0, 1, 1) + theta(0); //jacobian term
      }
      
      // [[Rcpp::export]]
      Rcpp::List create_xptrs() {
        
        // typedef for a pointer of nonlinear function of model equation 
        // returning vec (T, Z)
        typedef arma::vec (*nvec_fnPtr)(const unsigned int t, 
          const arma::vec& alpha, const arma::vec& theta, 
          const arma::vec& known_params, const arma::mat& known_tv_params);
        // typedef for a pointer of nonlinear fn returning mat (Tg, Zg, H, R)
        typedef arma::mat (*nmat_fnPtr)(const unsigned int t, 
          const arma::vec& alpha, const arma::vec& theta, 
          const arma::vec& known_params, const arma::mat& known_tv_params);
        
        // typedef for a pointer returning a1
        typedef arma::vec (*a1_fnPtr)(const arma::vec& theta, 
          const arma::vec& known_params);
        // typedef for a pointer returning P1
        typedef arma::mat (*P1_fnPtr)(const arma::vec& theta, 
          const arma::vec& known_params);
        // typedef for a pointer of log-prior function
        typedef double (*prior_fnPtr)(const arma::vec& theta);
        
        return Rcpp::List::create(
          Rcpp::Named("a1_fn") = Rcpp::XPtr<a1_fnPtr>(new a1_fnPtr(&a1_fn)),
          Rcpp::Named("P1_fn") = Rcpp::XPtr<P1_fnPtr>(new P1_fnPtr(&P1_fn)),
          Rcpp::Named("Z_fn") = Rcpp::XPtr<nvec_fnPtr>(new nvec_fnPtr(&Z_fn)),
          Rcpp::Named("H_fn") = Rcpp::XPtr<nmat_fnPtr>(new nmat_fnPtr(&H_fn)),
          Rcpp::Named("T_fn") = Rcpp::XPtr<nvec_fnPtr>(new nvec_fnPtr(&T_fn)),
          Rcpp::Named("R_fn") = Rcpp::XPtr<nmat_fnPtr>(new nmat_fnPtr(&R_fn)),
          Rcpp::Named("Z_gn") = Rcpp::XPtr<nmat_fnPtr>(new nmat_fnPtr(&Z_gn)),
          Rcpp::Named("T_gn") = Rcpp::XPtr<nmat_fnPtr>(new nmat_fnPtr(&T_gn)),
          Rcpp::Named("log_prior_pdf") = 
            Rcpp::XPtr<prior_fnPtr>(new prior_fnPtr(&log_prior_pdf)));
      }
      '
    },
    "sin_exp" = {
      '
      #include <RcppArmadillo.h>
      // [[Rcpp::depends(RcppArmadillo)]]
      // [[Rcpp::interfaces(r, cpp)]]
      
      // [[Rcpp::export]]
      arma::vec a1_fn(const arma::vec& theta, const arma::vec& known_params) {
        
        arma::vec a1(1);
        a1(0) = 0;
        return a1;
      }
      // [[Rcpp::export]]
      arma::mat P1_fn(const arma::vec& theta, const arma::vec& known_params) {
        
        arma::mat P1(1, 1);
        P1(0,0) = 1;
        return P1;
      }
      
      // [[Rcpp::export]]
      arma::mat H_fn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        arma::mat H(1,1);
        H(0, 0) = exp(theta(0));
        return H;
      }
      
      // Function for the Cholesky of state level covariance
      // [[Rcpp::export]]
      arma::mat R_fn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        arma::mat R(1, 1);
        R(0, 0) = exp(theta(1));
        return R;
      }
      
      // Z function
      // [[Rcpp::export]]
      arma::vec Z_fn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        return exp(alpha);
      }
      // Jacobian of Z function
      // [[Rcpp::export]]
      arma::mat Z_gn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        arma::mat Z_gn(1, 1);
        Z_gn(0, 0) = exp(alpha(0));
        return Z_gn;
      }
      
      // T function
      // [[Rcpp::export]]
      arma::vec T_fn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        return sin(alpha);
      }
      
      // Jacobian of T function
      // [[Rcpp::export]]
      arma::mat T_gn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        arma::mat Tg(1, 1);
        Tg(0, 0) = cos(alpha(0));
        return Tg;
      }
      
      // log-prior pdf for theta
      // [[Rcpp::export]]
      double log_prior_pdf(const arma::vec& theta) {
        return R::dnorm(exp(theta(0)), 0, 1, 1) + theta(0) + 
          R::dnorm(exp(theta(1)), 0, 1, 1) + theta(1);
      }
      
      // [[Rcpp::export]]
      Rcpp::List create_xptrs() {
        
        // typedef for a pointer of nonlinear function of model equation 
        // returning vec (T, Z)
        typedef arma::vec (*nvec_fnPtr)(const unsigned int t, 
          const arma::vec& alpha, const arma::vec& theta, 
          const arma::vec& known_params, const arma::mat& known_tv_params);
        // typedef for a pointer of nonlinear fn returning mat (Tg, Zg, H, R)
        typedef arma::mat (*nmat_fnPtr)(const unsigned int t, 
          const arma::vec& alpha, const arma::vec& theta, 
          const arma::vec& known_params, const arma::mat& known_tv_params);
        
        // typedef for a pointer returning a1
        typedef arma::vec (*a1_fnPtr)(const arma::vec& theta, 
          const arma::vec& known_params);
        // typedef for a pointer returning P1
        typedef arma::mat (*P1_fnPtr)(const arma::vec& theta, 
          const arma::vec& known_params);
        // typedef for a pointer of log-prior function
        typedef double (*prior_fnPtr)(const arma::vec& theta);
        
        return Rcpp::List::create(
          Rcpp::Named("a1_fn") = Rcpp::XPtr<a1_fnPtr>(new a1_fnPtr(&a1_fn)),
          Rcpp::Named("P1_fn") = Rcpp::XPtr<P1_fnPtr>(new P1_fnPtr(&P1_fn)),
          Rcpp::Named("Z_fn") = Rcpp::XPtr<nvec_fnPtr>(new nvec_fnPtr(&Z_fn)),
          Rcpp::Named("H_fn") = Rcpp::XPtr<nmat_fnPtr>(new nmat_fnPtr(&H_fn)),
          Rcpp::Named("T_fn") = Rcpp::XPtr<nvec_fnPtr>(new nvec_fnPtr(&T_fn)),
          Rcpp::Named("R_fn") = Rcpp::XPtr<nmat_fnPtr>(new nmat_fnPtr(&R_fn)),
          Rcpp::Named("Z_gn") = Rcpp::XPtr<nmat_fnPtr>(new nmat_fnPtr(&Z_gn)),
          Rcpp::Named("T_gn") = Rcpp::XPtr<nmat_fnPtr>(new nmat_fnPtr(&T_gn)),
          Rcpp::Named("log_prior_pdf") = 
            Rcpp::XPtr<prior_fnPtr>(new prior_fnPtr(&log_prior_pdf)));
      }
      '
    })
  if (!return_code) {
    # create dummy variable to get rid of "undefined variable" note
    create_xptrs <- NULL
    Rcpp::sourceCpp(code = code)
    create_xptrs()
  } else code
}
