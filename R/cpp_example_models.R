#' Example C++ Codes for Non-Linear and SDE Models
#'
#' @param example Name of the example model. 
#' Run \code{cpp_example_model("abc")} to get the names of possible models.
#' @param return_code If TRUE, will not compile the model but only returns the 
#' corresponding code.
#' @return Returns pointers to the C++ snippets defining the model, or in case 
#' of \code{return_code = TRUE}, returns the example code without compiling.
#' @export
#' @srrstats {G5.4} sde_gbm model used in Vihola, Helske, Franks, (2020). See 
#' also tests/testthat/test_sde.R.
#' @examples
#' cpp_example_model("sde_poisson_OU", return_code = TRUE)
#' 
cpp_example_model <- function(example, return_code = FALSE) {
  
  example <- match.arg(tolower(example), c("nlg_linear_gaussian", 
    "nlg_sin_exp", "nlg_growth", "nlg_ar_exp", "sde_poisson_ou", "sde_gbm"))
  
  if (!test_flag(return_code)) 
    stop("Argument 'return_code' should be TRUE or FALSE. ")
  
  code <- switch(example,
    "sde_poisson_ou" = {
      '
      // A latent Ornstein-Uhlenbeck process with Poisson observations
      // dalpha_t = rho (nu - alpha_t) dt + sigma dB_t, t>=0
      // y_k ~ Poisson(exp(alpha_k)), k = 1,...,n
      
      #include <RcppArmadillo.h>
      // [[Rcpp::depends(RcppArmadillo)]]
      // [[Rcpp::interfaces(r, cpp)]]
      
      // x: state
      // theta: vector of parameters
      
      // theta(0) = log_rho
      // theta(1) = nu
      // theta(2) = log_sigma
      
      // Drift function
      // [[Rcpp::export]]
      double drift(const double x, const arma::vec& theta) {
        return exp(theta(0)) * (theta(1) - x);
      }
      // diffusion function
      // [[Rcpp::export]]
      double diffusion(const double x, const arma::vec& theta) {
        return exp(theta(2));
      }
      // Derivative of the diffusion function
      // [[Rcpp::export]]
      double ddiffusion(const double x, const arma::vec& theta) {
        return 0.0;
      }
      
      // log-density of the prior
      // [[Rcpp::export]]
      double log_prior_pdf(const arma::vec& theta) {
        
        // rho ~ gamma(2, 0.5) // shape-scale parameterization
        // nu ~ N(0, 4)
        // sigma ~ half-N(0,1) (theta(2) is log(sigma))
        double log_pdf = 
          R::dgamma(exp(theta(0)), 2, 0.5, 1) + 
          R::dnorm(theta(1), 0, 4, 1) + 
          R::dnorm(exp(theta(2)), 0, 1, 1) + 
          theta(0) + theta(2); // jacobians of transformations
        return log_pdf;
      }
      
      // log-density of observations
      // given vector of sampled states alpha
      // [[Rcpp::export]]
      arma::vec log_obs_density(const double y, 
        const arma::vec& alpha, const arma::vec& theta) {
        
        arma::vec log_pdf(alpha.n_elem);
        for (unsigned int i = 0; i < alpha.n_elem; i++) {
          log_pdf(i) = R::dpois(y, exp(alpha(i)), 1);
        }
        return log_pdf;
      }
      
      // Function returning the pointers to above functions (no need to modify)
      
      // [[Rcpp::export]]
      Rcpp::List create_xptrs() {
        // typedef for a pointer of drift/volatility function
        typedef double (*fnPtr)(const double x, const arma::vec& theta);
        // typedef for log_prior_pdf
        typedef double (*prior_fnPtr)(const arma::vec& theta);
        // typedef for log_obs_density
        typedef arma::vec (*obs_fnPtr)(const double y, 
          const arma::vec& alpha, const arma::vec& theta);
        
        return Rcpp::List::create(
          Rcpp::Named("drift") = Rcpp::XPtr<fnPtr>(new fnPtr(&drift)),
          Rcpp::Named("diffusion") = Rcpp::XPtr<fnPtr>(new fnPtr(&diffusion)),
          Rcpp::Named("ddiffusion") = 
          Rcpp::XPtr<fnPtr>(new fnPtr(&ddiffusion)),
          Rcpp::Named("prior") = 
          Rcpp::XPtr<prior_fnPtr>(new prior_fnPtr(&log_prior_pdf)),
          Rcpp::Named("obs_density") = 
          Rcpp::XPtr<obs_fnPtr>(new obs_fnPtr(&log_obs_density)));
      }
    ' 
    },
    "sde_gbm" = {
      ' 
      // A latent Geometric Brownian motion with Gaussian observations
      // dx_t = mu * x_t * dt + sigma_x * x_t * dB_t, t>=0, 
      // y_k ~ N(log(x_k), sigma_y^2), k = 1,...,n
      // See Vihola, Helske, and Franks (2020)
      
      // x: state
      // theta: vector of parameters
      
      // theta(0) = mu
      // theta(1) = sigma_x
      // theta(2) = sigma_y
      
      #include <RcppArmadillo.h>
      // [[Rcpp::depends(RcppArmadillo)]]
      // [[Rcpp::interfaces(r, cpp)]]
      
      // Drift function
      // [[Rcpp::export]]
      double drift(const double x, const arma::vec& theta) {
        return theta(0) * x;
      }
      // diffusion function
      // [[Rcpp::export]]
      double diffusion(const double x, const arma::vec& theta) {
        return std::max(0.0, theta(1) * x);
      }
      // Derivative of the diffusion function
      // [[Rcpp::export]]
      double ddiffusion(const double x, const arma::vec& theta) {
        return theta(1) * (x > 0.0);
      }
      
      // log-density of the prior
      // [[Rcpp::export]]
      double log_prior_pdf(const arma::vec& theta) {
      
        double log_pdf = 0.0;
      
        if(theta(0) < 0 || theta(1) < 0 || theta(2) < 0.5) {
          log_pdf = -std::numeric_limits<double>::infinity();
        }
        else {
          log_pdf = R::dnorm(theta(0), 0, 0.1, 1) +
            R::dnorm(theta(1), 0, 0.5, 1) +
            R::dnorm(theta(2), 1.5, 0.5, 1);
        }
        return log_pdf;
      }
      
      // log-density of observations
      // given vector of sampled states alpha
      // [[Rcpp::export]]
      arma::vec log_obs_density(const double y, 
        const arma::vec& alpha, const arma::vec& theta) {
        
        arma::vec log_pdf(alpha.n_elem);
        for (unsigned int i = 0; i < alpha.n_elem; i++) {
          log_pdf(i) = R::dnorm(y, log(alpha(i)), theta(2), 1);
        }
        return log_pdf;
      }
      
      // [[Rcpp::export]]
      Rcpp::List create_xptrs() {
        // typedef for a pointer of drift/volatility function
        typedef double (*fnPtr)(const double x, const arma::vec& theta);
        // typedef for log_prior_pdf
        typedef double (*prior_fnPtr)(const arma::vec& theta);
        // typedef for log_obs_density
        typedef arma::vec (*obs_fnPtr)(const double y, 
          const arma::vec& alpha, const arma::vec& theta);
        
        return Rcpp::List::create(
          Rcpp::Named("drift") = Rcpp::XPtr<fnPtr>(new fnPtr(&drift)),
          Rcpp::Named("diffusion") = Rcpp::XPtr<fnPtr>(new fnPtr(&diffusion)),
          Rcpp::Named("ddiffusion") = Rcpp::XPtr<fnPtr>(new fnPtr(&ddiffusion)),
          Rcpp::Named("prior") = 
            Rcpp::XPtr<prior_fnPtr>(new prior_fnPtr(&log_prior_pdf)),
          Rcpp::Named("obs_density") = 
            Rcpp::XPtr<obs_fnPtr>(new obs_fnPtr(&log_obs_density)));
      }
  '
    },
    "nlg_ar_exp" = {
      '
      // alpha_t+1 = (1-rho)mu + rho * alpha_t + eta_t, eta_t ~ N(0, sigma_x^2)
      // y_t ~ N(exp(alpha_t), sigma_y^2)
      
      // theta(0) = mu
      // theta(1) = rho
      // theta(2) = sigma_x
      // theta(3) = sigma_y
      
      #include <RcppArmadillo.h>
      // [[Rcpp::depends(RcppArmadillo)]]
      // [[Rcpp::interfaces(r, cpp)]]
      
      // Function for the prior mean of alpha_1
      // [[Rcpp::export]]
      arma::vec a1_fn(const arma::vec& theta, const arma::vec& known_params) {
        
        arma::vec a1(1);
        a1(0) = theta(0);
        return a1;
      }
      // Function for the prior covariance matrix of alpha_1
      // [[Rcpp::export]]
      arma::mat P1_fn(const arma::vec& theta, const arma::vec& known_params) {
        
        arma::mat P1(1,1);
        P1(0,0) = pow(exp(theta(2)), 2) / (1 - pow(theta(1), 2));
        return P1;
      }
      
      // Function for the observational level standard deviation
      // [[Rcpp::export]]
      arma::mat H_fn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        arma::mat H(1,1);
        H(0, 0) = exp(theta(3));
        return H;
      }
      
      // Function for the Cholesky of state level covariance
      // [[Rcpp::export]]
      arma::mat R_fn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        arma::mat R(1, 1);
        R(0, 0) = exp(theta(2));
        return R;
      }
      
      
      // Z function
      // [[Rcpp::export]]
      arma::vec Z_fn(const unsigned int t, const arma::vec& alpha, const 
        arma::vec& theta, const arma::vec& known_params, 
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
        
        return theta(0) * (1 - theta(1)) + theta(1) * alpha;
      }
      
      // Jacobian of T function
      // [[Rcpp::export]]
      arma::mat T_gn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        arma::mat Tg(1, 1);
        Tg(0, 0) = theta(1);
        
        return Tg;
      }
      
      //  log-prior pdf for theta
      // [[Rcpp::export]]
      double log_prior_pdf(const arma::vec& theta) {
        
        double log_pdf = 
          R::dnorm(theta(0), 0, 10, 1) + // N(0,10) for mu
          R::dbeta(theta(1), 2, 2, 1) +  // beta(2, 2) for rho
          R::dnorm(exp(theta(2)), 0, 1, 1) + theta(2) +
          R::dnorm(exp(theta(3)), 0, 1, 1) + theta(3);//half-N(0, 1) for sigmas
        
        return log_pdf;
      }
      
      // [[Rcpp::export]]
      Rcpp::List create_xptrs() {
        // typedef for a pointer of nonlinear function returning vec
        typedef arma::vec (*vec_fnPtr)(const unsigned int t, 
          const arma::vec& alpha, const arma::vec& theta, 
          const arma::vec& known_params, const arma::mat& known_tv_params);
        // for a pointer of nonlinear function of model equation returning mat
        typedef arma::mat (*mat_fnPtr)(const unsigned int t, 
          const arma::vec& alpha, const arma::vec& theta, 
          const arma::vec& known_params, const arma::mat& known_tv_params);
        // for a pointer of nonlinear function of model equation returning vec
        typedef arma::vec (*vec_initfnPtr)(const arma::vec& theta, 
          const arma::vec& known_params);
        // for a pointer of nonlinear function of model equation returning mat
        typedef arma::mat (*mat_initfnPtr)(const arma::vec& theta, 
          const arma::vec& known_params);
        // typedef for a pointer of log-prior function
        typedef double (*double_fnPtr)(const arma::vec&);
        
        return Rcpp::List::create(
          Rcpp::Named("a1_fn") = 
            Rcpp::XPtr<vec_initfnPtr>(new vec_initfnPtr(&a1_fn)),
          Rcpp::Named("P1_fn") = 
            Rcpp::XPtr<mat_initfnPtr>(new mat_initfnPtr(&P1_fn)),
          Rcpp::Named("Z_fn") = Rcpp::XPtr<vec_fnPtr>(new vec_fnPtr(&Z_fn)),
          Rcpp::Named("H_fn") = Rcpp::XPtr<mat_fnPtr>(new mat_fnPtr(&H_fn)),
          Rcpp::Named("T_fn") = Rcpp::XPtr<vec_fnPtr>(new vec_fnPtr(&T_fn)),
          Rcpp::Named("R_fn") = Rcpp::XPtr<mat_fnPtr>(new mat_fnPtr(&R_fn)),
          Rcpp::Named("Z_gn") = Rcpp::XPtr<mat_fnPtr>(new mat_fnPtr(&Z_gn)),
          Rcpp::Named("T_gn") = Rcpp::XPtr<mat_fnPtr>(new mat_fnPtr(&T_gn)),
          Rcpp::Named("log_prior_pdf") = 
            Rcpp::XPtr<double_fnPtr>(new double_fnPtr(&log_prior_pdf)));
      }
      '
    },
    "nlg_growth" = {
      '
      //univariate growth model (see vignette growth_model)
      
      #include <RcppArmadillo.h>
      // [[Rcpp::depends(RcppArmadillo)]]
      // [[Rcpp::interfaces(r, cpp)]]
      
      // Unknown parameters theta:
      // theta(0) = log(H)
      // theta(1) = log(R_1)
      // theta(2) = log(R_2)
        
      // Function for the prior mean of alpha_1
      // [[Rcpp::export]]
      arma::vec a1_fn(const arma::vec& theta, const arma::vec& known_params) {
          
        arma::vec a1(2);
        a1(0) = known_params(2);
        a1(1) = known_params(3);
        return a1;
      }
      // Function for the prior covariance matrix of alpha_1
      // [[Rcpp::export]]
      arma::mat P1_fn(const arma::vec& theta, const arma::vec& known_params) {
          
        arma::mat P1(2, 2, arma::fill::zeros);
        P1(0,0) = known_params(4);
        P1(1,1) = known_params(5);
        return P1;
      }
        
      // Function for the observational level standard deviation
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
        
        arma::mat R(2, 2, arma::fill::zeros);
        R(0, 0) = exp(theta(1));
        R(1, 1) = exp(theta(2));
        return R;
      }
        
      // Z function
      // [[Rcpp::export]]
      arma::vec Z_fn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        arma::vec tmp(1);
        tmp(0) = alpha(1);
        return tmp;
      }
      // Jacobian of Z function
      // [[Rcpp::export]]
      arma::mat Z_gn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        arma::mat Z_gn(1, 2);
        Z_gn(0, 0) = 0.0;
        Z_gn(0, 1) = 1.0;
        return Z_gn;
      }
      
      // T function
      // [[Rcpp::export]]
      arma::vec T_fn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        double dT = known_params(0);
        double K = known_params(1);
        
        arma::vec alpha_new(2);
        alpha_new(0) = alpha(0);
        double r = exp(alpha(0)) / (1.0 + exp(alpha(0)));
        alpha_new(1) = K * alpha(1) * exp(r * dT) / 
          (K + alpha(1) * (exp(r * dT) - 1));
        return alpha_new;
      }
        
      // Jacobian of T function
      // [[Rcpp::export]]
      arma::mat T_gn(const unsigned int t, const arma::vec& alpha, 
        const arma::vec& theta, const arma::vec& known_params, 
        const arma::mat& known_tv_params) {
        
        double dT = known_params(0);
        double K = known_params(1);
        double r = exp(alpha(0)) / (1 + exp(alpha(0)));
        double tmp = 
          exp(r * dT) / std::pow(K + alpha(1) * (exp(r * dT) - 1), 2);
          
        arma::mat Tg(2, 2);
        Tg(0, 0) = 1.0;
        Tg(0, 1) = 0;
        Tg(1, 0) = 
          dT * K * alpha(1) * (K - alpha(1)) * tmp * r / (1 + exp(alpha(0)));
        Tg(1, 1) = K * K * tmp;
          
        return Tg;
      }
        
      // log-prior pdf for theta
      // [[Rcpp::export]]
      double log_prior_pdf(const arma::vec& theta) {
        
        // weakly informative half-N(0, 4) priors. 
        // Note that the sampling is on log-scale, 
        // so we need to add jacobians of the corresponding transformations
        // we could also sample on natural scale with check such as
        // if(arma::any(theta < 0))
        //  return -std::numeric_limits<double>::infinity();
        // but this would be less efficient.
          
        // You can use R::dnorm and similar functions, see, e.g.
        // https://teuder.github.io/rcpp4everyone_en/220_dpqr_functions.html
        double log_pdf =  
          R::dnorm(exp(theta(0)), 0, 2, 1) +
          R::dnorm(exp(theta(1)), 0, 2, 1) +
          R::dnorm(exp(theta(2)), 0, 2, 1) + 
          arma::accu(theta); //jacobian term
        
        return log_pdf;
      }
        
        
      // [[Rcpp::export]]
      Rcpp::List create_xptrs() {
          
        // typedef for a pointer of nonlinear function returning vec (T, Z)
        typedef arma::vec (*nvec_fnPtr)(const unsigned int t, 
          const arma::vec& alpha, const arma::vec& theta, 
          const arma::vec& known_params, const arma::mat& known_tv_params);
        // for a pointer of nonlinear function returning mat (Tg, Zg, H, R)
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
    "nlg_linear_gaussian" = {
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
    "nlg_sin_exp" = {
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
