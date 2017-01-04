// #ifndef BSM_H
// #define BSM_H
// 
// #include "gssm.h"
// 
// class bsm: public gssm {
//   
// public:
//   
//   bsm(const Rcpp::List&, unsigned int, bool);
//   
//   // log[q(y,x)/q(x,y)]
//   double proposal(const arma::vec&, const arma::vec&);
//   
//   // update model given the parameters theta
//   void update_model(arma::vec);
//   // extract theta from the model
//   arma::vec get_theta();
//   
// private:
//   const bool slope;
//   const bool seasonal;
//   const arma::uvec fixed;
//   const bool y_est;
//   const bool level_est;
//   const bool slope_est;
//   const bool seasonal_est;
//   const bool log_space;
//   
// };
// 
// #endif
