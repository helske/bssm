// #include "prior.h"
// 
// prior::prior(List x) {
// 
//   if (!x.inherits("bssm_prior")) {
//     stop("Priors must be defined as bssm_prior object.");
//   }
//   prior_type = as<std::string>(x["prior_type"]);
// 
// }
// 
// uniform::uniform(List x) : prior(x) {
// 
//   min = x["min"];
//   max = x["max"];
// 
// }
// 
// halfnormal::halfnormal(List x) : prior(x) {
// 
//   sd = x["sd"];
// 
// }
// 
// normal::normal(List x) : prior(x) {
// 
//   mean = x["mean"];
//   sd = x["sd"];
// 
// }
// 
// double uniform::pdf(double x, int log_scale) {
//   return R::dunif(x, min, max, log_scale);
// }
// 
// 
// double halfnormal::pdf(double x, int log_scale) {
//   return 2.0 + R::dnorm(x, 0.0, sd, log_scale);
// }
// 
// double normal::pdf(double x, int log_scale) {
//   return R::dnorm(x, mean, sd, log_scale);
// }