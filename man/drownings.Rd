% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/bssm-package.R
\docType{data}
\name{drownings}
\alias{drownings}
\title{Deaths by drowning in Finland in 1969-2019}
\format{
A time series object containing 51 observations.
}
\source{
Statistics Finland
\url{https://stat.fi/tup/tilastotietokannat/index_en.html}.
}
\description{
Dataset containing number of deaths by drowning in Finland in 1969-2019,
corresponding population sizes (in hundreds of thousands), and
yearly average summer temperatures (June to August), based on simple
unweighted average of three weather stations: Helsinki (Southern Finland),
Jyvaskyla (Central Finland), and Sodankyla (Northern Finland).
}
\examples{
data("drownings")
model <- bsm_ng(drownings[, "deaths"], u = drownings[, "population"],
  xreg = drownings[, "summer_temp"], distribution = "poisson",
  beta = normal(0, 0, 1),
  sd_level = gamma_prior(0.1,2, 10), sd_slope = gamma_prior(0, 2, 10))

fit <- run_mcmc(model, iter = 5000,
  output_type = "summary", mcmc_type = "approx")
fit
ts.plot(model$y/model$u, exp(fit$alphahat[, 1]), col = 1:2)
}
\keyword{datasets}
