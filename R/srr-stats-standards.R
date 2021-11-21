#' srr_stats
#'
#' All of the following standards initially have `@srrstatsTODO` tags.
#' These may be moved at any time to any other locations in your code.
#' Once addressed, please modify the tag from `@srrstatsTODO` to `@srrstats`,
#' or `@srrstatsNA`, ensuring that references to every one of the following
#' standards remain somewhere within your code.
#' (These comments may be deleted at any time.)
#'
#' @srrstatsVerbose TRUE
#' @noRd
NULL
#' NA_standards
#'
#' Any non-applicable standards can have their tags changed from 
#' `@srrstatsTODO` to `@srrstatsNA`, and placed together in this block, 
#' along with explanations for why each of these standards have been deemed not 
#' applicable. (These comments may also be deleted at any time.)
#'
#' @srrstatsNA {G2.4d, G2.4e, G2.5} Factor types are not used nor supported.
#' @srrstatsNA {G2.10, G2.11, G2.12, G2.13} No data.frame style tabular data is 
#' used/supported as input
#' @srrstatsNA {G3.1, G3.1a} No sample covariance calculations done.
#' @srrstatsNA {G4.0} No output is written to local files.
#' @srrstatsNA {G5.3} Some functions can produce NAs and nonfinite values, 
#' and there are some checks for these (e.g. in C++ side) but not explicitly 
#' tested everywhere.
#' @srrstatsNA {G5.10, G5.11, G5.11a, G5.12} Package does not contain extended 
#' tests (although benchmarks folder contains template for running such very 
#' time-consuming tests), although some of the automatic tests are switched off 
#' for CRAN due to the time limits.
#' @srrstatsNA {BS2.10} Not applicable as only single-chain runs are supported 
#' (but several such runs can be combined with posterior package).
#' @srrstatsNA {BS2.11} Starting values are not accepted in this form.
#' @srrstatsNA {BS1.4, BS1.5, BS4.3, BS4.4, BS4.5, BS4.6, BS4.7, BS5.4} No 
#' support for automatic stopping at converge (converge checkers).
#' @srrstatsNA {BS2.15} Errors are normal R errors so they can be caught? But 
#' not sure what is meant here. 
#' @srrstatsNA {BS3.1, BS3.2} Not really relevant for SSMs, or at least 
#' difficult to check this kind of thing in general.
#' @srrstatsNA {BS6.1, BS6.2, BS6.3, BS6.5} Just suggests and illustrates using 
#' ggplot or bayesplot packages, with several examples. 
#' @noRd
NULL
