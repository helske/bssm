model_type <- function(model) {
  if (inherits(model, "lineargaussian")) {
    switch(class(model)[1],
      "ssm_mlg" = 0L,
      "ssm_ulg" = 1L,
      "bsm_lg" = 2L,
      "ar1_lg" = 3L)
  } else {
    switch(class(model)[1],
      "ssm_mng" = 0L,
      "ssm_ung" = 1L,
      "bsm_ng" = 2L,
      "svm" = 3L,
      "ar1_ng" = 4L)
  }
}
