Sys.setenv("OMP_NUM_THREADS" = 2)

library("testthat")
test_check("bssm")
