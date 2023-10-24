Sys.setenv("OMP_THREAD_LIMIT" = 2)

library("testthat")
test_check("bssm")
