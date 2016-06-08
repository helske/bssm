#' Plot predictions based on bssm package
#'
#' @importFrom ggplot2 autoplot ggplot geom_line geom_ribbon
#' @method autoplot predict_bssm
#' @param object Object of class \code{predict_bssm}.
#' @param ... Ignored.
#' @export
autoplot.predict_bssm <- function(object, plot_mean = TRUE,
  plot_median = TRUE, y, fit,
  obs_colour = "black", mean_colour = "red", median_colour = "blue",
  fit_colour = "red", interval_colour = "#000000", alpha_fill = 0.25,
  y_label = "observations", title = NULL, ...) {

  if (!missing(y)) {
    if (!is.ts(y)) {
      y <- ts(y, start = start(object$y), frequency = frequency(object$y))
    }
    object$y <- y
  }
  if (!missing(fit) && !is.ts(fit)) {
    fit <- ts(fit, start = start(object$y), frequency = frequency(object$y))
  }
  plot_fit <- !missing(fit)
  d <- ts.union(object$y, if (plot_mean) object$mean, object$interval, if (plot_fit) fit)
  d[nrow(d) - length(object$mean), 2:(ncol(d) - plot_fit)] <-  d[nrow(d) - length(object$mean), 1]
  times <- seq(from = tsp(d)[1], to = tsp(d)[2], by = 1/tsp(d)[3])
  d <- data.frame("time" = times, d)
  names(d) <-  c("time", "y", if (plot_mean) "mean",
    paste0("a",sub("%","", colnames(object$intervals))), if (plot_fit) "fit")

  # if (plot_fit) {
  #   d[nrow(d) - length(object$mean), "fit"] <-  d[nrow(d) - length(object$mean), "fit"]
  # } else {
  #   d[nrow(d) - length(object$mean), -(1:2)] <-  d[nrow(d) - length(object$mean), "y"]
  # }

  p <- ggplot(d[1:length(object$y), ], aes(time, y)) +
    geom_line(colour = obs_colour) +
    scale_x_continuous(limits = range(times)) +
    scale_y_continuous(limits = range(d[,-1], na.rm = TRUE))
  intv <- names(d)[-(1:(2 + plot_mean))]
  n_intvs <- floor(length(intv)/2)
  for (i in 1:n_intvs) {
    p <- p + geom_ribbon(aes_string(x = "time", ymin = intv[i], ymax = rev(intv)[i]),
      data = d[(nrow(d) - length(object$mean)):nrow(d), ], inherit.aes = FALSE,
      fill = interval_colour, alpha = alpha_fill)
  }
  if (plot_mean) {
    p <- p + geom_line(aes(time, mean), colour = mean_colour,
      d[(nrow(d) - length(object$mean)):nrow(d), ], inherit.aes = FALSE)
  }
  if (plot_median) {
    p <- p + geom_line(aes_string(x = "time", "a50"), colour = median_colour,
      d[(nrow(d) - length(object$mean)):nrow(d), ], inherit.aes = FALSE)
  }
  if (plot_fit) {
    p <- p + geom_line(aes(time, fit), colour = fit_colour,
      d[1:length(fit), ], inherit.aes = FALSE)
  }
  p + ylab(y_label) + labs(title = title)
}
