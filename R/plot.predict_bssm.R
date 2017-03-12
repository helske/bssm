#' Plot predictions based on bssm package
#'
#' @importFrom ggplot2 autoplot ggplot geom_line geom_ribbon
#' @method autoplot predict_bssm
#' @param object Object of class \code{predict_bssm}.
#' @param plot_mean Draw mean predictions. Default is \code{TRUE}.
#' @param plot_median Draw median predictions. Default is \code{TRUE}.
#' @param y Optional values for observations. Defaults to \code{object$y}.
#' @param fit Optional values for fitted values such as smoothed estimates of past observations.
#' @param obs_color,mean_color,median_color,fit_color,interval_color Colors for corssponding components of the plot.
#' @param alpha_fill Alpha value for controlling the transparency of the intervals.
#' @param ... Ignored.
#' @export
autoplot.predict_bssm <- function(object, plot_mean = TRUE,
  plot_median = TRUE, y, fit,
  obs_color = "black", mean_color = "red", median_color = "blue",
  fit_color = mean_color, interval_color = "#000000", alpha_fill = 0.25,
  ...) {
  
  if (!missing(y) && !is.ts(y)) {
    y <- ts(y, end = end(object$mean), frequency = frequency(object$mean))
  }
  
  plot_y <- !missing(y)
  if (!missing(fit) && !is.ts(fit)) {
    fit <- ts(fit, end = end(object$mean), frequency = frequency(object$mean))
  }
  plot_fit <- !missing(fit)
  d <- ts.union(if (plot_y) y,  if (plot_fit) fit, if (plot_mean) object$mean, 
    object$intervals)
  #d[nrow(d) - length(object$mean), 2:(ncol(d) - plot_fit)] <-  d[nrow(d) - length(object$mean), 1]
  times <- seq(from = tsp(d)[1], to = tsp(d)[2], by = 1/tsp(d)[3])
  d <- data.frame("time" = times, d)
  names(d) <-  c("time", if (plot_y) "y", if (plot_fit) "fit", 
    if (plot_mean) "mean", paste0("a",sub("%","", colnames(object$intervals))))
  
  n_mean <- length(object$mean)
  p <- ggplot(tail(d, n_mean), aes(time, mean)) +
    # geom_line(colour = mean_color) +
    scale_x_continuous(limits = range(times)) +
    scale_y_continuous()
    #scale_y_continuous(limits = range(d[,-1], na.rm = TRUE))
  intv <- names(d)[-(1:(1 + plot_y + plot_fit + plot_mean))]
  
  n_intvs <- floor(length(intv)/2)
  
  for (i in 1:n_intvs) {
    p <- p + geom_ribbon(aes_string(x = "time", ymin = intv[i], ymax = rev(intv)[i]),
      data = tail(d, n_mean), inherit.aes = FALSE,
      fill = interval_color, alpha = alpha_fill)
  }
  if (plot_mean) {
    p <- p + geom_line(colour = mean_color)
  }
  if (plot_y) {
    p <- p + geom_line(aes(time, y), colour = obs_color,
      data = head(d, length(y)), inherit.aes = FALSE)
  }
  if (plot_median) {
    p <- p + geom_line(aes_string(x = "time", "a50"), colour = median_color,
      data = tail(d, n_mean), inherit.aes = FALSE)
  }
  if (plot_fit) {
    p <- p + geom_line(aes(time, fit), colour = fit_color,
      head(d, length(fit)), inherit.aes = FALSE)
  }
  p
}
