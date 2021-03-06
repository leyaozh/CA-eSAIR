# This is the source code for R package CA-eSAIR
# Built on Jun 14, 2020, and last edited on Jun 14, 2020
# Correspondence : Peter X.K. Song, Ph.D. (pxsong@umich.edu)
# Creator: Leyao Zhang, M.S. (leyaozh@umich.edu)
# Model: An extend CA-eSIR model with time-varying immunization.
#'
#' Extended state-space CA-eSIR with a subset of the population showing antibody positivity.
#'
#' @param Y the time series of daily observed infected compartment proportions.
#' @param R the time series of daily observed removed compartment proportions, including death and recovered.
#' @param phi0 a vector of values of the dirac delta function \eqn{\phi_t}. Each entry denotes the proportion that will be qurantined at each change time point. Note that all the entries lie between 0 and 1, its default is \code{NULL}.
#' @param change_time the change points over time corresponding to \code{phi0}, to formulate the dirac delta function \eqn{\phi_t}; its defalt value is \code{NULL}.
#' @param begin_str the character of starting time, the default is "01/13/2020".
#' @param T_fin the end of follow-up time after the beginning date \code{begin_str}, the default is 200.
#' @param nchain the number of MCMC chains generated by \code{\link[rjags]{rjags}}, the default is 4.
#' @param nadapt the iteration number of adaptation in the MCMC. We recommend using at least the default value 1e4 to obtained fully adapted chains.
#' @param M the number of draws in each chain, with no thinning. The default is M=5e2 but suggest using 5e5.
#' @param thn the thinning interval between mixing. The total number of draws thus would become \code{round(M/thn)*nchain}. The default is 10.
#' @param nburnin the burn-in period. The default is 2e2 but suggest 2e5.
#' @param dic logical, whether compute the DIC (deviance information criterion) for model selection.
#' @param death_in_R the numeric value of average of cumulative deaths in the removed compartments. The default is 0.4 within Hubei and 0.02 outside Hubei.
#' @param casename the string of the job's name. The default is "qh.eSIR".
#' @param beta0 the hyperparameter of average transmission rate, the default is the one estimated from the SARS first-month outbreak (0.2586).
#' @param gamma0 the hyperparameter of average removed rate, the default is the one estimated from the SARS first-month outbreak (0.0821).
#' @param R0 the hyperparameter of the mean reproduction number R0. The default is thus the ratio of \code{beta0/gamma0}, which can be specified directly.
#' @param gamma0_sd the standard deviation for the prior distrbution of the removed rate \eqn{\gamma}, the default is 0.1.
#' @param R0_sd the standard deviation for the prior disbution of R0, the default is 1.
#' @param file_add the string to denote the location of saving output files and tables.
#'
#' @param save_mcmc logical, whether save (\code{TRUE}) all the MCMC outputs or not (\code{FALSE}).The output file will be an \code{.RData} file named by the \eqn{casename}. We include arrays of prevalence values of the three compartments with their matrices of posterior draws up to the last date of the collected data as \code{theta_p[,,1]} and afterwards as \code{theta_pp[,,1]} for \eqn{\theta_t^S}, \code{theta_p[,,2]} and \code{theta_pp[,,2]} for \eqn{\theta_t^I}, and \code{theta_p[,,3]} and \code{theta_pp[,,3]} for \eqn{\theta_t^R}. The posterior draws of the prevalence process of the quarantine compartment can be obtained via \code{thetaQ_p} and \code{thetaQ_pp}. Moreover, the input and predicted proportions \code{Y}, \code{Y_pp}, \code{R} and \code{R_pp} can also be retrieved. The prevalence and prediceted proportion matrices have rows for MCMC replicates, and columns for days. The MCMC posterior draws of other parameters including \code{beta}, \code{gamma}, \code{R0}, and variance controllers \code{k_p}, \code{lambdaY_p}, \code{lambdaR_p} are also available.
#' @param save_plot_data logical, whether save the plotting data or not.
#' @param save_files logical, whether to save plots to file.
#' @param add_death logical, whether add the approximate death curve to the plot, default is false.
#' @param eps a non-zero controller so that all the input \code{Y} and \code{R} values would be bounded above 0 (at least \code{eps}). Its default value is 1e-10
#'
#' @return
#' \item{casename}{the predefined \code{casename}.}
#' \item{incidence_mean}{mean cumulative incidence, the mean prevalence of cumulative confirmed cases at the end of the study.}
#' \item{incidence_ci}{2.5\%, 50\%, and 97.5\% quantiles of the incidences.}
#' \item{out_table}{summary tables including the posterior mean of the prevalance processes of the 3 states compartments (\eqn{\theta_t^S,\theta_t^I,\theta_t^R,\theta_t^H}) at last date of data collected ((\eqn{t^\prime}) decided by the lengths of your input data \code{Y} and \code{R}), and their respective credible inctervals (ci); the respective means and ci's of the reporduction number (R0), removed rate (\eqn{\gamma}), transmission rate  (\eqn{\beta}).}
#' \item{plot_infection}{plot of summarizing and forecasting for the infection compartment, in which the vertial blue line denotes the last date of data collected (\eqn{t^\prime}), the vertial darkgray line denotes the deacceleration point (first turning point) that the posterior mean first-derivative of infection prevalence \eqn{\dot{\theta}_t^I} achieves the  maximum, the vertical purple line denotes the second turning point that the posterior mean first-derivative infection proportion \eqn{\dot{\theta}_t^I} equals zero, the darkgray line denotes the posterior mean of the infection prevalence \eqn{\theta_t^I} and the red line denotes its posterior median. }
#' \item{plot_removed}{plot of summarizing and forecasting for the removed compartment with lines similar to those in the \code{plot_infection}. The vertical lines are identical, but the horizontal mean and median correspond to the posterior mean and median of the removed process \eqn{\theta_t^R}. An additional line indicates the estimated death prevalence from the input \code{death_in_R}.}
#' \item{spaghetti_plot}{20 randomly selected MCMC draws of the first-order derivative of the posterior prevalence of infection, namely \eqn{\dot{\theta}_t^I}. The black curve is the posterior mean of the derivative, and the vertical lines mark times of turning points corresponding respectively to those shown in \code{plot_infection} and \code{plot_removed}. Moreover, the 95\% credible intervals of these turning points are also highlighted by semi-transparent rectangles. }
#' \item{first_tp_mean}{the date t at which \eqn{\ddot{\theta}_t^I=0}, calculated as the average of the time points with maximum posterior first-order derivatives \eqn{\dot{\theta}_t^I}; this value may be slightly different from the one labeled by the "darkgreen" lines in the two plots \code{plot_infection} and \code{plot_removed}, which indicate the stationary point such that the first-order derivative of the averaged posterior of \eqn{\theta_t^I} reaches its maximum.}
#' \item{first_tp_mean}{the date t at which \eqn{\ddot{\theta}_t^I=0}, calculated as the average of the time points with maximum posterior first-order derivatives \eqn{\dot{\theta}_t^I}; this value may be slightly different from the one labeled by the "darkgreen" lines in the two plots \code{plot_infection} and \code{plot_removed}, which indicate the stationary point such that the first-order derivative of the averaged posterior of \eqn{\theta_t^I} reaches its maximum.}
#'
#'\item{first_tp_ci}{fwith \code{first_tp_mean}, it reports the corresponding credible interval and median.}
#' \item{second_tp_mean}{the date t at which \eqn{\theta_t^I=0}, calculated as the average of the stationary points of all of posterior first-order derivatives \eqn{\dot{\theta}_t^I}; this value may be slightly different from the one labeled by the "pruple" lines in the plots of \code{plot_infection} and \code{plot_removed}. The latter indicate stationary t at which the first-order derivative of the averaged posterior of \eqn{\theta_t^I} equals zero.}
#' \item{second_tp_ci}{with \code{second_tp_mean}, it reports the corresponding credible interval and median.}
#' \item{dic_val}{the output of \code{dic.samples()} in \code{\link[rjags]{dic.samples}}, computing deviance information criterion for model comparison.}
#' \item{gelman_diag_list}{ Since version 0.3.3, we incorporated Gelman And Rubin's Convergence Diagnostic using \code{\link[coda]{gelman.diag}}. We included both the statistics and their upper C.I. limits. Values substantially above 1 indicate lack of convergence. Error messages would be printed as they are. This would be only valid for multiple chains (e.g. nchain > 1). Note that for time dependent processes, we only compute the convergence of the last observation data (\code{T_prime}), though it shows to be \code{T_prime+1}, which is due to the day 0 for initialization.}
#'
#' @examples
#' \dontrun{
#' NI_complete <- c( 41,41,41,45,62,131,200,270,375,444,549, 729,
#'              1052,1423,2714,3554,4903,5806,7153,9074,11177,
#'              13522,16678,19665,22112,24953,27100,29631,31728,33366)
#' RI_complete <- c(1,1,7,10,14,20,25,31,34,45,55,71,94,121,152,213,
#'              252,345,417,561,650,811,1017,1261,1485,1917,2260,
#'              2725,3284,3754)
#' N=58.5e6
#' R <- RI_complete/N
#' Y <- NI_complete/N- R #Jan13->Feb 11
#'
#' change_time <- c("01/23/2020","02/04/2020","02/08/2020")
#' phi0 <- c(0.1,0.4,0.4)
#' res.q <- qh.eSIR (Y,R,begin_str="01/13/2020",death_in_R = 0.4,
#'                  phi0=phi0,change_time=change_time,
#'                 casename="Hubei_q",save_files = T,save_mcmc = F,
#'                  M=5e2,nburnin = 2e2)
#' res.q$plot_infection
#' #res.q$plot_removed
#'
#' res.noq <- qh.eSIR (Y,R,begin_str="01/13/2020",death_in_R = 0.4,
#'                     T_fin=200,casename="Hubei_noq",
#'                     M=5e2,nburnin = 2e2)
#' res.noq$plot_infection
#' }
#' @export
#'

### Weighted sum of inter-county connectivity effect for each county.
weightSum <- function(beta, eta, dMat, iMat, pMat0, pMat, Ip, pred, thres) {

  ### inter-county connectivity coefficient
  omega = rep(0, length(Ip))
  omega = iMat * dMat ^ eta

  wetVec = rep(0, length(Ip))
  if (pred == "one"){
    wet = pMat0 %*% Inum
    ##assume a threshold of probability for infected individuals meeting individuals from another county
    wet[which(wet > thres)] = thres
    wetVec = colSums(beta * wet * Ip %*% t(omega))
  }else if(pred == "t"){
    wet = pMat[i, ]
    wet[which(wet > thres)] = thres
    wetVec = colSums(beta * wet * Ip %*% t(omega))
  }
}

### one-step CA-eSAIR
CA-eSAIR = function(beta, gamma, alpha, pi, eta, dMat, iMat, pMat0, pMat, Sp, Ap, Ip, Rp, Inum, pred, thres){

}
