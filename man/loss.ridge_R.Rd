% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/optim_1_source.R
\name{loss.ridge_R}
\alias{loss.ridge_R}
\title{Ridge regression Loss Function for Linear Models}
\usage{
loss.ridge_R(y, A, x, lambda)
}
\arguments{
\item{y}{A (n x 1) vector of response variables}

\item{A}{A (n x p) matrix of predictor variables}

\item{x}{A (p x 1) vector of effect size for each predictor variable}

\item{lambda}{Regularization parameter in ridge regression}
}
\value{
Ridge Regression errors with penalty
}
\description{
Ridge regression Loss Function for Linear Models
}
