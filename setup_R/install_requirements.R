#!/usr/bin/env Rscript

source("setup_R/requirements.R")

repos <- "https://cloud.r-project.org"
installed <- rownames(installed.packages())
missing <- setdiff(required_packages, installed)

if (length(missing) == 0) {
  message("All required R packages are already installed.")
  quit(save = "no", status = 0)
}

message("Installing missing R packages: ", paste(missing, collapse = ", "))
install.packages(missing, repos = repos)
