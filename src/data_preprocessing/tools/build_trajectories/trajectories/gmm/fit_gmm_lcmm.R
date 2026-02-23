#!/usr/bin/env Rscript

# -----------------------------------------------------------------------------
# GMM — R implementation via `lcmm::hlme`.
#
# Purpose:
#   Fit a growth mixture model from long-format data and export standardised
#   artefacts consumed by the Python `GMM` builder:
#     - assignments.csv
#     - posterior_probabilities.csv
#     - class_parameters.csv
#     - centroids.csv
#     - diagnostics.csv
#
# Expected arguments:
#   --input --output-dir --id-col --time-col --value-col
#   --n-trajectories --max-iter --n-init
#   --random-effects --idiag --nwg [--seed]
#
# Random-effect controls:
#   --random-effects: `intercept` or `intercept_slope`
#   --idiag: constrain random-effect covariance to diagonal (`true`/`false`)
#   --nwg: class-specific random-effect scale (`true`/`false`)
# -----------------------------------------------------------------------------

# Abort the script with a readable error message.
stop_with <- function(message) {
  cat(paste0(message, "\n"), file = stderr())
  quit(save = "no", status = 1)
}

# Parse CLI args passed as key-value pairs: --key value.
parse_args <- function(args) {
  if (length(args) == 0) {
    stop_with("No arguments passed to GMM bridge.")
  }
  if ((length(args) %% 2) != 0) {
    stop_with("Arguments must be passed as key-value pairs.")
  }
  options <- list()
  idx <- 1L
  while (idx <= length(args)) {
    key <- args[[idx]]
    value <- args[[idx + 1L]]
    if (!startsWith(key, "--")) {
      stop_with(paste0("Unexpected argument key: ", key))
    }
    clean_key <- sub("^--", "", key)
    options[[clean_key]] <- value
    idx <- idx + 2L
  }
  options
}

# Retrieve a required argument from parsed options.
get_required <- function(options, key) {
  value <- options[[key]]
  if (is.null(value) || !nzchar(value)) {
    stop_with(paste0("Missing required argument: --", key))
  }
  value
}

# Parse an integer argument and fail fast on invalid input.
to_int <- function(raw, key) {
  value <- suppressWarnings(as.integer(raw))
  if (is.na(value)) {
    stop_with(paste0("Expected integer for --", key, ", got: ", raw))
  }
  value
}

# Parse a boolean argument and fail fast on invalid input.
to_bool <- function(raw, key) {
  normalised <- tolower(trimws(raw))
  if (normalised %in% c("true", "t", "yes", "y", "1")) {
    return(TRUE)
  }
  if (normalised %in% c("false", "f", "no", "n", "0")) {
    return(FALSE)
  }
  stop_with(paste0("Expected boolean for --", key, ", got: ", raw))
}

# Main bridge workflow:
#   1) Validate inputs and dependencies.
#   2) Fit GMM model with `hlme` (and `gridsearch` for random starts).
#   3) Export subject/class outputs for Python.
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  options <- parse_args(args)

  input_path <- get_required(options, "input")
  output_dir <- get_required(options, "output-dir")
  id_col <- get_required(options, "id-col")
  time_col <- get_required(options, "time-col")
  value_col <- get_required(options, "value-col")
  n_trajectories <- to_int(get_required(options, "n-trajectories"), "n-trajectories")
  max_iter <- to_int(get_required(options, "max-iter"), "max-iter")
  n_init <- to_int(get_required(options, "n-init"), "n-init")
  random_effects <- get_required(options, "random-effects")
  idiag <- to_bool(get_required(options, "idiag"), "idiag")
  nwg <- to_bool(get_required(options, "nwg"), "nwg")
  seed_raw <- options[["seed"]]

  if (n_trajectories <= 0L) {
    stop_with("--n-trajectories must be >= 1.")
  }
  if (max_iter <= 0L) {
    stop_with("--max-iter must be >= 1.")
  }
  if (n_init <= 0L) {
    stop_with("--n-init must be >= 1.")
  }
  if (!(random_effects %in% c("intercept", "intercept_slope"))) {
    stop_with("--random-effects must be one of: intercept, intercept_slope.")
  }

  if (!requireNamespace("lcmm", quietly = TRUE)) {
    stop_with("R package 'lcmm' is not installed.")
  }
  suppressPackageStartupMessages(library(lcmm))

  if (!file.exists(input_path)) {
    stop_with(paste0("Input CSV does not exist: ", input_path))
  }
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  if (!is.null(seed_raw) && nzchar(seed_raw)) {
    seed <- to_int(seed_raw, "seed")
    set.seed(seed)
  }

  data <- utils::read.csv(
    input_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  required_cols <- c(id_col, time_col, value_col)
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0L) {
    stop_with(
      paste0(
        "Input data is missing required columns: ",
        paste(missing_cols, collapse = ", ")
      )
    )
  }

  model_data <- data[, required_cols, drop = FALSE]
  if (anyNA(model_data)) {
    stop_with("Input data contains missing values in id/time/value columns.")
  }

  model_data[[time_col]] <- suppressWarnings(as.numeric(model_data[[time_col]]))
  model_data[[value_col]] <- suppressWarnings(as.numeric(model_data[[value_col]]))
  if (anyNA(model_data[[time_col]])) {
    stop_with("Time column could not be converted to numeric.")
  }
  if (anyNA(model_data[[value_col]])) {
    stop_with("Value column could not be converted to numeric.")
  }
  if (length(unique(model_data[[time_col]])) < 2L) {
    stop_with("At least two distinct time points are required.")
  }

  duplicate_keys <- duplicated(model_data[, c(id_col, time_col), drop = FALSE])
  if (any(duplicate_keys)) {
    model_data <- stats::aggregate(
      model_data[[value_col]],
      by = model_data[, c(id_col, time_col), drop = FALSE],
      FUN = mean
    )
    names(model_data)[names(model_data) == "x"] <- value_col
  }

  model_data <- model_data[order(model_data[[id_col]], model_data[[time_col]]), ]

  fixed_formula <- stats::as.formula(paste(value_col, "~", time_col))
  mixture_formula <- stats::as.formula(paste("~", time_col))
  random_formula <- if (random_effects == "intercept") {
    stats::as.formula("~ 1")
  } else {
    stats::as.formula(paste("~", time_col))
  }

  base_fit <- hlme(
    fixed = fixed_formula,
    random = random_formula,
    subject = id_col,
    ng = 1L,
    data = model_data,
    idiag = idiag,
    maxiter = max_iter,
    verbose = FALSE
  )

  if (n_trajectories == 1L) {
    fit <- base_fit
  } else {
    if (n_init > 1L) {
      fit <- gridsearch(
        rep = n_init,
        maxiter = min(max_iter, 50L),
        minit = base_fit,
        hlme(
          fixed = fixed_formula,
          mixture = mixture_formula,
          random = random_formula,
          subject = id_col,
          ng = n_trajectories,
          data = model_data,
          idiag = idiag,
          nwg = nwg,
          maxiter = max_iter,
          verbose = FALSE
        )
      )
    } else {
      fit <- hlme(
        fixed = fixed_formula,
        mixture = mixture_formula,
        random = random_formula,
        subject = id_col,
        ng = n_trajectories,
        data = model_data,
        idiag = idiag,
        nwg = nwg,
        maxiter = max_iter,
        verbose = FALSE,
        B = base_fit
      )
    }
  }

  if (!(fit$conv %in% c(1, 3))) {
    stop_with(
      paste0(
        "lcmm::hlme did not converge to an acceptable solution (conv=",
        fit$conv,
        ")."
      )
    )
  }

  if (n_trajectories == 1L) {
    subject_values <- unique(model_data[[id_col]])
    assignments <- data.frame(
      subject_id = subject_values,
      trajectory_id = 0L,
      check.names = FALSE
    )
    posterior <- data.frame(
      subject_id = subject_values,
      class_0 = 1.0,
      check.names = FALSE
    )
  } else {
    pprob <- as.data.frame(fit$pprob)
    if (ncol(pprob) < (2L + n_trajectories)) {
      stop_with("Unexpected lcmm posterior-probability output format.")
    }
    subject_values <- pprob[[1L]]
    assigned_class <- as.integer(pprob[[2L]]) - 1L
    prob_col_idx <- 3L:(2L + n_trajectories)
    probs <- pprob[, prob_col_idx, drop = FALSE]
    names(probs) <- paste0("class_", seq_len(n_trajectories) - 1L)
    assignments <- data.frame(
      subject_id = subject_values,
      trajectory_id = assigned_class,
      check.names = FALSE
    )
    posterior <- data.frame(
      subject_id = subject_values,
      probs,
      check.names = FALSE
    )
  }

  class_prob_names <- names(posterior)[names(posterior) != "subject_id"]
  class_prob_by_id <- posterior
  names(class_prob_by_id)[1L] <- id_col
  merged <- merge(
    x = model_data,
    y = class_prob_by_id,
    by = id_col,
    all.x = TRUE,
    sort = FALSE
  )

  time_grid <- sort(unique(model_data[[time_col]]))
  class_params <- vector("list", n_trajectories)
  centroid_rows <- vector("list", n_trajectories * length(time_grid))
  centroid_index <- 1L

  for (idx in seq_len(n_trajectories)) {
    class_name <- class_prob_names[[idx]]
    weights <- merged[[class_name]]
    weights[is.na(weights)] <- 0

    if (sum(weights) <= .Machine$double.eps) {
      intercept <- 0
      slope <- 0
    } else {
      weighted_fit <- tryCatch(
        stats::lm(
          formula = fixed_formula,
          data = merged,
          weights = weights
        ),
        error = function(e) NULL
      )
      if (is.null(weighted_fit)) {
        intercept <- stats::weighted.mean(merged[[value_col]], weights)
        slope <- 0
      } else {
        coefficients <- stats::coef(weighted_fit)
        intercept <- if ("(Intercept)" %in% names(coefficients)) {
          unname(coefficients["(Intercept)"])
        } else {
          0
        }
        slope <- if (time_col %in% names(coefficients)) {
          unname(coefficients[time_col])
        } else {
          0
        }
      }
    }

    if (!is.finite(intercept)) {
      intercept <- 0
    }
    if (!is.finite(slope)) {
      slope <- 0
    }

    class_params[[idx]] <- data.frame(
      trajectory_id = idx - 1L,
      intercept_mean = intercept,
      slope_mean = slope
    )

    for (time_value in time_grid) {
      centroid_rows[[centroid_index]] <- data.frame(
        trajectory_id = idx - 1L,
        time = time_value,
        value = intercept + slope * time_value
      )
      centroid_index <- centroid_index + 1L
    }
  }

  class_parameters <- do.call(rbind, class_params)
  centroids <- do.call(rbind, centroid_rows)

  diagnostics <- data.frame(
    n_trajectories = n_trajectories,
    random_effects = random_effects,
    idiag = idiag,
    nwg = if (n_trajectories > 1L) nwg else FALSE,
    loglik = fit$loglik,
    AIC = fit$AIC,
    BIC = fit$BIC,
    conv = fit$conv
  )

  utils::write.csv(
    assignments,
    file = file.path(output_dir, "assignments.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    posterior,
    file = file.path(output_dir, "posterior_probabilities.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    class_parameters,
    file = file.path(output_dir, "class_parameters.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    centroids,
    file = file.path(output_dir, "centroids.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    diagnostics,
    file = file.path(output_dir, "diagnostics.csv"),
    row.names = FALSE
  )
}

main()
