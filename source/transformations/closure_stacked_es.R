library(data.table)
library(fixest)
library(ggplot2)
library(broom)   # <- add this

bcy_path  <- "source/transformations/data/sba_bcy.csv"
cy_path  <- "source/transformations/data/sba_cy.csv"

out_dir   <- "output/transformations/event_study"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
es_log_path <- file.path(out_dir, "event_study_results.log")


build_stacked_data <- function(dt, cohort_var = "cohort", never_treat_var = "never_treat", unit_fe = "unit_id", time_var = "year") {
  dt <- copy(dt)
  dt[[time_var]] <- as.integer(dt[[time_var]])
  dt[[cohort_var]] <- as.integer(dt[[cohort_var]])
  treated_cohorts <- sort(unique(dt[get(never_treat_var) == 0 & !is.na(get(cohort_var)), get(cohort_var)]))
  if (length(treated_cohorts) == 0L) {stop("No treated cohorts found: check cohort / never_treat variables.")}
  stacks <- vector("list", length(treated_cohorts))
  for (i in seq_along(treated_cohorts)) {
    cval <- treated_cohorts[i]
    sub <- dt[get(cohort_var) == cval | get(never_treat_var) == 1L]
    sub[, stack := cval]
    stacks[[i]] <- sub}
  
  stacked_dt <- rbindlist(stacks, use.names = TRUE, fill = TRUE)
  stacked_dt[, unit_stack := interaction(get(unit_fe), stack, drop = TRUE)]
  stacked_dt[, time_stack := interaction(get(time_var), stack, drop = TRUE)]
  print(head(stacked_dt))
  return (stacked_dt)
}

add_event_time <- function(stacked_dt, time_var = "year", cohort_var = "cohort") {
  # For treated units (ever_treat == 1), cohort is their first treatment year
  # For never-treated, cohort in each stack is the cohort of the treated group that stack is built around
  stacked_dt <- copy(stacked_dt)
  stacked_dt[, rel_time := get(time_var) - get(cohort_var)]
  return (stacked_dt)
}

run_stacked_es <- function(stacked_dt,
                           outcome     = "dist_mean_next5",
                           covariates  = NULL,
                           fe_other    = "county",
                           ref_period  = -1,
                           cluster_var = "unit_stack",
                           treatment   = NA) {
  
  dt <- data.table::copy(stacked_dt)
  
  if (!outcome %in% names(dt)) {
    stop(paste("Outcome", outcome, "not found in stacked_dt"))
  }
  
  dt <- dt[!is.na(get(outcome))]
  
  # RHS: event-time dummies + covariates
  rhs_terms <- c(paste0("i(rel_time, ref = ", ref_period, ")"), covariates)
  rhs_terms <- rhs_terms[!is.na(rhs_terms)]
  rhs <- paste(rhs_terms, collapse = " + ")
  
  # FE
  fml <- as.formula(
    paste0(outcome, " ~ ", rhs, " | loan_cert + county + year")
  )
  
  # Estimate
  est <- fixest::feols(
    fml,
    data    = dt,
    cluster = as.formula(paste0("~", cluster_var))
  )
  
  # ----------------------------
  # LOG FULL SUMMARY
  # ----------------------------
  header <- paste0(
    "\n\n====================================================\n",
    "Event Study Regression\n",
    "Treatment: ", treatment, "\n",
    "Outcome:   ", outcome, "\n",
    "Timestamp: ", Sys.time(), "\n",
    "====================================================\n"
  )
  
  es_log_write(header)
  es_log_write(capture.output(summary(est)))
  es_log_write("\n")
  
  return(est)
}

run_stacked_es_cy <- function(stacked_dt,
                              outcome     = "dist_mean_next5",
                              covariates  = NULL,
                              ref_period  = -1,
                              cluster_var = "unit_stack",
                              treatment   = NA) {
  
  dt <- data.table::copy(stacked_dt)
  
  if (!outcome %in% names(dt)) {
    stop(paste("Outcome", outcome, "not found in stacked_dt"))
  }
  
  dt <- dt[!is.na(get(outcome))]
  
  # RHS: event-time dummies + covariates
  rhs_terms <- c(paste0("i(rel_time, ref = ", ref_period, ")"), covariates)
  rhs_terms <- rhs_terms[!is.na(rhs_terms)]
  rhs <- paste(rhs_terms, collapse = " + ")
  
  # FE: county + year (no loan_cert in county-year panel)
  fml <- as.formula(
    paste0(outcome, " ~ ", rhs, " | county + year")
  )
  
  est <- fixest::feols(
    fml,
    data    = dt,
    cluster = as.formula(paste0("~", cluster_var))
  )
  
  # ---- log summary ----
  header <- paste0(
    "\n\n====================================================\n",
    "Event Study Regression (COUNTY-YEAR)\n",
    "Treatment: ", treatment, "\n",
    "Outcome:   ", outcome, "\n",
    "Timestamp: ", Sys.time(), "\n",
    "====================================================\n"
  )
  
  es_log_write(header)
  es_log_write(capture.output(summary(est)))
  es_log_write("\n")
  
  return(est)
}

plot_event_study <- function(est, title = "Event Study", save_path = NULL) {
  etable <- broom::tidy(est)
  etable <- etable[grep("^rel_time::", etable$term), ]
  
  if (nrow(etable) == 0) {
    message("No rel_time terms found for plotting.")
    return(invisible(NULL))
  }
  
  # Parse rel_time (use data.frame syntax, not data.table :=)
  etable$rel_time <- as.integer(gsub("rel_time::", "", etable$term))
  
  # Build ggplot
  p <- ggplot(etable, aes(x = rel_time, y = estimate)) +
    geom_point(size = 2) +
    geom_line() +
    geom_errorbar(aes(ymin = estimate - 1.96 * std.error,
                      ymax = estimate + 1.96 * std.error),
                  width = 0.2) +
    geom_vline(xintercept = -1, color = "red", linetype = "dashed") +
    theme_minimal() +
    labs(title = title,
         x = "Event Time (Years Relative to Closure)",
         y = "Coefficient")
  
  # Save plot if location provided
  if (!is.null(save_path)) {
    ggplot2::ggsave(filename = save_path, plot = p, width = 7, height = 5, dpi = 300)
    message(sprintf("Saved event study plot to: %s", normalizePath(save_path)))
  }
  
  return(p)
}


es_log_write <- function(text) {
  cat(text, file = es_log_path, append = TRUE)
}

main <- function() {
  sba_bcy <- data.table::fread(bcy_path)
  dt      <- as.data.table(sba_bcy)
  dt[, unit_id := interaction(loan_cert, county, drop = TRUE)]
  window_pre  <- 10
  window_post <- 10
  treatments <- c("any_closure", "any_exog_closure", "any_true_exog")
  outcomes   <- c("dist_median", "dist_median_next5", "dist_mean_next5", "dist_mean", "default_rate", "default_rate_next5", "loan_growth_next5")
  
  for (tr in treatments) {
    message("\n==============================")
    message("Treatment: ", tr)
    message("==============================")
    dtr <- copy(dt)
    dtr[, treat := as.integer(get(tr) == 1L)]
    dtr[, cohort := ifelse(any(treat == 1L),
                           min(year[treat == 1L]),
                           NA_integer_),
        by = unit_id]
    dtr[, ever_treat  := as.integer(!is.na(cohort))]
    dtr[, never_treat := 1L - ever_treat]
    
    # Build stacked dataset
    stacked_dt <- build_stacked_data(dtr)
    stacked_dt <- add_event_time(stacked_dt)
    stacked_dt <- stacked_dt[rel_time >= -window_pre & rel_time <= window_post]
    
    for (y in outcomes) {
      
      if (!y %in% names(stacked_dt)) {
        message("  Skipping outcome ", y, " (not found).")
        next
      }
      
      message("  Running ES for outcome: ", y)
      est <- run_stacked_es(stacked_dt = stacked_dt, outcome = y, treatment  = tr)
      
      # Save plot
      plot_title <- paste0("Stacked Event Study at Bank-County-Yr: ", y, " (", tr, ")")
      plot_path  <- file.path(out_dir, paste0("es_", y, "_", tr, ".png"))
      plot_event_study(est, title = plot_title, save_path = plot_path)
      
      message(sprintf("  ✔ Saved plot for outcome = %s, treatment = %s", y, tr))
      message(sprintf("    → %s", normalizePath(plot_path)))
    }
  }
}

main_cy <- function() {
  sba_cy <- data.table::fread(cy_path)
  dt     <- as.data.table(sba_cy)
  
  # Standardize county name
  if ("borrower_county" %in% names(dt) && !"county" %in% names(dt)) {
    data.table::setnames(dt, "borrower_county", "county")
  }
  dt[, unit_id := county]
  window_pre  <- 5
  window_post <- 10
  treatments <- c("any_closure", "any_exog_closure", "any_true_exog")
  outcomes   <- c("dist_median",
                  "dist_mean",
                  "dist_median_next5",
                  "n_branches_next5"
                  "default_rate_next5",
                  "loan_growth_next5")
  
  for (tr in treatments) {
    if (!tr %in% names(dt)) {
      message("Skipping treatment ", tr, " (not found in sba_cy).")
      next
    }
    
    message("\n==============================")
    message("COUNTY-YEAR Treatment: ", tr)
    message("==============================")
    
    dtr <- copy(dt)
    dtr[, treat := as.integer(get(tr) == 1L)]
    
    # Cohort: first treatment year per county
    dtr[, cohort := ifelse(any(treat == 1L),
                           min(year[treat == 1L]),
                           NA_integer_),
        by = unit_id]
    
    dtr[, ever_treat  := as.integer(!is.na(cohort))]
    dtr[, never_treat := 1L - ever_treat]
    
    # Build stacked dataset (same function as before)
    stacked_dt <- build_stacked_data(dtr)
    stacked_dt <- add_event_time(stacked_dt)
    stacked_dt <- stacked_dt[rel_time >= -window_pre & rel_time <= window_post]
    
    for (y in outcomes) {
      
      if (!y %in% names(stacked_dt)) {
        message("  Skipping outcome ", y, " (not found in stacked_dt).")
        next
      }
      
      message("  Running ES (county-year) for outcome: ", y)
      est <- run_stacked_es_cy(stacked_dt = stacked_dt, outcome = y, treatment  = tr)
      
      # Save plot
      plot_title <- paste0("Stacked ES (CY): ", y, " (", tr, ")")
      plot_path  <- file.path(out_dir, paste0("es_cy_", y, "_", tr, ".png"))
      
      plot_event_study(est, title = plot_title, save_path = plot_path)
      
      message(sprintf("  ✔ Saved county-year plot for outcome = %s, treatment = %s", y, tr))
      message(sprintf("    → %s", normalizePath(plot_path)))
    }
  }
}


if (sys.nframe() == 0) {
  # Choose one of these depending on what you want to run:
  # main()      # bank × county × year panel (sba_bcy)
  main_cy()     # county × year panel (sba_cy)
}
