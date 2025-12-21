# ============================================================
# DiD + Event Study (stacked) for BIG market-share closures
# Works for:
#   (1) BCY: bank × county × year
#   (2) CY:  county × year
#
# Outputs:
#   - regression logs
#   - event study plots (png)
#   - CSV of coefficient paths
# ============================================================

library(data.table)
library(fixest)
library(ggplot2)
library(broom)

# -----------------------------
# Paths
# -----------------------------
setwd("/Users/himalbamzai-wokhlu/PycharmProjects/DistanceLending")

bcy_path <- "source/sba_analysis/data/derived/sba_bcy_market_share_flags.csv"
cy_path  <- "source/sba_analysis/data/derived/sba_cy_market_share_flags.csv"

out_dir  <- "output/transformations/event_study/big_share_closures"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_path <- file.path(out_dir, "did_event_study_big_share.log")
coef_csv <- file.path(out_dir, "event_study_coefficients.csv")

log_write <- function(txt) cat(txt, file = log_path, append = TRUE)

# -----------------------------
# User params
# -----------------------------
WINDOW_PRE  <- 10
WINDOW_POST <- 10

# High market-share threshold (set here)
MSHARE_THRESH <- 0.05  # e.g. 5%

# Which market share field to use (must exist in your panel)
# If you have both, use the one you trust.
# common in your older code: combined_market_share_of_closure
MSHARE_VAR <- "combined_market_share_of_closure"

# Treatments to test (must exist)
TREATMENTS <- c("any_closure", "any_exog_closure", "any_true_exog")

# Outcomes (edit as needed)
OUTCOMES_BCY <- c(
  "avg_distance_miles",
  "med_distance_miles",
  "default_rate",
  "log_lending",
  "loan_growth"
)

OUTCOMES_CY <- c(
  "avg_distance_miles",
  "med_distance_miles",
  "default_rate",
  "log_lending",
  "loan_growth"
)

# ============================================================
# Helpers
# ============================================================

# build cohort (first year treated) at unit level
make_cohort <- function(dt, unit_id, year = "year", treat = "treat") {
  dt <- copy(dt)
  dt[, cohort := ifelse(any(get(treat) == 1L),
                        min(get(year)[get(treat) == 1L]),
                        NA_integer_),
     by = ..unit_id]
  dt[, ever_treat  := as.integer(!is.na(cohort))]
  dt[, never_treat := 1L - ever_treat]
  dt
}

# stacked data builder (your earlier approach)
build_stacked <- function(dt, cohort_var = "cohort", never_treat_var = "never_treat",
                          unit_fe = "unit_id", time_var = "year") {
  
  dt <- copy(dt)
  dt[[time_var]] <- as.integer(dt[[time_var]])
  dt[[cohort_var]] <- as.integer(dt[[cohort_var]])
  
  treated_cohorts <- sort(unique(dt[get(never_treat_var) == 0 & !is.na(get(cohort_var)), get(cohort_var)]))
  if (length(treated_cohorts) == 0L) stop("No treated cohorts found.")
  
  stacks <- vector("list", length(treated_cohorts))
  for (i in seq_along(treated_cohorts)) {
    cval <- treated_cohorts[i]
    sub  <- dt[get(cohort_var) == cval | get(never_treat_var) == 1L]
    sub[, stack := cval]
    stacks[[i]] <- sub
  }
  
  out <- rbindlist(stacks, use.names = TRUE, fill = TRUE)
  out[, unit_stack := interaction(get(unit_fe), stack, drop = TRUE)]
  out[, time_stack := interaction(get(time_var), stack, drop = TRUE)]
  out
}

add_rel_time <- function(stacked_dt, time_var = "year", cohort_var = "cohort") {
  stacked_dt <- copy(stacked_dt)
  stacked_dt[, rel_time := get(time_var) - get(cohort_var)]
  stacked_dt
}

# clean event-study coefficient table from fixest
tidy_es <- function(est) {
  tt <- broom::tidy(est)
  tt <- tt[grepl("^rel_time::", tt$term), ]
  if (nrow(tt) == 0) return(tt)
  tt$rel_time <- as.integer(gsub("rel_time::", "", tt$term))
  tt
}

plot_es <- function(tt, title, save_path) {
  if (nrow(tt) == 0) return(invisible(NULL))
  
  p <- ggplot(tt, aes(x = rel_time, y = estimate)) +
    geom_hline(yintercept = 0, linetype = "dotted") +
    geom_vline(xintercept = -1, linetype = "dashed") +
    geom_point(size = 2) +
    geom_line() +
    geom_errorbar(aes(ymin = estimate - 1.96 * std.error,
                      ymax = estimate + 1.96 * std.error),
                  width = 0.2) +
    theme_minimal() +
    labs(
      title = title,
      x = "Event time (years relative to closure cohort)",
      y = "Effect (relative to t = -1)"
    )
  
  ggsave(save_path, p, width = 7, height = 5, dpi = 300)
  p
}

# ============================================================
# Variable construction (BCY / CY)
# ============================================================

prep_bcy <- function(dt) {
  dt <- as.data.table(copy(dt))
  if ("borrower_county" %in% names(dt)) setnames(dt, "borrower_county", "county")
  
  # Outcomes used in your older runs
  setorder(dt, loan_cert, county, year)
  dt[, loan_growth := (n_loans / shift(n_loans) - 1), by = .(loan_cert, county)]
  dt[is.infinite(loan_growth), loan_growth := NA_real_]
  
  dt[, log_lending := log1p(total_gross_approval)]
  
  # unit id for BCY
  dt[, unit_id := interaction(loan_cert, county, drop = TRUE)]
  dt
}

prep_cy <- function(dt) {
  dt <- as.data.table(copy(dt))
  if ("borrower_county" %in% names(dt)) setnames(dt, "borrower_county", "county")
  
  setorder(dt, county, year)
  dt[, loan_growth := (n_loans / shift(n_loans) - 1), by = county]
  dt[is.infinite(loan_growth), loan_growth := NA_real_]
  
  dt[, log_lending := log1p(total_gross_approval)]
  
  # unit id for CY
  dt[, unit_id := county]
  dt
}

# ============================================================
# “Big share closure” definition
# ============================================================

add_bigshare_treatments <- function(dt,
                                    mshare_var = MSHARE_VAR,
                                    thresh = MSHARE_THRESH,
                                    base_treats = TREATMENTS) {
  dt <- copy(dt)
  
  if (!mshare_var %in% names(dt)) {
    stop(paste0("Missing mshare variable: ", mshare_var))
  }
  
  dt[, mshare_loss := (get(mshare_var) >= thresh)]
  
  for (tr in base_treats) {
    if (!tr %in% names(dt)) next
    dt[, paste0("big_", tr) := as.integer(mshare_loss & (get(tr) == 1L))]
  }
  
  dt
}

# ============================================================
# DiD (plain TWFE) for “big” closures
#   outcome ~ treat + unit FE + year FE
# (This is not a dynamic ES; it’s your quick DiD check.)
# ============================================================

run_did_twfe <- function(dt, outcome, treat, fe, cluster) {
  
  fml <- as.formula(paste0(outcome, " ~ ", treat, " | ", fe))
  est <- feols(fml, data = dt, cluster = as.formula(paste0("~", cluster)))
  
  log_write("\n\n==============================\n")
  log_write(paste0("TWFE DiD\nOutcome: ", outcome, "\nTreat: ", treat, "\nTime: ", Sys.time(), "\n"))
  log_write(capture.output(summary(est)))
  log_write("\n")
  
  est
}

# ============================================================
# Event Study (stacked) for “big” closures
#   outcome ~ i(rel_time, ref=-1) + FE
# ============================================================

run_stacked_es <- function(stacked_dt, outcome,
                           ref_period = -1,
                           fe = "loan_cert + county + year",
                           cluster = "unit_stack") {
  
  # event-time dummies only (you can add controls if you want)
  fml <- as.formula(paste0(outcome, " ~ i(rel_time, ref = ", ref_period, ") | ", fe))
  
  est <- feols(fml, data = stacked_dt, cluster = as.formula(paste0("~", cluster)))
  
  log_write("\n\n==============================\n")
  log_write(paste0("Stacked Event Study\nOutcome: ", outcome, "\nTime: ", Sys.time(), "\n"))
  log_write(capture.output(summary(est)))
  log_write("\n")
  
  est
}

# ============================================================
# MAIN: BCY
# ============================================================

main_bcy <- function() {
  
  dt <- fread(bcy_path)
  dt <- prep_bcy(dt)
  dt <- add_bigshare_treatments(dt)
  
  # pick big treatments
  big_treats <- paste0("big_", TREATMENTS)
  big_treats <- big_treats[big_treats %in% names(dt)]
  if (length(big_treats) == 0L) stop("No big_* treatments found in BCY panel.")
  
  # TWFE DiD quick checks (bank×county unit FE + year FE)
  # (bank FE + county FE + year FE is also OK; but unit FE is tighter)
  # Here: unit_id FE + year FE
  dt[, unit_fe := unit_id]
  fe_twfe <- "unit_fe + year"
  
  for (tr in big_treats) {
    for (y in OUTCOMES_BCY) {
      if (!y %in% names(dt)) next
      dd <- dt[!is.na(get(y)) & !is.na(get(tr))]
      if (nrow(dd) == 0) next
      run_did_twfe(dd, outcome = y, treat = tr, fe = fe_twfe, cluster = "unit_fe")
    }
  }
  
  # Event study (stacked)
  for (tr in big_treats) {
    dtr <- copy(dt)
    dtr[, treat := as.integer(get(tr) == 1L)]
    dtr <- make_cohort(dtr, unit_id = "unit_id", year = "year", treat = "treat")
    
    stacked <- build_stacked(dtr, unit_fe = "unit_id", time_var = "year")
    stacked <- add_rel_time(stacked, time_var = "year", cohort_var = "cohort")
    stacked <- stacked[rel_time >= -WINDOW_PRE & rel_time <= WINDOW_POST]
    
    # FE for BCY ES
    fe_es <- "loan_cert + county + year"
    
    for (y in OUTCOMES_BCY) {
      if (!y %in% names(stacked)) next
      dd <- stacked[!is.na(get(y))]
      if (nrow(dd) == 0) next
      
      est <- run_stacked_es(dd, outcome = y, fe = fe_es, cluster = "unit_stack")
      tt  <- tidy_es(est)
      
      # save plot
      title <- paste0("Stacked Event Study (BCY): ", y, " | ", tr,
                      " | mshare≥", MSHARE_THRESH)
      savep <- file.path(out_dir, paste0("es_bcy_", y, "_", tr, ".png"))
      plot_es(tt, title, savep)
      
      # append coefficient table
      if (nrow(tt) > 0) {
        tt$outcome <- y
        tt$treatment <- tr
        tt$panel <- "BCY"
        fwrite(as.data.table(tt), coef_csv, append = file.exists(coef_csv))
      }
    }
  }
  
  message("BCY done. Outputs in: ", normalizePath(out_dir))
}

# ============================================================
# MAIN: CY
# ============================================================

main_cy <- function() {
  
  dt <- fread(cy_path)
  dt <- prep_cy(dt)
  dt <- add_bigshare_treatments(dt)
  
  big_treats <- paste0("big_", TREATMENTS)
  big_treats <- big_treats[big_treats %in% names(dt)]
  if (length(big_treats) == 0L) stop("No big_* treatments found in CY panel.")
  
  # TWFE DiD (county FE + year FE)
  fe_twfe <- "county + year"
  
  for (tr in big_treats) {
    for (y in OUTCOMES_CY) {
      if (!y %in% names(dt)) next
      dd <- dt[!is.na(get(y)) & !is.na(get(tr))]
      if (nrow(dd) == 0) next
      run_did_twfe(dd, outcome = y, treat = tr, fe = fe_twfe, cluster = "county")
    }
  }
  
  # Event study (stacked)
  for (tr in big_treats) {
    dtr <- copy(dt)
    dtr[, treat := as.integer(get(tr) == 1L)]
    dtr <- make_cohort(dtr, unit_id = "unit_id", year = "year", treat = "treat")
    
    stacked <- build_stacked(dtr, unit_fe = "unit_id", time_var = "year")
    stacked <- add_rel_time(stacked, time_var = "year", cohort_var = "cohort")
    stacked <- stacked[rel_time >= -WINDOW_PRE & rel_time <= WINDOW_POST]
    
    # FE for CY ES
    fe_es <- "county + year"
    
    for (y in OUTCOMES_CY) {
      if (!y %in% names(stacked)) next
      dd <- stacked[!is.na(get(y))]
      if (nrow(dd) == 0) next
      
      est <- run_stacked_es(dd, outcome = y, fe = fe_es, cluster = "unit_stack")
      tt  <- tidy_es(est)
      
      title <- paste0("Stacked Event Study (CY): ", y, " | ", tr,
                      " | mshare≥", MSHARE_THRESH)
      savep <- file.path(out_dir, paste0("es_cy_", y, "_", tr, ".png"))
      plot_es(tt, title, savep)
      
      if (nrow(tt) > 0) {
        tt$outcome <- y
        tt$treatment <- tr
        tt$panel <- "CY"
        fwrite(as.data.table(tt), coef_csv, append = file.exists(coef_csv))
      }
    }
  }
  
  message("CY done. Outputs in: ", normalizePath(out_dir))
}

# ============================================================
# RUN
# ============================================================

# Reset log at start
cat("", file = log_path)

main_bcy()
main_cy()
