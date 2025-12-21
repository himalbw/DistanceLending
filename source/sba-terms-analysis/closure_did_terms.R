suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(fixest)
})

# -------------------------
# Paths
# -------------------------
IN_BCY <- "source/sba_analysis/data/derived/sba_bcy_terms.csv"
IN_CY  <- "source/sba_analysis/data/derived/sba_cy_terms.csv"

OUTDIR <- "output/descriptives/did_terms"
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

LOGFILE <- file.path(OUTDIR, "did_terms.log")
OUTCSV  <- file.path(OUTDIR, "did_terms_results_k5.csv")

# -------------------------
# Logging helper
# -------------------------
log_start <- function() {
  sink(LOGFILE, split = TRUE)
  cat("============================================\n")
  cat("DID Terms Run\n")
  cat("Timestamp:", as.character(Sys.time()), "\n")
  cat("============================================\n\n")
}
log_stop <- function() sink()

# -------------------------
# Window maker: prevK/nextK means within groups
# -------------------------
make_windows <- function(dt, group_cols, time_col, vars, k = 5) {
  setDT(dt)
  setorderv(dt, c(group_cols, time_col))
  
  for (v in vars) {
    if (!v %in% names(dt)) next
    
    lag_names  <- paste0("__", v, "_lag", 1:k)
    lead_names <- paste0("__", v, "_lead", 1:k)
    
    for (j in 1:k) {
      dt[, (lag_names[j])  := shift(get(v),  n = j,  type = "lag"),  by = group_cols]
      dt[, (lead_names[j]) := shift(get(v),  n = j,  type = "lead"), by = group_cols]
    }
    
    dt[, paste0(v, "_prev", k) := rowMeans(.SD, na.rm = TRUE), .SDcols = lag_names]
    dt[, paste0(v, "_next", k) := rowMeans(.SD, na.rm = TRUE), .SDcols = lead_names]
    
    dt[, c(lag_names, lead_names) := NULL]
  }
  
  return(dt)
}

# -------------------------
# YoY growth maker (within groups)
# -------------------------
add_yoy_growth <- function(dt, group_cols, time_col, var, out_name) {
  setDT(dt)
  setorderv(dt, c(group_cols, time_col))
  
  # growth = (x_t / x_{t-1}) - 1 ; guard against 0/NA denominators
  dt[, (out_name) := {
    x  <- as.numeric(get(var))
    xl <- shift(x, 1, type = "lag")
    out <- ifelse(!is.na(x) & !is.na(xl) & xl > 0, (x / xl) - 1, NA_real_)
    out
  }, by = group_cols]
  
  return(dt)
}

# -------------------------
# Run DID for a dataset
# -------------------------
run_did_block <- function(dt, level, outcomes, treatments, k = 5,
                          fe, cluster) {
  rows <- list()
  
  for (y in outcomes) {
    y_prev <- paste0(y, "_prev", k)
    y_next <- paste0(y, "_next", k)
    if (!all(c(y_prev, y_next) %in% names(dt))) next
    
    for (tr in treatments) {
      if (!tr %in% names(dt)) next
      
      fml <- as.formula(paste0(y_next, " ~ ", tr, " + ", y_prev, " | ", fe))
      
      cat("\n\n============================================\n")
      cat("[", level, "]\n", sep = "")
      cat("Outcome:", y, " (next", k, ")\n", sep = "")
      cat("Treatment:", tr, "\n")
      cat("Controls:", y_prev, "\n")
      cat("FE:", fe, "\n")
      cat("Cluster:", cluster, "\n")
      cat("============================================\n")
      
      est <- feols(fml, data = dt, cluster = as.formula(paste0("~", cluster)))
      print(summary(est))
      
      ct <- as.data.frame(coeftable(est))
      ct$term <- rownames(ct)
      
      tr_row   <- ct[ct$term == tr, , drop = FALSE]
      prev_row <- ct[ct$term == y_prev, , drop = FALSE]
      
      rows[[length(rows) + 1]] <- data.frame(
        level = level,
        base_var = y,
        window_k = k,
        treatment = tr,
        coef_treat = if (nrow(tr_row) > 0) tr_row$Estimate[1] else NA_real_,
        se_treat   = if (nrow(tr_row) > 0) tr_row$`Std. Error`[1] else NA_real_,
        p_treat    = if (nrow(tr_row) > 0) tr_row$`Pr(>|t|)`[1] else NA_real_,
        coef_prev  = if (nrow(prev_row) > 0) prev_row$Estimate[1] else NA_real_,
        se_prev    = if (nrow(prev_row) > 0) prev_row$`Std. Error`[1] else NA_real_,
        p_prev     = if (nrow(prev_row) > 0) prev_row$`Pr(>|t|)`[1] else NA_real_,
        nobs       = nobs(est),
        stringsAsFactors = FALSE
      )
    }
  }
  
  bind_rows(rows)
}

# -------------------------
# MAIN
# -------------------------
log_start()

cat("Loading BCY:", IN_BCY, "\n")
bcy <- fread(IN_BCY)

cat("Loading CY :", IN_CY, "\n")
cy  <- fread(IN_CY)

# Ensure key columns
stopifnot(all(c("loan_cert", "borrower_county", "year") %in% names(bcy)))
stopifnot(all(c("borrower_county", "year") %in% names(cy)))

# Rename county for convenience
setnames(bcy, "borrower_county", "county")
setnames(cy,  "borrower_county", "county")

# --- Create log approval outcomes ---
bcy[, log_total_gross_approval := log(as.numeric(total_gross_approval) + 1)]
cy[,  log_total_gross_approval := log(as.numeric(total_gross_approval) + 1)]

# --- Create YoY loan growth (based on n_loans) ---
bcy <- add_yoy_growth(bcy, group_cols = c("loan_cert", "county"),
                      time_col = "year", var = "n_loans", out_name = "loan_growth_yoy")
cy  <- add_yoy_growth(cy,  group_cols = c("county"),
                      time_col = "year", var = "n_loans", out_name = "loan_growth_yoy")

# Treatments
treatments <- c("closure", "exog_closure", "true_exog_closure")

# Outcomes (edit freely)
# NOTE: we include loan_growth_yoy and log_total_gross_approval (instead of total_gross_approval)
outcomes_bcy <- c(
  "avg_distance_miles",
  "chgoff_rate_all",
  "chgoff_rate_resolved",
  "n_loans",
  "loan_growth_yoy",
  "log_total_gross_approval",
  "interest_rate_mean",
  "term_months_mean",
  "fixed_share",
  "variable_share",
  "collateral_share",
  "revolver_share"
)

outcomes_cy <- c(
  "avg_distance_miles",
  "chgoff_rate_all",
  "chgoff_rate_resolved",
  "n_loans",
  "loan_growth_yoy",
  "log_total_gross_approval",
  "interest_rate_mean_w",
  "term_months_mean_w",
  "fixed_share_w",
  "variable_share_w",
  "collateral_share_w",
  "revolver_share_w"
)

# Build prev/next windows (k=5)
cat("\nBuilding prev/next windows (k=5)...\n")
bcy <- make_windows(bcy, group_cols = c("loan_cert", "county"),
                    time_col = "year", vars = outcomes_bcy, k = 5)
cy  <- make_windows(cy, group_cols = c("county"),
                    time_col = "year", vars = outcomes_cy, k = 5)

# Run DID
cat("\nRunning BCY DID...\n")
res_bcy <- run_did_block(
  dt = bcy,
  level = "BCY",
  outcomes = outcomes_bcy,
  treatments = treatments,
  k = 5,
  fe = "loan_cert + county + year",
  cluster = "loan_cert"
)

cat("\nRunning CY DID...\n")
res_cy <- run_did_block(
  dt = cy,
  level = "CY",
  outcomes = outcomes_cy,
  treatments = treatments,
  k = 5,
  fe = "county + year",
  cluster = "county"
)

res <- bind_rows(res_bcy, res_cy)

cat("\nSaving results to:", OUTCSV, "\n")
fwrite(res, OUTCSV)

cat("\nDone.\n")
log_stop()
