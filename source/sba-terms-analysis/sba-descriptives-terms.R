# ------------------------------------------------------------
# Distance to Branch and Loan Pricing / Terms
# SBA 7(a) Loans
# ------------------------------------------------------------
# Outputs:
#   (1) dist-bin levels: No bank FE vs Bank FE
#   (2) dist-bin levels by bank size (Small/Med/Large), Bank FE only
# Saved to: output/descriptives/dist_mile_pricing
#
# NOTE ON ASSET UNITS:
#   In FDIC/Call Report style data, ASSET is often in thousands of dollars.
#   Your summary (median ~1.5e7, max ~3.5e9) strongly suggests "thousands".
#   We convert: asset_bil = ASSET / 1e6  (thousands -> billions)
# ------------------------------------------------------------

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(fixest)
  library(broom)
  library(ggplot2)
})

# -------------------------
# Paths
# -------------------------
DATA_PATH <- "source/sba_analysis/data/derived/sba_7a_loan_to_branch_with_terms.csv"
OUT_DIR   <- "output/descriptives/dist_mile_pricing"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# -------------------------
# Load
# -------------------------
df_raw <- read_csv(DATA_PATH, show_col_types = FALSE)
cat("Loaded:", nrow(df_raw), "loans\n")

# -------------------------
# Construct core variables
# -------------------------
df <- df_raw %>%
  mutate(
    # Distance bins
    dist_bin = cut(
      distance_miles,
      breaks = c(0, 1, 5, 10, 25, 50, 100, 250, 500, Inf),
      right  = FALSE,
      labels = c("0–1", "1–5", "5–10", "10–25", "25–50",
                 "50–100", "100–250", "250–500", "500+")
    ),
    
    # Outcomes (continuous)
    interest_rate  = as.numeric(InitialInterestRate),
    term_months    = as.numeric(TerminMonths),
    log_gross      = log(GrossApproval + 1),
    
    # Outcomes (binary contract features)
    collateralized = as.integer(CollateralInd == "Y"),
    fixed_rate     = as.integer(FixedorVariableInterestRate == "F"),
    variable_rate  = as.integer(FixedorVariableInterestRate == "V"),
    
    # FE IDs
    bank_id = RSSDID,
    year_fe = year,
    
    # Assets: treat ASSET as "thousands of dollars" -> convert to billions
    asset_bil_row = as.numeric(ASSET) / 1e6
  ) %>%
  filter(!is.na(dist_bin), !is.na(year_fe), !is.na(bank_id))

df$dist_bin <- relevel(df$dist_bin, ref = "0–1")

# -------------------------
# Time-invariant bank size classification (median assets per bank)
# -------------------------
bank_size_map <- df %>%
  group_by(bank_id) %>%
  summarise(
    asset_bil_med = median(asset_bil_row, na.rm = TRUE),
    bank_size = case_when(
      is.na(asset_bil_med) ~ NA_character_,
      asset_bil_med < 10   ~ "Small (<$10B)",
      asset_bil_med < 100  ~ "Medium ($10–100B)",
      TRUE                 ~ "Large (≥$100B)"
    ),
    .groups = "drop"
  )

df <- df %>%
  left_join(bank_size_map, by = "bank_id") %>%
  mutate(
    bank_size = factor(
      bank_size,
      levels = c("Small (<$10B)", "Medium ($10–100B)", "Large (≥$100B)")
    )
  )

cat("\nBank-size counts (time-invariant):\n")
print(table(df$bank_size, useNA = "ifany"))

# -------------------------
# Outcomes + labels
# -------------------------
outcomes <- list(
  interest_rate   = "Interest rate (%)",
  term_months     = "Maturity (months)",
  log_gross       = "Log loan amount",
  collateralized  = "Collateralized (share)",
  fixed_rate      = "Fixed-rate (share)",
  variable_rate   = "Variable-rate (share)"
)

binary_outcomes <- c("collateralized", "fixed_rate", "variable_rate")

# -------------------------
# Helper: recover absolute bin levels (includes 0–1)
# -------------------------
recover_levels <- function(model, data, yvar) {
  
  # Mean in reference bin (0–1 miles)
  ref_mean <- data %>%
    filter(dist_bin == "0–1") %>%
    summarise(mu = mean(.data[[yvar]], na.rm = TRUE)) %>%
    pull(mu)
  
  coef_df <- tidy(model) %>%
    filter(str_detect(term, "dist_bin::")) %>%
    mutate(
      dist_bin = str_replace(term, "dist_bin::", ""),
      level = ref_mean + estimate
    )
  
  ref_row <- tibble(
    dist_bin = "0–1",
    level = ref_mean,
    std.error = NA_real_
  )
  
  bind_rows(ref_row, coef_df) %>%
    mutate(dist_bin = factor(dist_bin, levels = levels(data$dist_bin)))
}

# -------------------------
# Helper: dynamic y-lims with padding (for binary outcomes)
# -------------------------
pad_ylim <- function(yvals, pad = 0.1) {
  y_min <- min(yvals, na.rm = TRUE)
  y_max <- max(yvals, na.rm = TRUE)
  c(y_min - pad, y_max + pad)
}

# ============================================================
# (1) Main plots: No bank FE vs Bank FE
# ============================================================
for (y in names(outcomes)) {
  
  reg_nofe <- feols(
    as.formula(paste0(y, " ~ i(dist_bin) | year_fe")),
    data = df
  )
  
  reg_bankfe <- feols(
    as.formula(paste0(y, " ~ i(dist_bin) | bank_id + year_fe")),
    data = df
  )
  
  lev_nofe <- recover_levels(reg_nofe, df, y) %>%
    mutate(model = "No bank fixed effects")
  
  lev_fe <- recover_levels(reg_bankfe, df, y) %>%
    mutate(model = "Bank fixed effects")
  
  plot_df <- bind_rows(lev_nofe, lev_fe)
  
  p <- ggplot(plot_df, aes(x = dist_bin, y = level, color = model)) +
    geom_point(position = position_dodge(width = 0.45), size = 2.4) +
    geom_errorbar(
      aes(
        ymin = level - 1.96 * std.error,
        ymax = level + 1.96 * std.error
      ),
      position = position_dodge(width = 0.45),
      width = 0.25,
      na.rm = TRUE
    ) +
    labs(
      title = paste0(outcomes[[y]], " vs. borrower–branch distance"),
      subtitle = "Predicted bin means from regressions; error bars show 95% CIs (when available)",
      x = "Borrower–branch distance (miles)",
      y = outcomes[[y]],
      color = ""
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "top",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  if (y %in% binary_outcomes) {
    p <- p + coord_cartesian(ylim = pad_ylim(plot_df$level, pad = 0.1))
  }
  
  out_path <- file.path(OUT_DIR, paste0("dist_bin_levels_", y, ".png"))
  ggsave(out_path, p, width = 9, height = 5)
  cat("Saved:", out_path, "\n")
}

# ============================================================
# (2) Bank-size splits: Bank FE only, estimated separately by size
# ============================================================
df_bs <- df %>% filter(!is.na(bank_size))

for (y in names(outcomes)) {
  
  res_list <- list()
  
  for (sz in levels(df_bs$bank_size)) {
    
    dsub <- df_bs %>% filter(bank_size == sz)
    
    # Skip tiny groups
    if (nrow(dsub) < 500) next
    
    reg_sz <- feols(
      as.formula(paste0(y, " ~ i(dist_bin) | bank_id + year_fe")),
      data = dsub
    )
    
    lev_sz <- recover_levels(reg_sz, dsub, y) %>%
      mutate(bank_size = sz)
    
    res_list[[sz]] <- lev_sz
  }
  
  plot_df <- bind_rows(res_list)
  if (nrow(plot_df) == 0) next
  
  p <- ggplot(plot_df, aes(x = dist_bin, y = level, color = bank_size)) +
    geom_point(position = position_dodge(width = 0.55), size = 2.4) +
    geom_errorbar(
      aes(
        ymin = level - 1.96 * std.error,
        ymax = level + 1.96 * std.error
      ),
      position = position_dodge(width = 0.55),
      width = 0.25,
      na.rm = TRUE
    ) +
    labs(
      title = paste0(outcomes[[y]], " vs. distance, by bank size (bank FE)"),
      subtitle = "Separate regressions by bank size; all include bank and year fixed effects",
      x = "Borrower–branch distance (miles)",
      y = outcomes[[y]],
      color = ""
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "top",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  if (y %in% binary_outcomes) {
    p <- p + coord_cartesian(ylim = pad_ylim(plot_df$level, pad = 0.1))
  }
  
  out_path <- file.path(OUT_DIR, paste0("dist_bin_levels_by_bank_size_", y, ".png"))
  ggsave(out_path, p, width = 9.5, height = 5.2)
  cat("Saved:", out_path, "\n")
}

cat("\nAll figures written to:", OUT_DIR, "\n")
