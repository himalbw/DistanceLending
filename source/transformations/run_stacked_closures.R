# import source/transformations/closure_stacked_es.R

library(data.table)
library(ggplot2)

bcy_path <- "source/transformations/data/sba_bcy.csv"
out_dir  <- "output/transformations/event_study"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Helper: save coefs + plot for a given ES result --------------------
save_event_study_outputs <- function(es_result, tag_suffix = "") {
  outcome   <- es_result$outcome
  treat_var <- es_result$treat_var
  ctrl_tag  <- if (is.null(es_result$controls) || length(es_result$controls) == 0L) {
    "nocontrols"
  } else {
    "withcontrols"
  }
  
  tag <- paste(outcome, treat_var, ctrl_tag, tag_suffix, sep = "_")
  
  # 1) Save coefficients
  coef_path <- file.path(out_dir, paste0("event_study_coefs_", tag, ".csv"))
  data.table::fwrite(es_result$coefs, file = coef_path)
  
  # 2) Save plot
  p <- plot_event_study(
    es_result,
    ref_period   = -1,
    shading_from = 0,
    title        = paste("Event study:", outcome, "around", treat_var),
    subtitle     = paste("Stacked design (", ctrl_tag, ")", sep = ""),
    ylab         = paste("Effect on", outcome)
  )
  
  plot_path <- file.path(out_dir, paste0("event_study_plot_", tag, ".png"))
  ggsave(plot_path, p, width = 7, height = 5, dpi = 300)
  
  message("Saved: ", coef_path)
  message("Saved: ", plot_path)
}

# Main: actually run the event studies --------------------------------
main <- function() {
  sba_bcy <- data.table::fread(bcy_path)
  outcomes   <- c("dist_mean", "default_rate", "loan_growth_next5")
  treatments <- c("any_closure", "any_exog_closure", "any_true_exog")
  control_candidates <- c("loan_growth_prev5", "bank_size", "n_branches_next5")
  for (y in outcomes) {
    for (tr in treatments) {
      if (!(y %in% names(sba_bcy)) || !(tr %in% names(sba_bcy))) {
        message("Skipping outcome=", y, ", treat=", tr,
                " (missing in data).")
        next
      }
      
      message("\n=======================================")
      message("Event study for outcome = ", y,
              ", treatment = ", tr)
      message("=======================================\n")
      es_plain <- run_stacked_event_study(
        data      = sba_bcy,
        outcome   = y,
        treat_var = tr,
        id_bank   = "loan_cert",
        id_county = "borrower_county",
        time_var  = "year",
        window_pre  = 5,
        window_post = 5
      )
      save_event_study_outputs(es_plain)
      es_ctrl <- run_stacked_with_controls(
        data         = sba_bcy,
        outcome      = y,
        treat_var    = tr,
        control_vars = control_candidates,
        id_bank      = "loan_cert",
        id_county    = "borrower_county",
        time_var     = "year",
        window_pre   = 5,
        window_post  = 5
      )
      save_event_study_outputs(es_ctrl, tag_suffix = "ctrl")
    }
  }
  
  message("\nAll event studies completed.")
}

# Run when called via Rscript -----------------------------------------
if (sys.nframe() == 0) {
  main()
}
