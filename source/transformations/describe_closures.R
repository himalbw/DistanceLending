library(data.table)
library(fixest)
library(ggplot2)
library(broom)   # <- add this

bcy_path  <- "source/transformations/data/sba_bcy.csv"
cy_path   <- "source/transformations/data/sba_cy.csv"
out_dir   <- "output/descriptives"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
es_log_path <- file.path(out_dir, "summary_stats.log")

sba_bcy <- data.table::fread(bcy_path)
dt_bcy  <- as.data.table(sba_bcy)

sba_cy  <- data.table::fread(cy_path)
dt_cy   <- as.data.table(sba_cy)

n_closure_cy <- dt_bcy[any_closure == 1,
                       .(n_closure = .N),
                       by = .(county, year)]

dt_cy <- merge(dt_cy, n_closure_cy, by = c("county", "year"), all.x = TRUE)
dt_cy[is.na(n_closure), n_closure := 0L]

n_exog_closure_cy <- dt_bcy[any_exog_closure == 1,
                            .(n_exog_closure = .N),
                            by = .(county, year)]

dt_cy <- merge(dt_cy, n_exog_closure_cy, by = c("county", "year"), all.x = TRUE)
dt_cy[is.na(n_exog_closure), n_exog_closure := 0L]

n_true_exog_cy <- dt_bcy[any_true_exog == 1,
                         .(n_true_exog = .N),
                         by = .(county, year)]

dt_cy <- merge(dt_cy, n_true_exog_cy, by = c("county", "year"), all.x = TRUE)
dt_cy[is.na(n_true_exog), n_true_exog := 0L]


vars_cy <- c("dist_mean", "dist_median", "total_loan_growth",
             "default_rate", "pif_rate", "n_branches_prev5")

# For BCY, take the subset of vars_cy that exist in dt_bcy
vars_bcy <- intersect(vars_cy, names(dt_bcy))

get_stats <- function(DT, idx, label, vars) {
  subDT <- DT[idx, ..vars]
  out_list <- c(
    list(group = label,
         N     = nrow(subDT)),
    lapply(subDT, function(x) mean(x, na.rm = TRUE))
  )
  as.data.table(out_list)
}

make_summary_table <- function(DT, vars, out_dir, prefix, caption) {
  # Build each row of the summary table
  stats_overall <- get_stats(DT, rep(TRUE, nrow(DT)), "Overall", vars)
  
  stats_no_closure <- get_stats(
    DT,
    DT$any_closure == 0 | is.na(DT$any_closure),
    "No closure",
    vars
  )
  
  stats_any_closure  <- get_stats(
    DT,
    DT$any_closure == 1,
    "Any closure",
    vars
  )
  
  stats_any_exog     <- get_stats(
    DT,
    DT$any_exog_closure == 1,
    "Any exog closure",
    vars
  )
  
  stats_true_exog    <- get_stats(
    DT,
    DT$any_true_exog == 1,
    "Any true exog closure",
    vars
  )
  
  # Combine into one table
  summary_table <- rbindlist(
    list(
      stats_overall,
      stats_no_closure,
      stats_any_closure,
      stats_any_exog,
      stats_true_exog
    ),
    use.names = TRUE,
    fill      = TRUE
  )
  
  # Round numeric columns for nicer display
  num_cols <- setdiff(names(summary_table), c("group", "N"))
  summary_table[, (num_cols) := lapply(.SD, function(x) round(x, 3)),
                .SDcols = num_cols]
  
  ## ------------------------------------------------------------------
  ## Output: console, CSV, and LaTeX
  ## ------------------------------------------------------------------
  
  # 1) Print to console (label which table this is)
  cat("\n============================\n")
  cat("Summary table:", prefix, "\n")
  cat("============================\n")
  print(summary_table)
  
  # 2) Save to CSV
  csv_path <- file.path(out_dir, paste0(prefix, "_summary_stats_by_closure.csv"))
  fwrite(summary_table, csv_path)
  
  # 3) Save to LaTeX
  tex_path <- file.path(out_dir, paste0(prefix, "_summary_stats_by_closure.tex"))
  
  col_names <- names(summary_table)
  header    <- paste(col_names, collapse = " & ")
  
  summary_char <- summary_table[, lapply(.SD, as.character)]
  body_lines <- apply(summary_char, 1, function(row) {
    paste(row, collapse = " & ")
  })
  
  tex_lines <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    paste0("\\caption{", caption, "}"),
    sprintf("\\begin{tabular}{%s}", paste(rep("c", length(col_names)), collapse = "")),
    "\\hline",
    paste0(header, " \\\\"),
    "\\hline",
    paste0(body_lines, " \\\\"),
    "\\hline",
    "\\end{tabular}",
    "\\end{table}"
  )
  
  writeLines(tex_lines, tex_path)
  
  # Also echo LaTeX to console if you want to copy/paste quickly
  cat("\nLaTeX for", prefix, "table:\n")
  cat(paste(tex_lines, collapse = "\n"), "\n")
  
  invisible(summary_table)
}

## ------------------------------------------------------------------
## Call for CY and BCY
## ------------------------------------------------------------------

# County-year summary table
summary_cy  <- make_summary_table(
  DT      = dt_cy,
  vars    = vars_cy,
  out_dir = out_dir,
  prefix  = "cy",
  caption = "County-year summary statistics by closure status"
)

# Branch-county-year summary table
summary_bcy <- make_summary_table(
  DT      = dt_bcy,
  vars    = vars_bcy,
  out_dir = out_dir,
  prefix  = "bcy",
  caption = "Branch-county-year summary statistics by closure status"
)
