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

## ------------------------------------------------------------------
## (Optional) Example: add n_closure to dt_cy from dt_bcy
## ------------------------------------------------------------------
## This assumes dt_bcy has:
## - a binary indicator `closure` (1 if branch closes in that year),
## - keys `county_id` and `year` (or whatever your county-year IDs are).
## Adjust names as needed.

n_closure_cy <- dt_bcy[any_closure == 1,
                       .(n_closure = .N),
                       by = .(county, year)]

dt_cy <- merge(dt_cy, n_closure_cy, by  = c("county", "year"), all.x = TRUE)
dt_cy[is.na(n_closure), n_closure := 0L]

vars <- c("dist_mean", "dist_median", "loan_growth_next5", "default_rate", "pif_rate")

get_stats <- function(DT, idx, label, vars) {
  subDT <- DT[idx, ..vars]
  out_list <- c(
    list(group = label,
         N     = nrow(subDT)),
    lapply(subDT, function(x) mean(x, na.rm = TRUE))
  )
  as.data.table(out_list)
}

# Build each row of the summary table
stats_overall <- get_stats(dt_cy, rep(TRUE, nrow(dt_cy)), "Overall", vars)

stats_no_closure <- get_stats(
  dt_cy,
  dt_cy$any_closure == 0 | is.na(dt_cy$any_closure),
  "No closure",
  vars
)

stats_any_closure  <- get_stats(
  dt_cy,
  dt_cy$any_closure == 1,
  "Any closure",
  vars
)

stats_any_exog     <- get_stats(
  dt_cy,
  dt_cy$any_exog_closure == 1,
  "Any exog closure",
  vars
)

stats_true_exog    <- get_stats(
  dt_cy,
  dt_cy$any_true_exog == 1,
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

# Optional: round numeric columns for nicer display
num_cols <- setdiff(names(summary_table), c("group", "N"))
summary_table[, (num_cols) := lapply(.SD, function(x) round(x, 3)), .SDcols = num_cols]

## ------------------------------------------------------------------
## Output: console, CSV, and LaTeX
## ------------------------------------------------------------------

# 1) Print to console
print(summary_table)

# 2) Save to CSV
csv_path <- file.path(out_dir, "summary_stats_by_closure.csv")
fwrite(summary_table, csv_path)

# 3) Save to LaTeX
tex_path <- file.path(out_dir, "summary_stats_by_closure.tex")

col_names <- names(summary_table)
header    <- paste(col_names, collapse = " & ")

# Convert all cells to character for LaTeX
summary_char <- summary_table[, lapply(.SD, as.character)]

body_lines <- apply(summary_char, 1, function(row) {
  paste(row, collapse = " & ")
})

tex_lines <- c(
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{Summary statistics by closure status}",
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

# Also echo LaTeX to console if you want to quickly copy/paste:
cat(paste(tex_lines, collapse = "\n"), "\n")
