args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) normalizePath(args[[1]]) else normalizePath(getwd())
folder <- file.path(root, "outputs", "visitor_arrivals", "gt")
.libPaths(c(file.path(root, "outputs", "visitor_arrivals", "R-library"), .libPaths()))
library(gt)
library(jsonlite)
jobs <- fromJSON(file.path(folder, "input.json"), simplifyVector = FALSE)
rendered <- list()
inventory <- list()
for (job in jobs) {
  columns <- c("term", names(job$labels))
  data <- as.data.frame(setNames(lapply(columns, function(column) {
    vapply(job$rows, function(row) if (is.null(row[[column]])) "—" else as.character(row[[column]]), character(1))
  }), columns), stringsAsFactors = FALSE)
  table_id <- paste0("gt_", job$key)
  labels <- lapply(job$labels, html)
  tbl <- gt(data, rowname_col = "term", id = table_id) |>
    tab_header(title = job$title, subtitle = job$subtitle) |>
    tab_stubhead(label = if (job$source == "profiles") "Origin market" else "Regressor / sample") |>
    cols_label(.list = labels) |>
    fmt_markdown(columns = everything()) |>
    cols_align(align = "center", columns = everything()) |>
    tab_source_note(source_note = job$note) |>
    tab_options(table.font.names = c("Arial", "Helvetica", "sans-serif"),
      table.font.size = px(13), table.width = pct(100),
      heading.align = "left", heading.title.font.size = px(17),
      heading.subtitle.font.size = px(12), data_row.padding = px(9),
      column_labels.background.color = "#eef4f7", column_labels.font.weight = "bold",
      source_notes.font.size = px(11), source_notes.padding = px(10),
      table.additional_css = paste0("#",table_id," .gt_table td, #",table_id," .gt_table th {white-space: normal;} #",table_id," .gt_col_heading, #",table_id," .gt_row {min-width: 130px;} #",table_id," .gt_stub {min-width: 175px;}"),
      table.border.top.color = "#456a82", table.border.bottom.color = "#d8e0e5",
      table_body.hlines.color = "#e5e9ed", stub.border.color = "#e5e9ed")
  content <- as_raw_html(tbl, inline_css = FALSE)
  rendered[[job$key]] <- content
  standalone <- paste0('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full visitor regression table</title></head><body>',content,'</body></html>')
  writeLines(standalone, file.path(folder, paste0(job$key, ".html")), useBytes = TRUE)
  inventory[[job$key]] <- list(rows = nrow(data), columns = ncol(data),
    model_ids = job$model_ids, inference = job$inference, source = job$source)
}
write_json(rendered, file.path(folder, "tables.json"), auto_unbox = TRUE, pretty = FALSE)
write_json(list(status = "rendered", renderer = "R gt", gt_version = as.character(packageVersion("gt")),
  r_version = R.version.string, tables = inventory), file.path(folder, "manifest.json"), auto_unbox = TRUE, pretty = TRUE)
cat("Rendered", length(rendered), "native R gt regression tables\n")
