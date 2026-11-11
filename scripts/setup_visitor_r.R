# Restore only the visitor HTML renderer's R dependencies, without activating an
# renv project or touching user/global libraries. Python checks do not need R.
#
# Rscript --vanilla scripts/setup_visitor_r.R ROOT [--library PATH] [--check-only]
# Tested runtime: R 4.6.1. On Debian/Ubuntu, source builds require build-essential,
# pkg-config, libuv1-dev, libcurl4-openssl-dev, libssl-dev, libxml2-dev and libnode-dev (or the
# distribution's libv8-dev provider). R itself is installed separately.
# Reference: https://rstudio.github.io/renv/reference/restore.html

args <- commandArgs(trailingOnly = TRUE)
all_args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", all_args, value = TRUE)
script_root <- if (length(script_arg)) dirname(dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))) else getwd()
root <- script_root
library_arg <- NULL
check_only <- FALSE
allow_r_mismatch <- FALSE
positional <- FALSE
i <- 1L
while (i <= length(args)) {
  arg <- args[[i]]
  if (arg == "--check-only") {
    check_only <- TRUE
  } else if (arg == "--allow-r-version-mismatch") {
    allow_r_mismatch <- TRUE
  } else if (arg == "--library") {
    i <- i + 1L
    if (i > length(args)) stop("--library requires a path", call. = FALSE)
    library_arg <- args[[i]]
  } else if (arg %in% c("--help", "-h")) {
    cat("Usage: Rscript --vanilla scripts/setup_visitor_r.R [ROOT] [--library PATH] [--check-only] [--allow-r-version-mismatch]\n")
    quit(status = 0L)
  } else if (!startsWith(arg, "-") && !positional) {
    root <- arg
    positional <- TRUE
  } else {
    stop("Unrecognized argument: ", arg, call. = FALSE)
  }
  i <- i + 1L
}
root <- normalizePath(root, mustWork = TRUE)
lockfile <- file.path(root, "replication", "renv.lock")
if (!file.exists(lockfile)) stop("Missing frozen R lockfile: ", lockfile, call. = FALSE)
target <- if (is.null(library_arg)) file.path(root, "outputs", "visitor_arrivals", "R-library") else library_arg
if (!grepl("^(/|[A-Za-z]:[/\\\\])", target)) target <- file.path(root, target)
target <- normalizePath(target, mustWork = FALSE)
tested_r <- "4.6.1"
renv_version <- "1.3.0"
bootstrap_url <- paste0("https://cran.r-project.org/src/contrib/Archive/renv/renv_", renv_version, ".tar.gz")
if (as.character(getRversion()) != tested_r) {
  message <- paste0("Locked/tested R is ", tested_r, "; this runtime is ", getRversion(), ". Install the tested R version, or explicitly use --allow-r-version-mismatch for an unvalidated runtime.")
  if (!allow_r_mismatch) stop(message, call. = FALSE)
  warning(message, call. = FALSE)
}
if (!dir.exists(target)) {
  if (check_only) stop("Local visitor R library is missing. Run this helper without --check-only first.", call. = FALSE)
  dir.create(target, recursive = TRUE, showWarnings = FALSE)
}
target <- normalizePath(target, mustWork = TRUE)
if (target == root) stop("R library must be a dedicated directory, not the repository root", call. = FALSE)

# All package state and any downloaded source/cache files stay below the excluded
# local library. No renv::init(), activation, personal .Rprofile, or user cache.
Sys.setenv(
  RENV_CONFIG_AUTO_SNAPSHOT = "FALSE", RENV_CONFIG_USER_PROFILE = "FALSE",
  RENV_CONFIG_CACHE_ENABLED = "FALSE", RENV_CONFIG_SANDBOX_ENABLED = "FALSE",
  RENV_PATHS_ROOT = file.path(target, ".renv-state"),
  RENV_PATHS_CACHE = file.path(target, ".renv-state", "cache"),
  RENV_PATHS_SOURCE = file.path(target, ".renv-state", "sources"),
  RENV_PATHS_BINARY = file.path(target, ".renv-state", "binaries"),
  RENV_PATHS_LIBRARY = target, R_LIBS_USER = target, R_LIBS_SITE = ""
)
.libPaths(c(target, .Library), include.site = FALSE)
options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 600, Ncpus = 1L,
        renv.config.auto.snapshot = FALSE, renv.config.user.profile = FALSE,
        renv.config.install.transactional = TRUE)
installed <- installed.packages(lib.loc = target)
has_renv <- "renv" %in% rownames(installed) && installed["renv", "Version"] == renv_version
if (!has_renv) {
  if (check_only) stop("Pinned renv ", renv_version, " is missing. Run this helper without --check-only first.", call. = FALSE)
  cat("Bootstrapping pinned renv", renv_version, "from", bootstrap_url, "\n")
  install.packages(bootstrap_url, repos = NULL, type = "source", lib = target, quiet = FALSE)
}
if (as.character(packageVersion("renv", lib.loc = target)) != renv_version) stop("renv bootstrap version mismatch", call. = FALSE)
invisible(loadNamespace("renv", lib.loc = target))
lock <- renv::lockfile_read(lockfile)
if (lock$R$Version != tested_r || lock$Packages$renv$Version != renv_version) stop("Setup helper and R lockfile pins disagree", call. = FALSE)
if (!check_only) {
  renv::restore(project = root, library = target, lockfile = lockfile,
                prompt = FALSE, clean = FALSE, retry = FALSE, transactional = TRUE)
}
installed <- installed.packages(lib.loc = target)
failures <- character()
for (name in names(lock$Packages)) {
  expected <- lock$Packages[[name]]$Version
  actual <- if (name %in% rownames(installed)) installed[name, "Version"] else "MISSING"
  if (!identical(actual, expected)) failures <- c(failures, paste0(name, ": expected ", expected, ", found ", actual))
}
if (length(failures)) stop("Locked R package verification failed:\n", paste(failures, collapse = "\n"), call. = FALSE)
invisible(loadNamespace("gt", lib.loc = target))
invisible(loadNamespace("jsonlite", lib.loc = target))
smoke <- gt::as_raw_html(gt::gt(data.frame(check = "locked native gt renderer")))
if (!is.character(smoke) || !grepl("<table", smoke, fixed = TRUE)) stop("Native gt smoke render failed", call. = FALSE)
cat("Verified", length(lock$Packages), "locked local R packages; R", as.character(getRversion()),
    "; gt", as.character(packageVersion("gt", lib.loc = target)),
    "; jsonlite", as.character(packageVersion("jsonlite", lib.loc = target)), "\n")
cat("Local library:", target, "\n")
