library(ggplot2)
library(ggridges)

#' Volcano plot function
#'
#' Creates a volcano plot showing the relationship between a statistic
#' (e.g., effect size, log fold change) and statistical significance.
#' The y-axis represents -log10(p-values) or adjusted p-values, and
#' points are colored by significance based on a chosen threshold.
#'
#' @param statistic Numeric vector. The effect size or test statistic
#'  for each feature.
#'
#' @param pval_statistic Numeric vector. Corresponding p-values for each
#' statistic.
#'
#' @param x_title Character. Label for the x-axis.
#'
#' @param alpha Numeric. Significance threshold for (adjusted) p-values.
#' Default is 0.05.
#'
#' @param method Character. Method for p-value adjustment (passed to
#' p.adjust), e.g., "BH", "bonferroni", etc.
#'
#' @param p_max Numeric. Lower bound for p-values to avoid taking log(0).
#' Default is 1e-4.
#'
#' @param poin_size Numeric. Size of the plotted points.
#'
#' @param title_size Numeric. Font size of axis titles.
#'
#' @param text_size Numeric. Font size of axis tick labels.
#'
#' @param legend_text_size Numeric. Font size of legend text.
#'
#' @param adjust Logical. Whether to adjust p-values (TRUE) or use raw
#' p-values (FALSE).
#'
#' @param legend_pos Numeric vector of length 2. Position of the legend
#' inside the plot (x, y coordinates).
#'
#' @return A ggplot2 object representing the volcano plot.
#'

plot_volcano <- function(statistic, pval_statistic, x_title, alpha = 0.05,
                         method="BH", p_max=1e-4, poin_size=8, title_size=60,
                         text_size=35, legend_text_size=30, adjust = TRUE,
                         legend_pos=c(0.05, 0.8)) {
  
  adj_pval = 0.
  y_title = "Adjusted p-value"
  
  # Adjust p-values
  if (adjust){
      adj_pval <- p.adjust(pval_statistic, method = method)
      adj_pval <- pmax(adj_pval, p_max)
  } else{
      adj_pval <- pval_statistic
      adj_pval <- pmax(adj_pval, p_max)
      y_title = "p-value"}
  
  # Create dataframe
  df <- data.frame(statistic = statistic, pval = pval_statistic,
                   adj_pval = adj_pval, neg_log10_adj_pval = -log10(adj_pval),
                   significance = ifelse(adj_pval < alpha, "Significant",
                                         "Not significant"))
  
  # Plot
  ggplot(df, aes(x = statistic, y = neg_log10_adj_pval, color = significance,
                 shape = significance)) +
    geom_point(size = poin_size, alpha = 0.5) +
    labs(x = x_title, y = bquote(-log[10](.(y_title)))) +
    scale_color_manual(values = c("Significant" = "#AE123A",
                                  "Not significant" = "#59636D")) +
    scale_shape_manual(values = c("Significant" = 16,
                                  "Not significant" = 15)) +
    theme_minimal(base_size = 14) +
    theme(panel.grid = element_blank(),
          axis.line = element_line(color = "black", linewidth = 2),     
          axis.title = element_text(size = title_size),         
          axis.text = element_text(color = "black", size = text_size),           
          legend.position = legend_pos, legend.justification = c("left", "top"),
          legend.text = element_text(size = legend_text_size),
          legend.title = element_blank(),
          legend.background = element_rect(fill = "white", color = "black",
                                           size = 1.5))}

#' LOESS smoothing plot
#'
#' Creates a scatter plot of (x, y) data with a LOESS and optional confidence
#' interval. 
#'
#' @param data Data frame containing at least two columns named `x` and `y`.
#'
#' @param x_title Character. Label for the x-axis.
#'
#' @param y_title Character. Label for the y-axis.
#'
#' @param size Numeric. Size of the scatter plot points.
#'
#' @param size_line Numeric. Line width of the LOESS curve.
#'
#' @param linewidth_axis Numeric. Line width of the plot axes.
#'
#' @param size_title Numeric. Font size of axis titles.
#'
#' @param size_text Numeric. Font size of axis tick labels.
#'
#' @param size_ticks Numeric. (Currently unused) Intended size for axis ticks.
#'
#' @param alpha Numeric. Transparency level of the points (0–1).
#'
#' @param line_color Character. Color of the LOESS curve.
#'
#' @param color Character. Fill color of the points.
#'
#' @param span Numeric. Smoothing parameter for LOESS.
#'
#' @param outline_color Character. Border color of the points.
#'
#' @param outline_size Numeric. Width of the point border (stroke).
#'
#' @return A ggplot2 object representing the scatter plot with LOESS smoothing.
#'
LR_smooth_plot <- function(data, x_title, y_title,
                           size = 8, size_line = 5,
                           linewidth_axis = 3,
                           size_title = 60,
                           size_text = 35,
                           size_ticks = 30,
                           alpha = 0.5,
                           line_color = "blue",
                           color = "black",
                           span = 0.75,
                           outline_color = "black",
                           outline_size = 1){
  
  smoother <- ggplot2::stat_smooth(method = "loess", span = span, se = TRUE,
                                   color = line_color, linewidth = size_line)
  
  p <- ggplot2::ggplot(data, ggplot2::aes(x = x, y = y)) + ggplot2::geom_point(
       size = size, alpha = alpha, shape = 21, fill = color,
       color = outline_color, stroke = outline_size) + smoother +
       ggplot2::labs(x = x_title, y = y_title) +
       ggplot2::theme_classic() + ggplot2::theme(
       panel.grid = ggplot2::element_blank(),
       axis.line = ggplot2::element_line(color = "black",
                                         linewidth = linewidth_axis),
       axis.title = ggplot2::element_text(size = size_title, color = "black"),
       axis.text = ggplot2::element_text(size = size_text, color = "black"))
  
  return(p)}

#' Principal Coordinates Analysis (PCoA)
#'
#' @param data Numeric matrix or data frame. Rows represent samples and
#' columns represent features.
#'
#' @param method Character. Dissimilarity metric to use in \code{vegdist},
#' such as "jaccard", "bray", "euclidean", etc. Default is "jaccard".
#'
#' @return A matrix of coordinates (n × 2), where each row corresponds to
#' a sample projected onto the first two principal coordinate axes.
#'

PCoA <- function(data, method = "jaccard") {
  
  dissimilarity_matrix <- vegdist(data, method = method)
  coordinates <- cmdscale(dissimilarity_matrix, k = 2)
  
  return(coordinates)
}

#' Plot recovery trajectories based on PCoA outputs.
#'
#' @param coordinates A matrix of coordinates (n × 2), where each row
#'  corresponds to a sample projected onto the first two principal
#'  coordinate axes.
#'  
#' @param colors Character vector. Color assigned to each sample. Must have
#' the same length as the number of rows in `coordinates`.
#'
#' @param shapes Numeric or integer vector. Plotting shape assigned to each
#' sample. Must have the same length as the number of rows in `coordinates`.
#'
#' @param m Integer. Number of final samples (rows) to connect with a path.
#' If `m > 0`, the last `m` points are joined in their existing order.
#'
#' @param title Character or NULL. Optional plot title.
#'
#' @param level Numeric. Confidence level for the normal-data ellipses drawn
#' by `stat_ellipse`. Default is 0.95.
#'
#' @return A ggplot2 object representing the recovery trajectory plot.
#' 

plot_recovery_trajectories <- function(coordinates, colors, shapes, m,
                                       title = NULL, level = 0.95){

  data_df <- data.frame(PCoA1 = coordinates[, 1], PCoA2 = coordinates[, 2],
                        CustomColor = colors, CustomShape = shapes)
   
   p <- ggplot(data_df, aes(x = PCoA1, y = PCoA2, colour = CustomColor,
                            shape  = CustomShape)) +
        geom_point(size = 6, stroke = 2, alpha = 0.6)
   
   p <- p + scale_color_manual(values = c("#ABABAB" = "#ABABAB",
                                          "#51A246" = "#51A246",
                                          "#1A71B8" = "#1A71B8",
                                          "#E33A3A" = "#E33A3A",
                                          "pink" = "#8C2928",
                                          "brown" = "#8C2928",
                                          "black" = "#8C2928"),
                               name = "Sample type",
                               guide = "legend")
   
   p <- p + scale_shape_identity(breaks = c( 7, 8, 15, 16, 17),
                                 labels = c("Baseline", "ABX", "Follow-up",
                                            "General healthy baseline",
                                            "Healthy baseline clusters"),
                                 name = "Sample type", guide = "legend")
   
   p <- p + stat_ellipse(data = subset(data_df, CustomColor == "#51A246"),
                         aes(group = CustomColor), type = "norm", level = level,
                         linewidth = 1)
  
  p <- p + labs(x = "PCo1", y = "PCo2") + theme_minimal() + theme(
    panel.background = element_rect(fill = "white", colour = "black", linewidth = 2),
    panel.grid = element_blank(),
    axis.title = element_text(size = 40),
    axis.text = element_blank())
  
  for (grp in setdiff(unique(colors),
                      c("#ABABAB", "#51A246", "#1A71B8", "#E33A3A"))) {
    p <- p + stat_ellipse(data   = subset(data_df, CustomColor == grp),
                          aes(group = CustomColor), colour = "#8C2928",
                          type = "norm", level = level, linewidth = 1)}
  
  if (m > 0 && m < nrow(coordinates)) {
     
    last_m_df <- tail(data_df, m)
      
    p <- p + geom_path(data = last_m_df, aes(x = PCoA1, y = PCoA2),
                       colour = "black", linewidth = 1)} 
  
  if (!is.null(title))
    p <- p + ggtitle(title) + theme(plot.title = element_text(size = 30))
  
  p <- p + guides(colour = "none", shape  = guide_legend(title = "Sample type",
                  override.aes = list(alpha = 1))) +
    theme(legend.key.width  = unit(1.4, "cm"),
          legend.key.height = unit(1.0, "cm"),
          legend.spacing.x  = unit(0.6, "cm"), legend.title = element_blank(),  
          legend.text  = element_text(size = 16))
  
  p <- p + theme(legend.position = "inside",
                 legend.position.inside = c(0.76, 0.83),
                 legend.justification = c(0.5, 0.5),
                 legend.background = element_rect(fill = alpha("white", 0.7)))
  return(p)}


plot_knn_richness_test <- function(X_s_binary, X_n_binary, taxa_id = 1, k = 5,
                                   k_far = NULL, c = NULL, partial = TRUE,
                                   binary = TRUE, add = TRUE,
                                   point_col = "grey70",
                                   nn_col = "#FF8C00", far_col = "#3B7DDD",
                                   test_col = "red", cex_first = 3, cex_nn = 3,
                                   cex_far = 3, cex_others = 2, legend = TRUE,
                                   legend_pos = 'bottomleft', title = " ",
                                   comp = c(1, 2),
                                   test_taxa = NULL, test_taxa_col = "#007A33", 
                                   test_taxa_alpha = 0.6) {
  
  # Input checks
  if (!is.matrix(X_s_binary) && !is.data.frame(X_s_binary))
    stop("X_s_binary must be a matrix/data.frame.")
  if (!is.matrix(X_n_binary) && !is.data.frame(X_n_binary))
    stop("X_n_binary must be a matrix/data.frame.")
  
  n <- nrow(X_s_binary)
  if (n != nrow(X_n_binary)) stop(
    "X_s_binary and X_n_binary must have same #rows (samples).")
  if (n < 3) stop("Need at least 3 samples (reference + >=2 others).")
  
  # taxa_id validation
  if (is.null(taxa_id) || length(taxa_id) < 1) stop(
    "taxa_id must select at least one column.")
  p_n <- ncol(X_n_binary)
  if (is.character(taxa_id)) {
    if (is.null(colnames(X_n_binary))) stop(
      "X_n_binary has no column names; taxa_id cannot be character.")
    if (!all(taxa_id %in% colnames(X_n_binary)))
      stop("Some taxa_id names not found in X_n_binary.")
  } else if (is.numeric(taxa_id)) {
    if (any(taxa_id < 1 | taxa_id > p_n)) stop(
      "taxa_id out of column range for X_n_binary.")
  } else if (is.logical(taxa_id)) {
    if (length(taxa_id) != p_n) stop(
      "Logical taxa_id must have length ncol(X_n_binary).")
    if (!any(taxa_id)) stop("Logical taxa_id selects no columns.")
  } else stop("taxa_id must be numeric, character, or logical vector.")
  if (length(taxa_id) > p_n) stop(
    "taxa_id length exceeds #features of X_n_binary.")
  
  if (length(k) != 1 || is.na(k) || k < 0 || k > (n - 2))
    stop("k must be an integer in [0, n - 2].")
  k <- as.integer(k)
  
  if (is.null(k_far)) {
    k_far <- n - k - 1
  } else {
    if (length(k_far) != 1 || is.na(k_far) || k_far < 0 || k_far > (n - k - 1))
      stop("k_far must be in [0, n - k - 1] or NULL.")
    k_far <- as.integer(k_far)
  }
  
  if (!is.null(c) && partial) {
    c <- as.numeric(c)
    fit <- dbrda(X_s_binary ~ Condition(c), distance = "jaccard",
                 binary = binary, add = add)
  } else {
    fit <- dbrda(X_s_binary ~ 1, distance = "jaccard", binary = binary,
                 add = add)
  }
  
  eig <- vegan::eigenvals(fit, model = "unconstrained")
  prop_axis <- eig / sum(eig)
  pct_selected <- 100 * sum(prop_axis[comp])
  sc <- scores(fit, display = "sites", choices = 1:max(comp), scaling = 1,
               model = "unconstrained")
  xy <- as.matrix(sc[, comp, drop = FALSE])
  
  ref  <- as.numeric(xy[1, ])
  d_all <- sqrt(rowSums(sweep(xy, 2, ref, FUN = "-")^2))
  others <- setdiff(seq_len(n), 1)
  
  # Selected taxa mask
  taxa_mask <- X_n_binary[, taxa_id, drop = FALSE]
  
  # Map selection back to original indices
  if (is.character(taxa_id)) {
    subset_ids_original <- match(taxa_id, colnames(X_n_binary))
  } else if (is.numeric(taxa_id)) {
    subset_ids_original <- taxa_id
  } else {
    subset_ids_original <- which(taxa_id)
  }
  
  hi_col_idx_in_subset <- NA_integer_
  if (!is.null(test_taxa)) {
    if (is.character(test_taxa)) {
      if (is.null(colnames(X_n_binary))) stop(
        "test_taxa is a name but X_n_binary lacks colnames.")
      hi_orig <- match(test_taxa, colnames(X_n_binary))
    } else if (is.numeric(test_taxa) && length(test_taxa) == 1L) {
      hi_orig <- test_taxa
    } else stop("test_taxa must be a single column name or numeric index.")
    if (is.na(hi_orig) || !(hi_orig %in% subset_ids_original)) {
      warning("test_taxa is not inside taxa_id; highlighting & test disabled.")
    } else {
      hi_col_idx_in_subset <- match(hi_orig, subset_ids_original)
    }
  }
  test_mask <- if (!is.na(hi_col_idx_in_subset))
    taxa_mask[, hi_col_idx_in_subset] > 0 else rep(FALSE, n)
  test_mask_no_ref <- test_mask & seq_len(n) != 1L   
  
  # Presence masks
  present_mask <- rowSums(taxa_mask) >= 1
  present_idx  <- which(present_mask)
  
  # Nearest and farthest
  nn_idx  <- others[order(d_all[others])[seq_len(k)]]
  far_idx <- others[order(-d_all[others])[seq_len(k_far)]]
  nn_or_far <- union(nn_idx, far_idx)
  
  present_nn  <- intersect(present_idx, nn_idx)
  present_far <- intersect(present_idx, far_idx)
  
  present_nn_black  <- setdiff(present_nn, which(test_mask_no_ref))
  present_far_black <- setdiff(present_far, which(test_mask_no_ref))
  
  col_alpha <- function(col, a) adjustcolor(col, alpha.f = a)
  star_alpha  <- 0
  multi_alpha <- 0
  
  present_counts <- rowSums(taxa_mask > 0)

  plot(xy[,1], xy[,2], type = "n", xlab = "", ylab = "", xaxt = "n", yaxt = "n")
  mtext(paste0("PCo", comp[1]), side = 1, line = 1, cex = 2)
  mtext(paste0("PCo", comp[2]), side = 2, line = 1, cex = 2)
  title(main = title, cex.main = 2, font.main = 2)

  others_only <- setdiff(others, nn_or_far)
  points(xy[others_only, 1], xy[others_only, 2],
         pch = 1, bg = point_col, col = point_col, cex = cex_others, lwd = 2)
  
  if (length(far_idx))
    points(xy[far_idx,1], xy[far_idx,2],
           pch = 6, bg = far_col, col = far_col, cex = cex_far, lwd = 2)
  
  if (length(nn_idx))
    points(xy[nn_idx,1],  xy[nn_idx,2],
           pch = 2, bg = nn_col,  col = nn_col, cex = cex_nn, lwd = 2)
  
  points(xy[1,1], xy[1,2], pch = 22, bg = test_col, col = NA, cex = cex_first)
  
  if (length(present_nn_black)) {
    points(xy[present_nn_black, 1], xy[present_nn_black, 2],
           pch = 20, cex = cex_nn, col = col_alpha("black", star_alpha))
  }
  if (length(present_far_black)) {
    points(xy[present_far_black, 1], xy[present_far_black, 2],
           pch = 20, cex = cex_far, col = col_alpha("black", star_alpha))
  }
  
  cat(paste0("Far: ", sum(present_counts[present_far])),
      paste0("Near: ", sum(present_counts[present_nn])),
      sep = "\n")
  
  multi_symbol <- 20
  multi_max    <- 100
  jitter_r_min <- 1
  jitter_r_max <- 1.8
  
  multi_idx <- intersect(which(present_counts >= 2 & seq_len(n) != 1), nn_or_far)
  if (length(multi_idx)) {
    xr <- diff(range(xy[,1], na.rm = TRUE)); yr <- diff(range(xy[,2], na.rm = TRUE))
    r_x <- 0.015 * xr; r_y <- 0.015 * yr
    
    draw_multi_jitter <- function(i, k){
      k  <- min(k, multi_max)
      x0 <- xy[i,1]; y0 <- xy[i,2]
      ang <- runif(k, 0, 2*pi)
      rad <- runif(k, jitter_r_min, jitter_r_max)
      xs  <- x0 + r_x * rad * cos(ang)
      ys  <- y0 + r_y * rad * sin(ang)
      base_cex <- if (i %in% nn_idx) cex_nn else cex_far
      points(xs, ys, pch = multi_symbol, cex = base_cex,
             col = col_alpha("black", multi_alpha))
    }
    for (i in multi_idx) draw_multi_jitter(i, present_counts[i])
  }
  
  fisher_table <- NULL
  fisher_res   <- NULL
  fisher_label <- NULL
  
  if (!is.na(hi_col_idx_in_subset) && length(nn_idx) > 0 && length(far_idx) > 0) {
    a <- sum(test_mask[nn_idx])                 # NN present
    b <- length(nn_idx) - a                     # NN absent
    c_ <- sum(test_mask[far_idx])               # FAR present
    d <- length(far_idx) - c_                   # FAR absent
    
    fisher_table <- matrix(c(a, b, c_, d), nrow = 2, byrow = TRUE,
                           dimnames = list(Presence = c("Present", "Absent"),
                                           Group    = c("NN", "FAR")))
    fisher_res <- fisher.test(fisher_table, alternative = "greater")
    
  if (any(test_mask_no_ref)) {
    hi_idx <- which(test_mask_no_ref)
    if (length(hi_idx)) {
      cex_vec <- rep(cex_others, length(hi_idx))
      cex_vec[hi_idx %in% nn_idx]  <- cex_nn
      cex_vec[hi_idx %in% far_idx] <- cex_far
      points(xy[hi_idx, 1], xy[hi_idx, 2],
             pch = 21,
             bg  = col_alpha(test_taxa_col, test_taxa_alpha),
             col = NA,
             cex = cex_vec, lwd = 1.4)
    }
  }
  
  leg_txt <- c("Test sample", "k farthest", "k nearest", "Others",
               "Taxon present (nearest)", "Taxon present (farthest)",
               "\u2265 2 taxa present (jittered)",
               "Selected taxon")
  leg_bg  <- c(test_col, far_col, nn_col, point_col, NA, NA, NA,
               col_alpha(test_taxa_col, test_taxa_alpha))
  leg_col <- c("black", "black", "black", "black", nn_col, far_col,
               "black", "black")
  leg_pch <- c(21, 21, 21, 21, NA, NA, NA, 21)
  leg_cex <- c(cex_first, cex_far, cex_nn, cex_others, cex_nn, cex_far, 1.0, 1.2)
  
  keep <- c(TRUE, length(far_idx) > 0, length(nn_idx)  > 0, TRUE,
            length(present_nn_black) > 0, length(present_far_black) > 0,
            length(multi_idx) > 0, any(test_mask_no_ref))
  
  if (legend && any(keep)) {
    L <- legend(legend_pos, legend = leg_txt[keep], pt.bg  = leg_bg[keep],
                col = leg_col[keep], pch = leg_pch[keep],
                pt.cex = leg_cex[keep], cex = 1, bty = "n", x.intersp = 0.8,
                y.intersp = 0.8) rect(L$rect$left, L$rect$top - L$rect$h,
                                      L$rect$left + L$rect$w, L$rect$top,
                                      border = "black", lwd = 1.6, col = NA,
                                      xpd = NA)}
  
  invisible(list(nn_indices = nn_idx, nn_distances = d_all[nn_idx],
                 far_indices = far_idx, far_distances = d_all[far_idx],
                 scores = sc, model = fit, fisher_table = fisher_table,
                 fisher_result = fisher_res))}
