library(ppcor)
library(ggplot2)
library(gridExtra)
library(vegan)
source("C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/scripts/r/visualization_utils/utils.R")

#### Volcano plots ####

# Load data
new_data <- as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/similarity_matrix_new_5.csv", header = FALSE))
others_data <- as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/similarity_matrix_others_5.csv", header = FALSE))
sizes_data <- as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/sizes_matrix_5.csv", header = FALSE))

# Calculate partial correlations and generate scatterplots
r_vals <- list()
pr_vals <- list()

s_vals <- list()
ps_vals <- list()

plot_list = list()

for (i in 1:nrow(new_data)) {
  data <- data.frame(x = others_data[i, ], y = new_data[i, ],
                     z = sizes_data[i, ])
  
  x_resid<-resid(lm(x~z, data))
  y_resid<-resid(lm(y~z, data))
  
  data_resid <- data.frame(x = x_resid, y = y_resid)
  
  p <- LR_smooth_plot(data=data_resid,
                      x_title="Similarity in the Survived subspace (Residuals)",
                      y_title="Similarity in the New subspace (Residuals)",
                      size=4, size_line=3, linewidth_axis=1, size_title=20,
                      size_text=15, size_ticks=20, span=0.95,
                      line_color="#2563EB", color = "#BB6508")
  
  plot_list[[length(plot_list) + 1]] <- ggplotGrob(p)
  
  partial_corr_r <- pcor.test(x=data$x, y=data$y, z=data$z, method="pearson")
  partial_corr_s <- pcor.test(x=data$x, y=data$y, z=data$z, method="spearman")
  
  r_vals[[length(r_vals) + 1]] <- partial_corr_r$estimate
  s_vals[[length(s_vals) + 1]] <- partial_corr_s$estimate
  
  pr_vals[[length(pr_vals) + 1]] <- partial_corr_r$p.value
  ps_vals[[length(ps_vals) + 1]] <- partial_corr_s$p.value
}

grid.arrange(grobs = plot_list[8], ncol = 1)

r_vals_vec <- unlist(r_vals)
s_vals_vec <- unlist(s_vals)
pr_vals_vec <- unlist(pr_vals)
ps_vals_vec <- unlist(ps_vals)

write.csv(r_vals_vec, file = "C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/sCorr_version_2_dataset_5.csv", row.names = FALSE)
write.csv(pr_vals_vec, file = "C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/sCorr_version_2_pvals_dataset_5.csv", row.names = FALSE)

# Plots
plot_volcano(r_vals_vec, pr_vals_vec, x_title="sCorr", alpha = 0.05,
             method="BH", p_max=1e-4, poin_size=8, title_size=30, text_size=20,
             legend_text_size=20, adjust = TRUE)
plot_volcano(s_vals_vec, ps_vals_vec, x_title="sCorr", alpha = 0.05,
             method="BH", p_max=1e-4)

# PCoA
feature_matrix_bin <- as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132/feature_matrix_bin.csv", header = FALSE))
new_space <- as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132/new_space.csv", header = FALSE)) + 1
survived_space <- as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132/survived_space.csv", header = FALSE)) + 1

feature_matrix_bin_new <- feature_matrix_bin[ ,new_space]
feature_matrix_bin_survived <- feature_matrix_bin[ ,survived_space]

c <- rowSums(feature_matrix_bin)

s <- colSums(feature_matrix_bin_new)
idx_most_less_prev <- order(s)
values_most_less_prev <- as.vector(s[idx_most_less_prev])
mask <- (values_most_less_prev > 1) & (values_most_less_prev < 14)
chosen_idx <- idx_most_less_prev[mask]

par(mfrow = c(1, 1))
p <- plot_knn_richness(feature_matrix_bin_survived, feature_matrix_bin_new,
                       taxa_id = chosen_idx, k = 58, k_far = NULL, c = c,
                       partial = TRUE, binary = TRUE, add = TRUE,
                       legend = FALSE, test_taxa = chosen_idx[8],
                       test_taxa_alpha = 0.9)
