library(ppcor)
library(ggplot2)
library(gridExtra)
source("C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/scripts/r/visualization_utils/utils.R")

#### Volcano plots ####

# Load data
new_data <- as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/similarity_matrix_new.csv", header = FALSE))
others_data <- as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/similarity_matrix_others.csv", header = FALSE))
sizes_data <- as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/sizes_matrix.csv", header = FALSE))

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
  
  p <- LR_smooth_plot(data_resid, expression(s^{(w)} ~ "|" ~ c),
                      expression(s^{(n)} ~ "|" ~ c), size=1, size_line=1,
                      linewidth_axis=1, size_title=1, size_text=1,
                      size_ticks = 1)
  
  plot_list[[length(plot_list) + 1]] <- ggplotGrob(p)
  
  partial_corr_r <- pcor.test(x=data$x, y=data$y, z=data$z, method="pearson")
  partial_corr_s <- pcor.test(x=data$x, y=data$y, z=data$z, method="spearman")
  
  r_vals[[length(r_vals) + 1]] <- partial_corr_r$estimate
  s_vals[[length(s_vals) + 1]] <- partial_corr_s$estimate
  
  pr_vals[[length(pr_vals) + 1]] <- partial_corr_r$p.value
  ps_vals[[length(ps_vals) + 1]] <- partial_corr_s$p.value
}

r_vals_vec <- unlist(r_vals)
s_vals_vec <- unlist(s_vals)
pr_vals_vec <- unlist(pr_vals)
ps_vals_vec <- unlist(ps_vals)

write.csv(r_vals_vec, file = "C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/sCorr_version_2.csv", row.names = FALSE)

# Plots
plot_volcano(r_vals_vec, pr_vals_vec, x_title="sCorr", alpha = 0.05,
             method="BH", p_max=1e-4, poin_size=8, title_size=30,
             text_size=20, legend_text_size=20)
plot_volcano(s_vals_vec, ps_vals_vec, x_title="sCorr", alpha = 0.05,
             method="BH", p_max=1e-4)
