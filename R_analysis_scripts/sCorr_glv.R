library(ppcor)
library(ggplot2)
library(gridExtra)
source("C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/R_analysis_scripts/utils.R")

#### sCorr version 2 plots for the GLV model ####

# Load data
sims_new <- drop(as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/sims_new_glv.csv", header = FALSE)))
sims_survived <- drop(as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/sims_survived_glv.csv", header = FALSE)))
sizes <- drop(as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/sizes_glv.csv", header = FALSE)))

sims_new_off <- drop(as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/sims_new_glv_off.csv", header = FALSE)))
sims_survived_off <- drop(as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/sims_survived_glv_off.csv", header = FALSE)))
sizes_off <- drop(as.matrix(read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/sizes_glv_off.csv", header = FALSE)))

# Residual correlation plot
data <- data.frame(x = sims_survived, y = sims_new, z = sizes)

x_resid<-resid(lm(x~z, data))
y_resid<-resid(lm(y~z, data))

data_resid <- data.frame(x = x_resid, y = y_resid)

p <- LR_plot_outline(data_resid, expression(s^{(phi)} ~ "|" ~ c),
                     expression(s^{(nu)} ~ "|" ~ c), linewidth_axis = 1,
                     size_title=30, size_text=20, line_color = '#DC143C',
                     color = '#006400')

grid.arrange(p, ncol = 1)

# Residual correlation plot (interactions are off)
data_off <- data.frame(x = sims_survived_off, y = sims_new_off, z = sizes_off)

x_resid_off<-resid(lm(x~z, data_off))
y_resid_off<-resid(lm(y~z, data_off))

data_resid_off <- data.frame(x = x_resid_off, y = y_resid_off)

p_off <- LR_plot_outline(data_resid_off, expression(s^{(phi)} ~ "|" ~ c),
                         expression(s^{(nu)} ~ "|" ~ c), linewidth_axis = 1,
                         size_title=30, size_text=20, 
                         show_lm = TRUE, line_color = '#DC143C',
                         color = '#006400')

grid.arrange(p_off, ncol = 1)

# Calculate statistics
partial_corr_r <- pcor.test(x=data$x, y=data$y, z=data$z, method="pearson")
partial_corr_s <- pcor.test(x=data$x, y=data$y, z=data$z, method="spearman")

r <- partial_corr_r$estimate
s <- partial_corr_s$estimate

pr <- partial_corr_r$p.value
ps <- partial_corr_s$p.value

partial_corr_r_off <- pcor.test(x=data_off$x, y=data_off$y, z=data_off$z, method="pearson")
partial_corr_s_off <- pcor.test(x=data_off$x, y=data_off$y, z=data_off$z, method="spearman")

r_off <- partial_corr_r_off$estimate
s_off <- partial_corr_s_off$estimate

pr_off <- partial_corr_r_off$p.value
ps_off <- partial_corr_s_off$p.value
