library(readxl)
library(openxlsx)
library(vegan)

#### Data preprocessing - "Effects..."

set.seed(1234)

# Load data
df <- read_excel("C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/OTU_table.xlsx")

# Transform to matrix
df_mat <- as.matrix(df)

# Transpose the matrix
df_mat_transposed <- t(df_mat)

# define the sample size
sample <- min(rowSums(df_mat_transposed))

# rarefraction curve
#cols_rr_plot <- grep("D\\.1$", names(df))
#rr_plot <- df[, cols]
#rarecurve(t(rr_plot), step = 1, sample, xlab = "Sample Size", ylab = "Species")

# apply rarefraction
df_mat_transposed_rar <- rrarefy(df_mat_transposed, sample)

# Transpose back
df_mat_rar <- t(df_mat_transposed_rar)

# Convert to data frame
df_rar <- as.data.frame(df_mat_rar)
df <- df_rar

# normalization
column_sums <- colSums(df)
normalized_df <- sweep(df, 2, column_sums, "/")

cols <- grep("D\\.1$", names(normalized_df))
baseline_absolute <- df[, cols]
write.xlsx(baseline_absolute, "C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/baseline_absolute.xlsx")
baseline <- normalized_df[, cols]

write.xlsx(baseline, "C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/baseline.xlsx")

cols_ABX_2 <- grep("D2$", names(normalized_df))
ABX_cohort_2 <- normalized_df[, cols_ABX_2]

write.xlsx(ABX_cohort_2, "C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/ABX_2.xlsx")

cols_ABX_4 <- grep("D4$", names(normalized_df))
ABX_cohort_4 <- normalized_df[, cols_ABX_4]

write.xlsx(ABX_cohort_4, "C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/ABX_4.xlsx")

cols_ABX_7 <- grep("D7$", names(normalized_df))
ABX_cohort_7 <- normalized_df[, cols_ABX_7]

write.xlsx(ABX_cohort_7, "C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/ABX_7.xlsx")

cols_ABX <- grep("D10$", names(normalized_df))
ABX_cohort <- normalized_df[, cols_ABX]

write.xlsx(ABX_cohort, "C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/ABX.xlsx")

cols_post <- grep("D180$", names(normalized_df))
post_cohort <- normalized_df[, cols_post]

post_cohort$S0027D180 <- NULL

write.xlsx(post_cohort, "C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/post_ABX.xlsx")

cols_post_90 <- grep("D90$", names(normalized_df))
post_cohort_90 <- normalized_df[, cols_post_90]

post_cohort_90$S0027D90 <- NULL

write.xlsx(post_cohort_90, "C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/post_ABX_90.xlsx")

cols_post_21 <- grep("D21$", names(normalized_df))
post_cohort_21 <- normalized_df[, cols_post_21]

write.xlsx(post_cohort_21, "C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/post_ABX_21.xlsx")

cols_post_14 <- grep("D14$", names(normalized_df))
post_cohort_14 <- normalized_df[, cols_post_14]

write.xlsx(post_cohort_14, "C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/post_ABX_14.xlsx")

S0023_baseline <- baseline["S0023D.1"]
S0023_ABX <- normalized_df[ ,c("S0023D4", "S0023D7", "S0023D10")]
S0023_post <- normalized_df[ ,c("S0023D14", "S0023D21", "S0023D90",
                                "S0023D180")]

subjects <- c("S0002", "S0003", "S0004", "S0006", "S0007", "S0008",  "S0011",
              "S0013", "S0014", "S0015", "S0017", "S0018", "S0019",
              "S0020", "S0021", "S0024", "S0029", "S0031", "S0033",
              "S0035", "S0036", "S0037", "S0040", "S0042", "S0045",
              "S0047", "S0049", "S0050") 

abx_suffixes <- c("D2", "D4", "D7", "D10")

post_suffixes <- c("D14", "D21", "D90", "D180")

for (subject in subjects) {
  baseline_col_name <- paste(subject, c("D.1"), sep="")
  assign(paste(subject, "baseline", sep="_"), baseline[, baseline_col_name,
                                                       drop=FALSE])
  
  abx_col_names <- paste(subject, abx_suffixes, sep="")
  assign(paste(subject, "ABX", sep="_"), normalized_df[, abx_col_names,
                                                       drop=FALSE])
  
  post_col_names <- paste(subject, post_suffixes, sep="")
  assign(paste(subject, "post", sep="_"), normalized_df[, post_col_names,
                                                        drop=FALSE])
}

data_list <- list()

for (subject_id in subjects) {
  baseline <- get(paste0(subject_id, "_baseline"))
  ABX <- get(paste0(subject_id, "_ABX"))
  post <- get(paste0(subject_id, "_post"))
  
  combined_data <- rbind(t(as.matrix(baseline)), t(as.matrix(ABX)),
                         t(as.matrix(post)))
  
  print(combined_data)
  
  data_list[[subject_id]] <- combined_data
}

data_subject_23 <- rbind(t(as.matrix(S0023_baseline)), t(as.matrix(S0023_ABX)), t(as.matrix(S0023_post)))
data_list[["S0023"]] <- data_subject_23

write.xlsx(data_list, "C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/data_container.xlsx")
