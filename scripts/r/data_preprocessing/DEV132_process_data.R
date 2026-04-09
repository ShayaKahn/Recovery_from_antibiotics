library(readxl)
library(dplyr)
library(stringr)
library(tidyverse)
library(openxlsx)
library(vegan)

data <- read_excel('C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132/ASV_table.xlsx',
                   sheet = 'feature_table_modified', skip = 1)

data <- data %>% remove_rownames %>% column_to_rownames(var="#OTU ID")

# convert values to numeric
data <- data %>% mutate(across(everything(), ~ as.numeric(as.character(.x))))

# rarefraction curve

cols_rr_plot <- grep("Day1_", names(data))
rr_plot <- data#[, cols_rr_plot]
rarecurve(t(rr_plot), step = 100, max(rowSums(t(rr_plot))),
          xlab = "Sample Size", ylab = "Species", label  = FALSE)

# rarefraction
data_transposed <- t(data)

# based on rarefraction curve and trying to not loose important samples
sample <- 7200

# apply rarefraction
data_transposed_rar <- rrarefy(data_transposed, sample)

data = data.frame(t(data_transposed_rar), check.names = FALSE)

write.xlsx(data, 'C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132/full_ASV_table.xlsx',
           rowNames = TRUE)
write.csv(data, 
          file = 'C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132/full_ASV_table.csv',
          row.names = TRUE)

create_subject_df <- function(data, subjects_id) {
  
  # extract column names
  column_names <- colnames(data)
  
  # extract subjects column names
  data_subject <- data[, startsWith(colnames(data), paste0(subjects_id, "_"))]
  
  day_num <- as.numeric(sub(".*_Day([0-9]+)_.*", "\\1", colnames(data_subject)))
  
  data_subject <- data_subject[ , order(day_num)]
  return(data_subject)
}

create_time_point_df <- function(data, day) {
  
  # extract column names
  column_names <- colnames(data)
  
  # extract time point column names
  data_timepoint <- data[, grep(paste0('Day', day, '_'), column_names)]
  
  # sort by subjects
  column_names_timepoint <- colnames(data_timepoint)
  num_part <- as.numeric(sub("_.*", "", column_names_timepoint))
  sorted_index <- order(num_part)
  data_timepoint <- data_timepoint[, sorted_index]
  
  return(data_timepoint)
}

richness <- function(data) {
  data_binary <- data
  
  data_binary[data_binary != 0] <- 1
  
  return(colSums(data_binary))
}

create_time_point_df_from_subject_dfs <- function(data_list, col_names){
  return(data.frame(setNames(lapply(seq_along(data_list),
                                    function(i) data_list[[i]][[col_names[i]]]),
                             col_names),
                    row.names = rownames(data_list[[1]]),
                    check.names = FALSE)
  )
}

# CZA subjects
data_subject_10_CZA <- create_subject_df(data, 10)
data_subject_18_CZA <- create_subject_df(data, 18)
data_subject_32_CZA <- create_subject_df(data, 32)
data_subject_40_CZA <- create_subject_df(data, 40)
data_subject_57_CZA <- create_subject_df(data, 57)
data_subject_65_CZA <- create_subject_df(data, 65)
data_subject_77_CZA <- create_subject_df(data, 77)
data_subject_85_CZA <- create_subject_df(data, 85)
data_subject_108_CZA <- create_subject_df(data, 108)
data_subject_123_CZA <- create_subject_df(data, 123)
data_subject_142_CZA <- create_subject_df(data, 142)
data_subject_5117_CZA <- create_subject_df(data, 5117)

# PTZ subjects
data_subject_9_PTZ <- create_subject_df(data, 9)
data_subject_15_PTZ <- create_subject_df(data, 15)
data_subject_34_PTZ <- create_subject_df(data, 34)
data_subject_41_PTZ <- create_subject_df(data, 41)
data_subject_51_PTZ <- create_subject_df(data, 51)
data_subject_72_PTZ <- create_subject_df(data, 72)
data_subject_75_PTZ <- create_subject_df(data, 75)
data_subject_93_PTZ <- create_subject_df(data, 93)
data_subject_105_PTZ <- create_subject_df(data, 105)
data_subject_112_PTZ <- create_subject_df(data, 112)
data_subject_122_PTZ <- create_subject_df(data, 122)

# CRO subjects
data_subject_3_CRO <- create_subject_df(data, 3)
data_subject_23_CRO <- create_subject_df(data, 23)
data_subject_25_CRO <- create_subject_df(data, 25)
data_subject_38_CRO <- create_subject_df(data, 38)
data_subject_52_CRO <- create_subject_df(data, 52)
data_subject_67_CRO <- create_subject_df(data, 67)
data_subject_78_CRO <- create_subject_df(data, 78)
data_subject_89_CRO <- create_subject_df(data, 89)
data_subject_107_CRO <- create_subject_df(data, 107)
data_subject_118_CRO <- create_subject_df(data, 118)
data_subject_125_CRO <- create_subject_df(data, 125)
data_subject_140_CRO <- create_subject_df(data, 140)

data_baseline_full <- create_time_point_df(data, 1)
data_day_3_full <- create_time_point_df(data, 3)
data_ABX_full <- create_time_point_df(data, 6)
data_day_9_full <- create_time_point_df(data, 9)
data_day_12_full <- create_time_point_df(data, 12)
data_day_16_full <- create_time_point_df(data, 16)
data_day_25_full <- create_time_point_df(data, 25)
data_post_ABX_full <- create_time_point_df(data, 37)

data_list_total <- list(data_subject_10_CZA, data_subject_18_CZA,
                        data_subject_40_CZA, data_subject_57_CZA,
                        data_subject_65_CZA, data_subject_77_CZA,
                        data_subject_85_CZA, data_subject_108_CZA,
                        data_subject_123_CZA, data_subject_5117_CZA,
                        data_subject_15_PTZ, data_subject_51_PTZ,
                        data_subject_72_PTZ, data_subject_93_PTZ,
                        data_subject_105_PTZ, data_subject_112_PTZ,
                        data_subject_122_PTZ, data_subject_25_CRO,
                        data_subject_38_CRO, data_subject_52_CRO,
                        data_subject_67_CRO, data_subject_78_CRO,
                        data_subject_89_CRO, data_subject_107_CRO,
                        data_subject_118_CRO, data_subject_125_CRO,
                        data_subject_140_CRO)

subjects_list <- list("10", "18", "40", "57", "65", "77", "85", "108",
                      "123", "5117", "15", "51", "72", "93", "105",
                      "112", "122", "25", "38", "52", "67", "78", "89",
                      "107", "118", "125", "140")

group_list <- list("CZA", "CZA", "CZA", "CZA", "CZA", "CZA", "CZA", "CZA",
                   "CZA", "CZA", "PTZ", "PTZ", "PTZ", "PTZ", "PTZ", "PTZ",
                   "PTZ", "CRO", "CRO", "CRO", "CRO", "CRO", "CRO", "CRO",
                   "CRO", "CRO", "CRO")

create_total_time_point_cohort <- function(data_list_total, subjects_list,
                                           group_list, day){
  col_names_total <- list()
  
  for (i in 1:length(data_list_total)) {
    col_names_total[[i]] <- paste0(subjects_list[[i]], '_Day', day, '_',
                                        group_list[[i]])
  }
  
  return(unlist(col_names_total))
}
  
col_names_total_base <- create_total_time_point_cohort(data_list_total,
                                                       subjects_list,
                                                       group_list,
                                                       1)
baseline_subjects <- create_time_point_df_from_subject_dfs(data_list_total,
                                                           col_names_total_base)

col_names_total_3 <- create_total_time_point_cohort(data_list_total,
                                                    subjects_list,
                                                    group_list,
                                                    3)
day3_subjects <- create_time_point_df_from_subject_dfs(data_list_total,
                                                       col_names_total_3)

col_names_total_ABX <- create_total_time_point_cohort(data_list_total,
                                                      subjects_list,
                                                      group_list,
                                                      6)
ABX_subjects <- create_time_point_df_from_subject_dfs(data_list_total,
                                                      col_names_total_ABX)

col_names_total_9 <- create_total_time_point_cohort(data_list_total,
                                                    subjects_list,
                                                    group_list,
                                                    9)
day9_subjects <- create_time_point_df_from_subject_dfs(data_list_total,
                                                       col_names_total_9)

col_names_total_12 <- create_total_time_point_cohort(data_list_total,
                                                     subjects_list,
                                                     group_list,
                                                     12)
day12_subjects <- create_time_point_df_from_subject_dfs(data_list_total,
                                                        col_names_total_12)

col_names_total_16 <- create_total_time_point_cohort(data_list_total,
                                                     subjects_list,
                                                     group_list,
                                                     16) 
day16_subjects <- create_time_point_df_from_subject_dfs(data_list_total,
                                                        col_names_total_16)

col_names_total_25 <- create_total_time_point_cohort(data_list_total,
                                                     subjects_list,
                                                     group_list,
                                                     25) 
day25_subjects <- create_time_point_df_from_subject_dfs(data_list_total,
                                                        col_names_total_25)

col_names_total_post_ABX <- create_total_time_point_cohort(data_list_total,
                                                           subjects_list,
                                                           group_list,
                                                           37)
post_ABX_subjects <- create_time_point_df_from_subject_dfs(data_list_total,
                                                           col_names_total_post_ABX)
#### save data ####

create_full_path <- function(dir_path, file_name ){
  
  return(file.path(dir_path, file_name))
}

dir_path <- 'C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132'

# special subjects
write.xlsx(data_subject_32_CZA, create_full_path(dir_path, "CZA_32.xlsx"),
           rowNames = TRUE)
write.csv(data_subject_32_CZA, 
          file = create_full_path(dir_path, "CZA_32.csv"),
          row.names = TRUE)

write.xlsx(data_subject_142_CZA, create_full_path(dir_path, "CZA_142.xlsx"),
           rowNames = TRUE)
write.csv(data_subject_142_CZA, 
          file = create_full_path(dir_path, "CZA_142.csv"),
          row.names = TRUE)

write.xlsx(data_subject_9_PTZ, create_full_path(dir_path, "PTZ_9.xlsx"),
           rowNames = TRUE)
write.csv(data_subject_9_PTZ, 
          file = create_full_path(dir_path, "PTZ_9.csv"),
          row.names = TRUE)

write.xlsx(data_subject_34_PTZ, create_full_path(dir_path, "PTZ_34.xlsx"),
           rowNames = TRUE)
write.csv(data_subject_34_PTZ, 
          file = create_full_path(dir_path, "PTZ_34.csv"),
          row.names = TRUE)

write.xlsx(data_subject_41_PTZ, create_full_path(dir_path, "PTZ_41.xlsx"),
           rowNames = TRUE)
write.csv(data_subject_41_PTZ, 
          file = create_full_path(dir_path, "PTZ_41.csv"),
          row.names = TRUE)


write.xlsx(data_subject_75_PTZ, create_full_path(dir_path, "PTZ_75.xlsx"),
           rowNames = TRUE)
write.csv(data_subject_75_PTZ, 
          file = create_full_path(dir_path, "PTZ_75.csv"),
          row.names = TRUE)

write.xlsx(data_subject_3_CRO, create_full_path(dir_path, "CRO_3.xlsx"),
           rowNames = TRUE)
write.csv(data_subject_3_CRO, 
          file = create_full_path(dir_path, "CRO_3.csv"),
          row.names = TRUE)

write.xlsx(data_subject_23_CRO, create_full_path(dir_path, "CRO_23.xlsx"),
           rowNames = TRUE)
write.csv(data_subject_23_CRO, 
          file = create_full_path(dir_path, "CRO_23.csv"),
          row.names = TRUE)

# subjects
write.xlsx(baseline_subjects, create_full_path(dir_path,
                                               "baseline_subjects.xlsx"),
           rowNames = TRUE)
write.csv(baseline_subjects, 
          file = create_full_path(dir_path,
                                  "baseline_subjects.csv"),
          row.names = TRUE)

write.xlsx(day3_subjects, create_full_path(dir_path, "day3_subjects.xlsx"),
           rowNames = TRUE)
write.csv(day3_subjects, 
          file = create_full_path(dir_path, "day3_subjects.csv"),
          row.names = TRUE)

write.xlsx(ABX_subjects, create_full_path(dir_path, "ABX_subjects.xlsx"),
           rowNames = TRUE)
write.csv(ABX_subjects, 
          file = create_full_path(dir_path, "ABX_subjects.csv"),
          row.names = TRUE)

write.xlsx(day9_subjects, create_full_path(dir_path, "day9_subjects.xlsx"),
           rowNames = TRUE)
write.csv(day9_subjects, 
          file = create_full_path(dir_path, "day9_subjects.csv"),
          row.names = TRUE)

write.xlsx(day12_subjects, create_full_path(dir_path, "day12_subjects.xlsx"),
           rowNames = TRUE)
write.csv(day12_subjects, 
          file = create_full_path(dir_path, "day12_subjects.csv"),
          row.names = TRUE)

write.xlsx(day16_subjects, create_full_path(dir_path, "day16_subjects.xlsx"),
           rowNames = TRUE)
write.csv(day16_subjects, 
          file = create_full_path(dir_path, "day16_subjects.csv"),
          row.names = TRUE)

write.xlsx(day25_subjects, create_full_path(dir_path, "day25_subjects.xlsx"),
           rowNames = TRUE)
write.csv(day25_subjects, 
          file = create_full_path(dir_path, "day25_subjects.csv"),
          row.names = TRUE)

write.xlsx(post_ABX_subjects, create_full_path(dir_path,
                                               "post_ABX_subjects.xlsx"),
           rowNames = TRUE)
write.csv(post_ABX_subjects, 
          file = create_full_path(dir_path, "post_ABX_subjects.csv"),
          row.names = TRUE)

write.xlsx(data_baseline_full, create_full_path(dir_path, "baseline_full.xlsx"),
           rowNames = TRUE)
write.csv(data_baseline_full, 
          file = create_full_path(dir_path, "baseline_full.csv"),
          row.names = TRUE)

write.xlsx(data_post_ABX_full, create_full_path(dir_path, "post_ABX_full.xlsx"),
           rowNames = TRUE)
write.csv(data_post_ABX_full, 
          file = create_full_path(dir_path, "post_ABX_full.csv"),
          row.names = TRUE)


