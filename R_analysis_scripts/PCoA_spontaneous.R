library(readxl)
library(vegan)
library(car)
library(ggplot2)
library(gridExtra)
source("C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/R_analysis_scripts/utils.R")

#### PCoA visualization ####

# Load data
data <- read_excel("C:/Users/USER/OneDrive/Desktop/Antibiotics/Eran Elinav/Data/spontaneous_combined.xlsx", col_names = FALSE)

data_abx_follow_up <- read.csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/Eran Elinav/Data/total_abx_follow_up.csv", header = FALSE)

data <- apply(data, 2, as.numeric)
data_abx_follow_up <- apply(data_abx_follow_up, 2, as.numeric)

baseline <- data[1:130,]

spo_abx_1 = data_abx_follow_up[1:5, ]
spo_abx_2 = data_abx_follow_up[19:25, ]
spo_abx_3 = data_abx_follow_up[42:48, ]
spo_abx_4 = data_abx_follow_up[64:70, ]
spo_abx_5 = data_abx_follow_up[83:89, ]
spo_abx_6 = data_abx_follow_up[105:110, ]
spo_abx_7 = data_abx_follow_up[122:128, ]
  
spo_follow_up_1 = data_abx_follow_up[6:18, ]
spo_follow_up_2 = data_abx_follow_up[26:41, ]
spo_follow_up_3 = data_abx_follow_up[49:63, ]
spo_follow_up_4 = data_abx_follow_up[71:82, ]
spo_follow_up_5 = data_abx_follow_up[90:104, ]
spo_follow_up_6 = data_abx_follow_up[111:121, ]
spo_follow_up_7 = data_abx_follow_up[129:139, ]

data_spo_1 <- rbind(baseline, spo_abx_1, spo_follow_up_1)
data_spo_2 <- rbind(baseline, spo_abx_2, spo_follow_up_2)
data_spo_3 <- rbind(baseline, spo_abx_3, spo_follow_up_3)
data_spo_4 <- rbind(baseline, spo_abx_4, spo_follow_up_4)
data_spo_5 <- rbind(baseline, spo_abx_5, spo_follow_up_5)
data_spo_6 <- rbind(baseline, spo_abx_6, spo_follow_up_6)
data_spo_7 <- rbind(baseline, spo_abx_7, spo_follow_up_7)

coordinates_spo_1 = PCoA(data_spo_1)
coordinates_spo_2 = PCoA(data_spo_2)
coordinates_spo_3 = PCoA(data_spo_3)
coordinates_spo_4 = PCoA(data_spo_4)
coordinates_spo_5 = PCoA(data_spo_5)
coordinates_spo_6 = PCoA(data_spo_6)
coordinates_spo_7 = PCoA(data_spo_7)

colors_spo_1 <- rep("grey", dim(coordinates_spo_1)[1])

colors_spo_1[87:91] <- "#355C4C"
colors_spo_1[131:135] <- "red"        
colors_spo_1[136:148] <- "dodgerblue3"

colors_spo_2 <- rep("#ABABAB", dim(coordinates_spo_2)[1])
shapes_spo_2 <- rep(16, dim(coordinates_spo_2)[1])

colors_spo_2[99:105] <- "pink"
colors_spo_2[119:123] <- "brown"
colors_spo_2[124:130] <- "black"
colors_spo_2[92:98] <- "#51A246"
colors_spo_2[131:137] <- "#E33A3A"        
colors_spo_2[138:148] <- "#1A71B8"
colors_spo_2[149:153] <- "#1A71B8"

shapes_spo_2[99:105] <- 17
shapes_spo_2[119:123] <- 17
shapes_spo_2[124:130] <- 17
shapes_spo_2[92:98] <- 7
shapes_spo_2[131:137] <- 8       
shapes_spo_2[138:148] <- 15
shapes_spo_2[149:153] <- 15

colors_spo_3 <- rep("#ABABAB", dim(coordinates_spo_3)[1])

colors_spo_3[99:105] <- "#355C4C"
colors_spo_3[131:137] <- "red"        
colors_spo_3[138:147] <- "dodgerblue3"
colors_spo_3[148:152] <- "dodgerblue3"

colors_spo_4 <- rep("#ABABAB", dim(coordinates_spo_4)[1])

colors_spo_4[106:111] <- "#355C4C"
colors_spo_4[131:137] <- "red"        
colors_spo_4[138:144] <- "dodgerblue3"
colors_spo_4[145:149] <- "dodgerblue3"

colors_spo_5 <- rep("#ABABAB", dim(coordinates_spo_5)[1])
shapes_spo_5 <- rep(16, dim(coordinates_spo_5)[1])

colors_spo_5[99:105] <- "pink"
colors_spo_5[119:123] <- "brown"
colors_spo_5[124:130] <- "black"
colors_spo_5[112:118] <- "#51A246"
colors_spo_5[131:137] <- "#E33A3A"        
colors_spo_5[138:147] <- "#1A71B8"
colors_spo_5[148:152] <- "#1A71B8"

shapes_spo_5[99:105] <- 17
shapes_spo_5[119:123] <- 17
shapes_spo_5[124:130] <- 17
shapes_spo_5[112:118] <- 7
shapes_spo_5[131:137] <- 8       
shapes_spo_5[138:147] <- 15
shapes_spo_5[148:152] <- 15

colors_spo_6 <- rep("grey", dim(coordinates_spo_6)[1])

colors_spo_6[119:123] <- "#355C4C"
colors_spo_6[131:136] <- "red"        
colors_spo_6[137:142] <- "dodgerblue3"
colors_spo_6[143:147] <- "dodgerblue3"

colors_spo_7 <- rep("grey", dim(coordinates_spo_7)[1])

colors_spo_7[124:130] <- "#355C4C"
colors_spo_7[131:137] <- "red"        
colors_spo_7[138:143] <- "dodgerblue3"
colors_spo_7[144:148] <- "dodgerblue3"

p2 = create_custom_plot(coordinates_spo_2, colors_spo_2, shapes_spo_2,
                        23, '', level = 0.9)

coordinates_spo_5[,2] = -coordinates_spo_5[,2]
p5 = create_custom_plot(coordinates_spo_5, colors_spo_5, shapes_spo_5,
                        22, '', level = 0.9)

grid.arrange(p5, ncol = 1, nrow = 1)

# Spo 5

spo_5_all_samples <- rbind(baseline[112:118,], spo_abx_5, spo_follow_up_5)
spo_5_all_samples[spo_5_all_samples != 0] <- 1

richness_spo_5 <- rowSums(spo_5_all_samples)

times_spo_5 <- c(-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,
                 0,1,2,3,4,5,6,7,14,21,28,42,56,120,150,180)

x <- times_spo_5
y <- richness_spo_5

seg1 <- x >= -13 & x <=   7
seg2 <- x >=  14 & x <=  56
seg3 <- x >= 120 & x <= 180

n1 <- sum(seg1);  n2 <- sum(seg2);  n3 <- sum(seg3)

gap <- 4                              

x_disp <- numeric(length(x))
x_disp[seg1] <- seq_len(n1)
x_disp[seg2] <- n1 + gap + seq_len(n2)
x_disp[seg3] <- n1 + gap + n2 + gap + seq_len(n3)

x_line <- c(x_disp[seg1], NA, x_disp[seg2], NA, x_disp[seg3])
y_line <- c(y[seg1], NA, y[seg2], NA, y[seg3])

par(mar = c(5.1, 6.2, 4.1, 2.1))                       
par(cex.lab = 2.5, cex.axis = 1.6, tcl = -0.3, las = 2, bty = "l", lwd = 2)                 

plot(x_line, y_line, type = "n", xaxt = "n", xlab = "Day", ylab = "Richness")

shade_idx   <- x >= -6 & x <= 0
usr <- par("usr")
rect(min(x_disp[shade_idx]), usr[3], max(x_disp[shade_idx]), usr[4],
     col = adjustcolor("#E33A3A", 0.25), border = NA)

y0 <- par("usr")[3]

gap1_left  <- n1 + 0.5
gap1_right <- n1 + gap + 0.5
gap2_left  <- n1 + gap + n2 + 0.5
gap2_right <- n1 + gap + n2 + gap + 0.5

segments(gap1_left,  y0, gap1_right, y0, col = "white", lwd = 6, xpd = NA)
segments(gap2_left,  y0, gap2_right, y0, col = "white", lwd = 6, xpd = NA)

lines(x_line, y_line, lwd = 4, col = "#C05A00")

lines(c(tail(x_disp[seg1], 1), head(x_disp[seg2], 1)), c(tail(y[seg1], 1),
                                                         head(y[seg2], 1)),         
      lwd = 4, col = "#C05A00", lty = c(3, 1))

lines(c(tail(x_disp[seg2], 1), head(x_disp[seg3], 1)), c(tail(y[seg2], 1),
                                                         head(y[seg3], 1)),
      lwd = 4, col = "#C05A00", lty = c(3, 1))

axis(1, at = x_disp[seg1], labels = x[seg1], lwd = 2, lwd.ticks = 1.5)
axis(1, at = x_disp[seg2], labels = x[seg2], lwd = 2, lwd.ticks = 1.5)
axis(1, at = x_disp[seg3], labels = x[seg3], lwd = 2, lwd.ticks = 1.5)

# Spo 2

spo_2_all_samples <- rbind(baseline[92:98,], spo_abx_2, spo_follow_up_2)
spo_2_all_samples[spo_2_all_samples != 0] <- 1

richness_spo_2 <- rowSums(spo_2_all_samples)

times_spo_2 <- c(-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,
                 0,1,2,3,4,5,6,7,14,21,28,42,56,90,120,150,180)

x <- times_spo_2
y <- richness_spo_2

seg1 <- x >= -13 & x <=   7
seg2 <- x >=  14 & x <=  56
seg3 <- x >= 90 & x <= 180

n1 <- sum(seg1);  n2 <- sum(seg2);  n3 <- sum(seg3)

gap <- 4                              

x_disp <- numeric(length(x))
x_disp[seg1] <- seq_len(n1)
x_disp[seg2] <- n1 + gap + seq_len(n2)
x_disp[seg3] <- n1 + gap + n2 + gap + seq_len(n3)

x_line <- c(x_disp[seg1], NA, x_disp[seg2], NA, x_disp[seg3])
y_line <- c(y[seg1], NA, y[seg2], NA, y[seg3])

par(mar = c(5.1, 6.2, 4.1, 2.1))                       
par(cex.lab = 2.5, cex.axis = 1.6, tcl = -0.3, las = 2, bty = "l", lwd = 2)                 

plot(x_line, y_line, type = "n", xaxt = "n", xlab = "Day", ylab = "Richness")

shade_idx   <- x >= -6 & x <= 0
usr <- par("usr")
rect(min(x_disp[shade_idx]), usr[3], max(x_disp[shade_idx]), usr[4],
     col = adjustcolor("#E33A3A", 0.25), border = NA)

y0 <- par("usr")[3]

gap1_left  <- n1 + 0.5
gap1_right <- n1 + gap + 0.5
gap2_left  <- n1 + gap + n2 + 0.5
gap2_right <- n1 + gap + n2 + gap + 0.5

segments(gap1_left,  y0, gap1_right, y0, col = "white", lwd = 6, xpd = NA)
segments(gap2_left,  y0, gap2_right, y0, col = "white", lwd = 6, xpd = NA)

lines(x_line, y_line, lwd = 4, col = "#C05A00")

lines(c(tail(x_disp[seg1], 1), head(x_disp[seg2], 1)),   
      c(tail(y[seg1], 1),     head(y[seg2], 1)),         
      lwd = 4, col = "#C05A00", lty = c(3, 1))

lines(c(tail(x_disp[seg2], 1), head(x_disp[seg3], 1)), c(tail(y[seg2], 1),
                                                         head(y[seg3], 1)),
      lwd = 4, col = "#C05A00", lty = c(3, 1))

axis(1, at = x_disp[seg1], labels = x[seg1], lwd = 2, lwd.ticks = 1.5)
axis(1, at = x_disp[seg2], labels = x[seg2], lwd = 2, lwd.ticks = 1.5)
axis(1, at = x_disp[seg3], labels = x[seg3], lwd = 2, lwd.ticks = 1.5)

#### Beta diversity plots ####

# spo 2

method <- "bray"
braycurtis_dmat_spo_2 <- as.matrix(vegdist(spo_2_all_samples,
                                           method = method))[, 1:7]

braycurtis_dmat_spo_2[braycurtis_dmat_spo_2 == 0] <- NA      
mean_braycurtis_to_base_spo_2 <- rowMeans(braycurtis_dmat_spo_2, na.rm = TRUE)

x <- times_spo_2
y <- mean_braycurtis_to_base_spo_2

seg1 <- x >= -13 & x <=   7
seg2 <- x >=  14 & x <=  56
seg3 <- x >= 120 & x <= 180

n1 <- sum(seg1);  n2 <- sum(seg2);  n3 <- sum(seg3)

gap <- 4                              

x_disp <- numeric(length(x))
x_disp[seg1] <- seq_len(n1)
x_disp[seg2] <- n1 + gap + seq_len(n2)
x_disp[seg3] <- n1 + gap + n2 + gap + seq_len(n3)

x_line <- c(x_disp[seg1], NA, x_disp[seg2], NA, x_disp[seg3])
y_line <- c(y[seg1], NA, y[seg2], NA, y[seg3])

par(mar = c(5.1, 6.2, 4.1, 2.1))                       
par(cex.lab = 2.5, cex.axis = 1.6, tcl = -0.3, las = 2, bty = "l", lwd = 2)                 

plot(x_line, y_line, type = "n", xaxt = "n", xlab = "Day", ylab = "Bray-Curtis")

shade_idx <- x >= -6 & x <= 0
usr <- par("usr")
rect(min(x_disp[shade_idx]), usr[3], max(x_disp[shade_idx]), usr[4],
     col = adjustcolor("#E33A3A", 0.25), border = NA)

y0 <- par("usr")[3]

gap1_left  <- n1 + 0.5
gap1_right <- n1 + gap + 0.5
gap2_left  <- n1 + gap + n2 + 0.5
gap2_right <- n1 + gap + n2 + gap + 0.5

segments(gap1_left,  y0, gap1_right, y0, col = "white", lwd = 6, xpd = NA)
segments(gap2_left,  y0, gap2_right, y0, col = "white", lwd = 6, xpd = NA)

lines(x_line, y_line, lwd = 4, col = "#C05A00")

lines(c(tail(x_disp[seg1], 1), head(x_disp[seg2], 1)), c(tail(y[seg1], 1),
                                                         head(y[seg2], 1)),
      lwd = 4, col = "#C05A00", lty = c(3, 1))

lines(c(tail(x_disp[seg2], 1), head(x_disp[seg3], 1)),
      c(tail(y[seg2], 1), head(y[seg3], 1)),
      lwd = 4, col = "#C05A00", lty = c(3, 1))

axis(1, at = x_disp[seg1], labels = x[seg1], lwd = 2, lwd.ticks = 1.5)
axis(1, at = x_disp[seg2], labels = x[seg2], lwd = 2, lwd.ticks = 1.5)
axis(1, at = x_disp[seg3], labels = x[seg3], lwd = 2, lwd.ticks = 1.5)

# spo 5

method <- "bray"

braycurtis_dmat_spo_5 <- as.matrix(vegdist(spo_5_all_samples,
                                           method = method))[, 1:7]

braycurtis_dmat_spo_5[braycurtis_dmat_spo_5 == 0] <- NA      
mean_braycurtis_to_base_spo_5 <- rowMeans(braycurtis_dmat_spo_5, na.rm = TRUE)

x <- times_spo_5
y <- mean_braycurtis_to_base_spo_5

seg1 <- x >= -13 & x <=   7
seg2 <- x >=  14 & x <=  56
seg3 <- x >= 120 & x <= 180

n1 <- sum(seg1);  n2 <- sum(seg2);  n3 <- sum(seg3)

gap <- 4                              

x_disp <- numeric(length(x))
x_disp[seg1] <- seq_len(n1)
x_disp[seg2] <- n1 + gap + seq_len(n2)
x_disp[seg3] <- n1 + gap + n2 + gap + seq_len(n3)

x_line <- c(x_disp[seg1], NA, x_disp[seg2], NA, x_disp[seg3])
y_line <- c(y[seg1], NA, y[seg2], NA, y[seg3])

par(mar = c(5.1, 6.2, 4.1, 2.1))

par(cex.lab = 2.5, cex.axis = 1.6, tcl = -0.3, las = 2, bty = "l", lwd = 2)                 

plot(x_line, y_line, type = "n", xaxt = "n", xlab = "Day", ylab = "Bray-Curtis")

shade_idx   <- x >= -6 & x <= 0
usr <- par("usr")
rect(min(x_disp[shade_idx]), usr[3], max(x_disp[shade_idx]), usr[4],
     col = adjustcolor("#E33A3A", 0.25), border = NA)

y0 <- par("usr")[3]

gap1_left  <- n1 + 0.5
gap1_right <- n1 + gap + 0.5
gap2_left  <- n1 + gap + n2 + 0.5
gap2_right <- n1 + gap + n2 + gap + 0.5

segments(gap1_left,  y0, gap1_right, y0, col = "white", lwd = 6, xpd = NA)
segments(gap2_left,  y0, gap2_right, y0, col = "white", lwd = 6, xpd = NA)

lines(x_line, y_line, lwd = 4, col = "#C05A00")

lines(c(tail(x_disp[seg1], 1), head(x_disp[seg2], 1)), c(tail(y[seg1], 1),
                                                         head(y[seg2], 1)),
      lwd = 4, col = "#C05A00", lty = c(3, 1))

lines(c(tail(x_disp[seg2], 1), head(x_disp[seg3], 1)), c(tail(y[seg2], 1),
                                                         head(y[seg3], 1)),
      lwd = 4, col = "#C05A00", lty = c(3, 1))

axis(1, at = x_disp[seg1], labels = x[seg1], lwd = 2, lwd.ticks = 1.5)
axis(1, at = x_disp[seg2], labels = x[seg2], lwd = 2, lwd.ticks = 1.5)
axis(1, at = x_disp[seg3], labels = x[seg3], lwd = 2, lwd.ticks = 1.5)
