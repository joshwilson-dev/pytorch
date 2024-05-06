# Title:        
# Description:  
# Author:       Joshua P. Wilson
# Date:         14/02/2024

# Load environment
renv::load("C:/Users/uqjwil54/Documents/Projects/venvs/DBBD")

# Install required packages
# install.packages("tidyverse")

# Load required libraries
library(tidyverse)
library(readxl)

# Clear the R environment
rm(list = ls())

# Import data from excel
root <- "C:/Users/uqjwil54/Documents/Projects/DBBD/balanced-2024_05_01/model-2024_05_06/data.xlsx"
dataset <- read_excel(path = root, sheet = "ST1-dataset")
cocoeval <- read_excel(path = root, sheet = "ST2-cocoeval")
catId_to_class <- read_excel(path = root, sheet = "ST5-catId_to_class") %>%
    # Create class column
    mutate(class = str_to_title(paste(name, age, sep = " - "))) %>%
    select(catId, class) %>%
    # Add detection and classification to index_to_class
    rbind(c(catId = -1, class = "Detection")) %>%
    rbind(c(catId = -2, class = "Classification"))

# General filters
data <- cocoeval %>%
    filter(
        iouType == "bbox",
        iouThr == 0.5,
        area == "[0, 10000000000]",
        maxDet == 100,
        precision != -1)

# Detection average precision
AP_det <- data %>%
    filter(catId == -1) %>%
    group_by(dataset, catId) %>%
    summarise(AP = mean(precision), .groups = "drop")

# Classification average precision
AP_clas <- data %>%
    filter(catId != -1) %>%
    group_by(dataset) %>%
    summarise(AP = mean(precision), .groups = "drop") %>%
    mutate(catId = -2)

# Class average precision
AP_class <- data %>%
    filter(catId != -1) %>%
    group_by(dataset, catId) %>%
    summarise(AP = mean(precision), .groups = "drop")

# Join AP
AP <- rbind(AP_det, AP_clas, AP_class) %>%
    merge(catId_to_class, by = 'catId')

# Get number of instances
instances_class <- dataset %>%
    filter(
        dataset != 'total',
        obscured == "no",
        overlap >= 0.7,
        species != "unknown",
        species != "background") %>%
        mutate(class = str_to_title(paste0(commonname, " - ", age))) %>%
    group_by(dataset, class) %>%
    summarise(instances = n(), .groups = "drop")

instances_det <- instances_class %>%
    group_by(dataset) %>%
    summarise(instances = sum(instances), .groups = "drop") %>%
    mutate(class = "Detection")

instances_clas <- instances_det %>%
    mutate(class = "Classification")

# Join instance counts
instances <- rbind(instances_det, instances_clas, instances_class)

# Combine AP and instances
performance <- AP %>%
    merge(instances) %>%
    filter(instances > 10) %>%
    arrange(desc(AP))

# Order by AP, but keep detection and classification first 
class_order <- unique(
    c(
        "Detection",
        "Classification",
        unique(filter(performance, dataset == 'test')$class)))
performance$class <- factor(performance$class, levels = class_order, ordered = TRUE)

# Create AP plot
AP_plot <- ggplot(performance, aes(x = class, y = AP, colour = dataset)) +
    geom_point() +
    geom_hline(yintercept = seq(0, 1, by = 0.25), color = "gray", linetype = "dashed") +
    theme_classic() +
    labs(x = "Class", y = "Average Precision") +
    scale_color_manual(values = c("grey20", "grey80")) +
    theme(
        axis.title = element_text(size = 13),
        axis.text = element_text(size = 13),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

ggsave("figures/f0-average_precision.jpg", AP_plot, width = 10, height = 5)