# Title:        CS6 Average Precision
# Description:  Calculate the average precision on the test and validation sets
#               for all classes, detection, and classification
# Author:       Anonymous
# Date:         05/06/2024

# Load required libraries
library(tidyverse)
library(readxl)

# Clear the R environment
rm(list = ls())

# Import data from excel
root <- "Supporting Tables TS1-6.xlsx"
dataset <- read_excel(path = root, sheet = "TS2-dataset")
cocoeval <- read_excel(path = root, sheet = "TS3-cocoeval")
catId_to_class <- read_excel(path = root, sheet = "TS6-catId_to_class") %>%
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

# PR plot
PR_data <- data %>%
    filter(catId == -1, dataset == "validation")

PR_plot <- ggplot(data = PR_data, aes(x = recThr, y = precision)) +
    geom_line() +
    geom_area(alpha= 0.4) +
    theme_classic() +
    labs(x = "Recall", y = "Precision") +
    theme(
        axis.text =element_text(colour = "black"),
        panel.grid.major = element_line(colour = "black", size = 0.25)) +
    scale_x_continuous(expand = c(0, 0, 0, 0.02)) +
    scale_y_continuous(expand = c(0, 0))

ggsave("figures/PR_plot.jpg", PR_plot, height = 3, width = 3)


# Detection average precision
AP_det <- data %>%
    filter(catId == -1) %>%
    group_by(dataset, catId) %>%
    summarise(
        AP = mean(precision),
        SD = sd(precision),
        .groups = "drop")

# Class average precision
AP_class <- data %>%
    filter(catId != -1) %>%
    group_by(dataset, catId) %>%
    summarise(
        AP = mean(precision),
        SD = sd(precision),
        .groups = "drop")

# Classification average precision
AP_clas <- AP_class %>%
    group_by(dataset) %>%
    summarise(
        SD = sd(AP),
        AP = mean(AP),
        .groups = "drop") %>%
    mutate(catId = -2)

# Join AP
AP <- rbind(AP_det, AP_clas, AP_class) %>%
    merge(catId_to_class, by = 'catId')

# Get number of instances

# For each class
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

# For all classes
instances_det <- instances_class %>%
    group_by(dataset) %>%
    summarise(instances = sum(instances), .groups = "drop") %>%
    mutate(class = "Detection")

# For all classes
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
        unique(filter(performance, dataset == 'validation')$class),
        unique(filter(performance, dataset == 'test')$class)))
performance$class <- factor(
    performance$class,
    levels = class_order,
    ordered = TRUE)

# Create AP plot
AP_plot <- ggplot(data = performance, aes(x = class, y = AP, shape = dataset)) +
    geom_point(size = 3) +
    theme_classic() +
    labs(x = "", y = "Average Precision", shape = "Dataset") + 
    scale_shape_discrete(labels = function(labels) str_to_title(labels)) +
    theme(
        panel.grid.major.y = element_line(),
        axis.title = element_text(size = 13, colour = 'black'),
        axis.text = element_text(size = 13, colour = 'black'),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        legend.text = element_text(size = 13),
        legend.title = element_text(size = 13))

ggsave("figures/f6-average_precision.jpg", AP_plot, width = 13, height = 6)