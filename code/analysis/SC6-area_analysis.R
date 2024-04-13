# Title:        
# Description:  
# Author:       Joshua P. Wilson
# Date:         14/02/2024

# Load environment
renv::load("C:/Users/uqjwil54/Documents/Projects/venvs/DBBD")

# Install required packages
# install.packages("tidyverse")
# install.packages("readxl")
# install.packages("jsonlite")

# Load required libraries
library(tidyverse)
library(readxl)
library(jsonlite)

# Clear the R environment
rm(list = ls())

# Import data from excel
data_raw <- read_excel(
    path = "models/bird_2024_04_04/data.xlsx",
    sheet = "ST4-cocoevalarea")

# Import index to class
index_to_class <- fromJSON("resources/index_to_class.json")
# Convert to dataframe and add detection and classification rows
detection <- c(catId = -1, class = "Detection")
classification <- c(catId = -2, class = "Classification")
index_to_class <- data.frame(
    catId = names(index_to_class),
    do.call(rbind, index_to_class)) %>%
    mutate(class = str_to_title(paste(name, age, sep = " - "))) %>%
    select(catId, class) %>%
    rbind(detection) %>%
    rbind(classification)

# Determine AP for each class at each area bin
class_AP <- data %>%
    group_by(catId, area) %>%
    filter(
        precision != -1,
        iouType == "bbox",
        iouThr == 0.5,
        maxDet == 100) %>%
    summarise(AP = mean(precision), .groups = "drop")

# Add on AP for classifiation
classification_AP <- data %>%
    group_by(area) %>%
    filter(
        catId != -1,
        precision != -1,
        iouType == "bbox",
        iouThr == 0.5,
        maxDet == 100) %>%
    summarise(AP = mean(precision), .groups = "drop") %>%
    mutate(catId = -2)

area_AP <- rbind(class_AP, classification_AP) %>%
    merge(data_raw, index_to_class, by = 'catId') %>%
    mutate(area = as.double(str_extract(area, "\\d+(?!.*\\d)"))) %>%
    filter(area != 10000000000)

# Add zeros
zero_rows <- area_AP %>%
    distinct(class) %>%
    mutate(area = 0, AP = 0)

area_data <- area_AP %>%
    bind_rows(zero_rows) %>%
    arrange(class, area)

# Plot detection, classification, and selected classes
plot_classes <- c(
    "Detection",
    "Classification",
    "Australian Pelican - Adult",
    "Australian White Ibis - Adult",
    "Pied Stilt - Adult")

plot_data <- area_data %>%
    filter(class %in% plot_classes) %>%
    mutate(class = factor(class, levels = plot_classes))
    
area_plot <- ggplot(plot_data, aes(x = area, y = AP, linetype = class)) +
    geom_line() +
    labs(x = "Area", y = "Average Precision", linetype = "Class:") +
    scale_linetype_manual(values = c("solid", "dashed", "dotted", "dotdash", "1F")) +
    theme_classic() +
    theme(
        legend.text = element_text(size = 7),
        legend.position = "bottom")
ggsave("area.jpg", area_plot)