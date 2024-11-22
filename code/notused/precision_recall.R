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
    sheet = "ST2-cocoeval")

# Import index to class
index_to_class <- fromJSON("resources/index_to_class.json")
# Convert to dataframe and add detection and classification classes
detection <- c(catId = -1, class = "Detection")
classification <- c(catId = -2, class = "Classification")
index_to_class <- data.frame(
    catId = names(index_to_class),
    do.call(rbind, index_to_class)) %>%
    mutate(class = str_to_title(paste(name, age, sep = " - "))) %>%
    select(catId, class) %>%
    rbind(detection) %>%
    rbind(classification)

# General filters
data <- data_raw %>%
    filter(
        iou_type == "bbox",
        iouThr == 0.5,
        area == "[0, 10000000000.0]",
        maxDet == 100,
        precision != -1) %>%
    select(precision, recThr, catId)

# Classification precision recall
classification_PR <- data %>%
    filter(catId != -1) %>%
    group_by(recThr) %>%
    summarise(precision = mean(precision)) %>%
    mutate(catId = -2)

# All precision recall
PR <- rbind(classification_PR, data) %>%
    merge(index_to_class, by = 'catId') %>%
    complete(class = index_to_class$class) %>%
    complete(recThr = seq(0, 1, 0.01))

# Plot
plot_data <- PR %>%
    filter(class == "Detection" | class == "Classification")

PR_plot <- ggplot(plot_data, aes(x = recThr, y = precision, linetype = class)) +
    geom_line() +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    labs(x = "Recall", y = "Precision", linetype = "Prediction Type:") +
    theme_classic() +
    theme(
        axis.text = element_text(size = 13),
        axis.title = element_text(size = 16),
        legend.text = element_text(size = 13),
        legend.title = element_text(size = 16),
        legend.position = "bottom")

ggsave("PR.jpg", PR_plot)
