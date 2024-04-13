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
cocoevalimg <- read_excel(
    path = "models/bird_2024_04_04/data.xlsx",
    sheet = "ST3-cocoevalimg")

# Import index to class
index_to_class <- fromJSON("resources/index_to_class.json")
# Convert to dataframe and add detection and classification rows
detection <- c(category_id = -1, class = "Detection")
classification <- c(category_id = -2, class = "Classification")
index_to_class <- data.frame(
    category_id = names(index_to_class),
    do.call(rbind, index_to_class)) %>%
    mutate(class = str_to_title(paste(name, age, sep = " - "))) %>%
    select(category_id, class) %>%
    rbind(detection) %>%
    rbind(classification)

# Calculate performance metrics at threshold
threshold = 0.75
data <- cocoevalimg %>%
    mutate(error = case_when(
        Matches > 0 ~ 'TP',
        det_type == 'gt' ~ 'FN',
        det_type != 'gt' ~ 'FP')) %>%
    filter(Scores >= threshold & det_type == "dt" | error == "FN")

# Calculate for classification
class_data <- data %>%
    group_by(error) %>%
    filter(category_id != -1) %>%
    summarise(count = n(), .groups = "drop") %>%
    mutate(category_id = -2)

# Calculate per class and merge with classification
performance <- data %>%
    group_by(category_id, error) %>%
    summarise(count = n(), .groups = "drop") %>%
    rbind(class_data) %>%
    group_by(category_id) %>%
    complete(error = c("FP", "FN", "TP")) %>%
    replace(is.na(.), 0) %>%
    pivot_wider(names_from = error, values_from = count) %>%
    mutate(
        gt = TP + FN,
        P = TP/(TP + FP),
        P = case_when(is.na(P) ~ 0, T ~ P)) %>%
    merge(index_to_class, by = "category_id") %>%
    arrange(desc(P))