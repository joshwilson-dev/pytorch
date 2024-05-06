# Title:        
# Description:  
# Author:       Joshua P. Wilson
# Date:         14/02/2024

# Load environment
renv::load("C:/Users/uqjwil54/Documents/Projects/venvs/DBBD")

# Install required packages
# install.packages("tidyverse")
# install.packages("readxl")

# Load required libraries
library(tidyverse)
library(readxl)

# Clear the R environment
rm(list = ls())

# Import data from excel
root <- "C:/Users/uqjwil54/Documents/Projects/DBBD/balanced-2024_05_01/model-2024_05_06/data.xlsx"
dataset <- read_excel(path = root, sheet = "ST1-dataset")
cocoevalimg <- read_excel(path = root, sheet = "ST3-cocoevalimg")
catId_to_class <- read_excel(path = root, sheet = "ST5-catId_to_class") %>%
    mutate(class = paste(name, "-", age)) %>%
    select(class, catId) %>%
    rbind(data.frame(class = c("classification", "detection"), catId = c(-2, -1)))

# Calculate performance metrics at threshold
data <- cocoevalimg %>%
    filter(dataset == 'test', area == "[0, 500]", Ignore == 0) %>%
    mutate(error = case_when(
        det_type == 'gt' ~ '',
        Matches > 0 ~ 'TP',
        T ~ 'FP')) %>%
    group_by(catId) %>%
    arrange(catId, desc(Scores)) %>%
    mutate(
        gt = sum(det_type == "gt"),
        tp = case_when(error == 'TP' ~ 1, T ~ 0),
        fp = case_when(error == 'FP' ~ 1, T ~ 0),
        tp = cumsum(tp),
        fp = cumsum(fp),
        r = floor(tp / gt * 100)/100,
        p = tp / (tp + fp)) %>%
    arrange(catId, desc(r)) %>%
    mutate(p = cummax(p)) %>%
    arrange(catId, r) %>%
    group_by(catId, r) %>%
    slice(1) %>%
    group_by(catId) %>%
    select(catId, r, p) %>%
    complete(r = round(seq(0, 1, 0.01), 2)) %>%
    arrange(catId, desc(r)) %>%
    fill(p) %>%
    mutate(p = case_when(is.na(p) ~ 0, T ~ p))

AP <- data %>%
    group_by(catId) %>%
    summarise(AP = mean(p))
