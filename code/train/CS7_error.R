# Title:        CS7 Error
# Description:  Plot the top causes of False Positive Detections
# Author:       Anonymous
# Date:         05/06/2024

# Load required libraries
library(tidyverse)
library(readxl)

# Clear the R environment
rm(list = ls())

# Import data from excel
root <- "Supporting Tables TS1-6.xlsx"
cocoevalimg <- read_excel(path = root, sheet = "TS4-cocoevalimg")
catId_to_class <- read_excel(path = root, sheet = "TS6-catId_to_class") %>%
    mutate(class = paste(name, "-", age)) %>%
    select(class, catId) %>%
    rbind(data.frame(class = "detection", catId = -1))

# Add error type
data <- cocoevalimg %>%
    filter(
        iouThrs == 0.5,
        maxDet == 100,
        area == "[0, 10000000000]") %>%
    mutate(
        error = case_when(
            Matches > 0 ~ 'TP',
            det_type == 'gt' ~ 'FN',
            det_type != 'gt' ~ 'FP')) %>%
    merge(catId_to_class)

# False Positive Detections
# Background objects detected as birds
# If it's detection, and it's a FP, and it doesn't overlap with anything,
# it must be an incorrect label or a FPD of a background
FPD <- data %>%
    filter(
        error == "FP",
        class == "detection",
        iou_max == 0,
        dataset == "validation") %>%
    arrange(desc(Scores))

# what objects were most commonly detected as false positives?
error_objects <- FPD %>%
    group_by(error_object) %>%
    count() %>%
    filter(
        error_object != "unknown",
        error_object != "not checked") %>%
    ungroup() %>%
    mutate(percentage = n/sum(n))
