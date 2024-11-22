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
root = "models/bird-2024_04_29/data.xlsx"
cocoevalimg <- read_excel(path = root, sheet = "ST3-cocoevalimg")

# Identify cause of error
data <- cocoevalimg %>%
    filter(aRng == "[0, 10000000000]") %>%
    group_by(Ids, image_id) %>%
    mutate(error = case_when(
        Matches > 0 ~ 'True Positive',
        det_type == 'gt' &
        any(category_id == -1 & Matches == 0 & det_type == 'gt') == TRUE ~
        'False Negative Detection',
        det_type == 'gt' &
        any(category_id == -1 & Matches == 0 & det_type == 'gt') == FALSE ~
        'False Negative Classification',
        det_type != 'gt' & iou_n > 0 ~ 'False Positive Overlap',
        det_type != 'gt' &
        any(
            category_id == -1 &
            Matches == 0 &
            det_type == 'dt' &
            iou_n == 0) == TRUE ~
        'False Positive Detection',
        det_type != 'gt' &
        any(
            category_id == -1 &
            Matches == 0 &
            det_type == 'dt' &
            iou_n == 0) == FALSE ~
        'False Positive Classification',
        TRUE ~ NA))

# Error types
thresholds <- seq(0, 1, 0.01)
error_data <- data.frame()
for (threshold in thresholds) {
    print(threshold)
    i <- data %>%
        filter(
            category_id == -1,
            error == "False Negative Detection" |
            error == "False Negative Classification" |
            det_type == 'dt') %>%
        mutate(error = case_when(
            error == 'True Positive' &
            Scores < threshold ~
            "False Negative Classification",
            TRUE ~ error)) %>%
        filter(
            Scores >= threshold |
            error == "False Negative Detection" |
            error == "False Negative Classification") %>%
        group_by(error) %>%
        summarise(count = n()) %>%
        mutate(score_thr = threshold) %>%
        filter(error != '', error != "True Positive") %>%
        mutate(proportion = count / sum(count))

    false <- i %>%
        summarise_if(is.numeric, sum) %>%
        mutate(error = "Total False Predictions", score_thr = threshold)
    error_data <- rbind(error_data, i, false)
}

# Plot
coeff = max(error_data["count"])
error_plot <- ggplot(
    data = filter(error_data, error != "Total False Predictions"),
    aes(x = score_thr, y = proportion, fill = error)) +
    geom_area(colour = "black") +
    geom_line(
        data = filter(error_data, error == "Total False Predictions"),
        aes(x = score_thr, y = count/coeff),
        linewidth = 1,
        linetype = "dashed") +
    scale_y_continuous(
        name = "Error Proportion",
        sec.axis = sec_axis(~.*coeff, name="Total False Predictions"),
        expand = expansion(mult = c(0, 0))) +
    scale_x_continuous(expand = expansion(mult = c(0, 0))) +
    labs(x = "Score Threshold", fill = "Error Type", linetype = "") +
    theme_classic() +
    theme(legend.position = "bottom")

ggsave("figures/f0-error.jpg", error_plot)
