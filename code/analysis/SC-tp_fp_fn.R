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
catId_to_class <- read_excel(path = root, sheet = "ST5-catId_to_class") %>%
    mutate(class = paste(name, "-", age)) %>%
    select(class, catId) %>%
    rbind(data.frame(class = c("classification", "detection"), catId = c(-2, -1)))

# Calculate performance metrics at threshold
data <- cocoevalimg %>%
    filter(area == "[0, 10000000000]", dataset == 'test') %>%
    mutate(error = case_when(
        Matches > 0 ~ 'TP',
        det_type == 'gt' ~ 'FN',
        det_type != 'gt' ~ 'FP'))

performance <- data.frame()
for (threshold in seq(0, 1, 0.01)) {
    print(threshold)  
    i <- data %>%
        filter(error == "FN" | det_type == 'dt') %>%
        mutate(error = case_when(
            error == 'TP' &
            Scores < threshold ~
            "FN",
            TRUE ~ error)) %>%
        filter(Scores >= threshold | error == "FN")
    
    classification_data <- i %>%
        group_by(error) %>%
        filter(catId != -1) %>%
        summarise(count = n(), .groups = "drop") %>%
        mutate(catId = -2)
    
    class_data <- i %>%
        group_by(catId, error) %>%
        summarise(count = n(), .groups = "drop") %>%
        rbind(classification_data) %>%
        group_by(catId) %>%
        complete(error = c("FP", "FN", "TP")) %>%
        replace(is.na(.), 0) %>%
        pivot_wider(names_from = error, values_from = count) %>%
        mutate(
            score = threshold,
            F = FP + FN,
            GT = TP + FN,
            P = TP/(TP + FP),
            P = case_when(is.na(P) ~ 0, T ~ P),
            R = TP/(TP + FN),
            F1 = (2 * P * R)/(P + R)) %>%
        merge(catId_to_class, by = "catId") %>%
        arrange(desc(P))
    
    performance <- rbind(performance, class_data)
}

# Best performance
best <- performance %>%
    group_by(class) %>%
    arrange(F) %>%
    slice(1) %>%
    mutate(
        class = str_to_title(class),
        across(where(is.numeric), round, 2)) %>%
    arrange(desc(F1))

write.csv(best, "best.csv")