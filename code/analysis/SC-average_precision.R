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
# install.packages("betareg")
# install.packages("boot")

# Load required libraries
library(tidyverse)
library(readxl)
library(jsonlite)

# Clear the R environment
rm(list = ls())

# Import data from excel
cocoeval <- read_excel(
    path = "models/bird_2024_04_04/data.xlsx",
    sheet = "ST2-cocoeval")

dataset <- read_excel(
    path = "models/bird_2024_04_04/data.xlsx",
    sheet = "ST1-dataset")

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

# General filters
data <- cocoeval %>%
    filter(
        iou_type == "bbox",
        iouThr == 0.5,
        area == "[0, 10000000000.0]",
        maxDet == 100,
        precision != -1)

# Detection average precision
detection_AP <- data %>%
    filter(catId == -1) %>%
    group_by(catId) %>%
    summarise(AP = mean(precision))

# Classification average precision
classification_AP <- data %>%
    filter(catId != -1) %>%
    summarise(AP = mean(precision)) %>%
    mutate(catId = -2)

# Class AP
class_AP <- data %>%
    group_by(catId) %>%
    filter(catId != -1) %>%
    summarise(AP = mean(precision), .groups = "drop")

# Get test instances
class_test <- dataset %>%
    filter(
        dataset == "test",
        obscured == "no",
        overlap >= 0.75,
        species != "unknown",
        species != "background") %>%
        mutate(class = str_to_title(paste0(commonname, " - ", age))) %>%
    group_by(class) %>%
    summarise(test_instances = n())

detection_test <- class_test %>%
    ungroup() %>%
    summarise(test_instances = sum(test_instances)) %>%
    mutate(class = "Detection")

classification_test <- class_test %>%
    ungroup() %>%
    summarise(test_instances = sum(test_instances)) %>%
    mutate(class = "Classification")

# Get train instances
class_train <- dataset %>%
    filter(
        dataset == "train",
        obscured == "no",
        overlap >= 0.75,
        species != "unknown",
        species != "background") %>%
        mutate(class = str_to_title(paste0(commonname, " - ", age))) %>%
    group_by(class) %>%
    summarise(train_instances = n())

detection_train <- class_train %>%
    ungroup() %>%
    summarise(train_instances = sum(train_instances)) %>%
    mutate(class = "Detection")

classification_train <- class_train %>%
    ungroup() %>%
    summarise(train_instances = sum(train_instances)) %>%
    mutate(class = "Classification")

# Combine AP and Test and Train
test <- rbind(detection_test, classification_test, class_test)

train <- rbind(detection_train, classification_train, class_train)

AP <- rbind(detection_AP, classification_AP, class_AP) %>%
    merge(index_to_class, by = 'catId')

performance <- merge(AP, test) %>%
    merge(train) %>%
    # filter(test_instances > 10) %>%
    arrange(desc(AP))

# Order by AP, but keep detection and classification first 
class_order <- unique(c("Detection", "Classification", unique(performance$class)))
performance$class <- factor(performance$class, levels = class_order, ordered = TRUE)

# Create AP plot
AP_plot <- ggplot(performance, aes(x = class, y = AP)) +
    geom_bar(stat = "identity") +
    geom_hline(yintercept = seq(0, 1, by = 0.25), color = "gray", linetype = "dashed") +
    theme_classic() +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    labs(x = "Class") +
    theme(
        axis.title = element_text(size = 13),
        axis.text = element_text(size = 13),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    # coord_flip()

ggsave("AP.jpg", AP_plot, width = 14, height = 7)

