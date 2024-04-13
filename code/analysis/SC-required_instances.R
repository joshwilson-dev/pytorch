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
library(betareg)
library(boot)

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
# Convert to dataframe
index_to_class <- data.frame(
    catId = names(index_to_class),
    do.call(rbind, index_to_class)) %>%
    mutate(class = str_to_title(paste(name, age, sep = " - "))) %>%
    select(catId, class)

# Get class AP
AP <- cocoeval %>%
    filter(
        iou_type == "bbox",
        iouThr == 0.5,
        area == "[0, 10000000000.0]",
        maxDet == 100,
        precision != -1,
        catId != -1) %>%
    group_by(catId) %>%
    summarise(AP = mean(precision), .groups = "drop") %>%
    merge(index_to_class, by = "catId")

# Get train instances
train <- dataset %>%
    filter(
        dataset == "train",
        obscured == "no",
        overlap >= 0.75,
        species != "unknown",
        species != "background") %>%
        mutate(class = str_to_title(paste0(commonname, " - ", age))) %>%
    group_by(class) %>%
    summarise(count = n())

# Merge train with AP and prepare for beta regression
train_AP <- train %>%
    merge(AP) %>%
    mutate(
        count = case_when(is.na(count) ~ 0, TRUE ~ count),
        AP = case_when(AP == 0 ~ 0.001, AP == 1 ~ 0.999, T ~ AP))

# Fit beta regression model
model <- betareg(AP ~ count, data = train_AP)
summary(model)

# Predict over data range
newdata <- data.frame(count = seq(min(train_AP$count), max(train_AP$count)))

# Predict AP values for the new count values
AP_pred <- predict(model, newdata = newdata)

# Create a dataframe with predictions
predictions <- cbind(newdata, as.data.frame(AP_pred))

instances_plot <- ggplot(predictions, aes(x = count, y = AP_pred)) +
    geom_line() +
    geom_point(data = train_AP, aes(x = count, y = AP)) +
    labs(x = "Training Instances", y = "Average Precision") +
    theme_classic()

ggsave("figures/f7-instances.jpg", instances_plot, height = 3, width = 3)
