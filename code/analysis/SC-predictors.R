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

cocoeval <- read_excel(path = root, sheet = "ST2-cocoeval")

dataset <- read_excel(path = root, sheet = "ST1-dataset")

catId_to_class <- read_excel(path = root, sheet = "ST5-catId_to_class") %>%
    mutate(class = str_to_title(paste(name, age, sep = " - "))) %>%
    select(catId, class, plumage)

# Get AP per area
by <- 500
pixel_bins = seq(0, 5000, by = by)
AP <- cocoeval %>%
    filter(
        dataset == "test",
        iouType == "bbox",
        iouThr == 0.5,
        area != "[0, 10000000000]",
        area != "[5000, 10000000000]",
        maxDet == 100,
        catId != -1,
        precision != -1) %>%
    mutate(
        pixels = as.numeric(str_extract(area, "\\d+(?=\\])")),
        pixels = (2 * pixels - by)/2) %>%
    group_by(catId, pixels) %>%
    summarise(AP = mean(precision), AP_sd = sd(precision), .groups = "drop") %>%
    merge(catId_to_class, by = "catId")

print(length(unique(AP$class)))

# Get train instances
train <- dataset %>%
    filter(
        dataset == "train",
        obscured == "no",
        overlap >= 0.7,
        species != "unknown",
        species != "background") %>%
    mutate(
        pixels = cut(area, pixel_bins, labels = FALSE) * by,
        pixels = (2 * pixels - by)/2,
        class = str_to_title(paste0(commonname, " - ", age))) %>%
    filter(!is.na(pixels)) %>%
    group_by(class, pixels) %>%
    summarise(
        train_locations = sum(),
        area_avg = mean(area * gsd**2),
        train_instances = n()) %>%
    merge(catId_to_class, by = "class")
length(unique(train$class))


# Number of similar classes based on size and colour
similar_classes <- train %>%
    group_by(class) %>%
    summarise(
        area_avg = mean(area_avg),
        plumage = max(plumage)) %>%
    group_by(plumage) %>%
    mutate(similar_classes = sapply(area_avg, function(x) sum(abs(area_avg - x) <= 0.1))) %>%
    select(-area_avg)

train <- train %>%
    select(-plumage, -catId, -area_avg)

# Get test metrics
test <- dataset %>%
    filter(
        dataset == "test",
        obscured == "no",
        overlap >= 0.7,
        species != "unknown",
        species != "background") %>%
    mutate(
        # area = case_when(gsd < 0.005 ~ area * (gsd/0.005)**2, TRUE ~ area),
        pose = case_when(pose == 'resting' ~ 1, T ~ 0),
        pixels = cut(area, pixel_bins, labels = FALSE) * by,
        pixels = (2 * pixels - by)/2,
        class = str_to_title(paste0(commonname, " - ", age))) %>%
    filter(!is.na(pixels)) %>%
    group_by(class, pixels) %>%
    summarise(
        test_instances = n(),
        area_avg = mean(area * gsd**2),
        pixel_avg = mean(area),
        resting_proportion = mean(pose))
length(unique(test$class))

# Merge train with AP and prepare for beta regression
performance <- AP %>%
    merge(train) %>%
    merge(test) %>%
    merge(similar_classes) %>%
    filter(test_instances >= 3)
    # filter(AP_sd > 0)

length(unique(performance$class))

# Fit beta regression model
model <- glm(
    # AP ~ train_locations + train_instances + pixel_avg + area_avg + plumage + resting_proportion + similar_classes,
    AP ~ train_instances + pixel_avg + area_avg + plumage + similar_classes,
    data = performance,
    family = binomial(link = "logit"))
    # weight = 1 / AP_sd)
summary(model)

# Perform backward selection using stepwise AIC
best_model <- step(model, direction = "backward")
summary(best_model)

generate_plot <- function(predictor) {
    # Create datafram of mean values
    newdata <- data.frame(
        train_instances = mean(performance$train_instances),
        area_avg = mean(performance$area_avg),
        pixel_avg = mean(performance$pixel_avg),
        similar_classes = mean(performance$pixel_avg),
        resting_proportion = mean(performance$resting_proportion),
        plumage = "brown")
    
    # Drop predictor column
    newdata <- select(newdata, -predictor)
    
    # Expand dataframe for value of interest
    predictor_type = typeof(performance[[predictor]])

    if (predictor_type == "double" | predictor_type == "integer") {
        newdata <- expand_grid(
            newdata,
            !!predictor := seq(
                min(performance[[predictor]]),
                max(performance[[predictor]]),
                length.out = 20))}
    else {
        newdata <- expand_grid(
            newdata,
            !!predictor := unique(performance[[predictor]]))}

    # Predict AP values for the new count values
    AP_pred <- predict(best_model, newdata = newdata, type = "link", se.fit = TRUE)

    # Calculate 95% CI
    z_value <- qnorm(0.975) ## approx 95% CI
    AP_pred$upr_ci <- AP_pred$fit + z_value * AP_pred$se.fit
    AP_pred$lwr_ci <- AP_pred$fit - z_value * AP_pred$se.fit

    # Convert to response scale
    AP_pred$AP_pred <- best_model$family$linkinv(AP_pred$fit)
    AP_pred$upr_ci <- best_model$family$linkinv(AP_pred$upr_ci)
    AP_pred$lwr_ci <- best_model$family$linkinv(AP_pred$lwr_ci)

    # Create a dataframe with predictions
    predictions <- cbind(newdata, as.data.frame(AP_pred))

    # Plot result
    if (predictor_type == "double" | predictor_type == "integer") {
        instances_plot <- ggplot(predictions, aes_string(x = predictor, y = "AP_pred")) +
            geom_point(data = performance, aes_string(x = predictor, y = "AP")) +
            geom_line() +
            geom_ribbon(aes(ymin = lwr_ci, ymax = upr_ci), alpha = 0.5) +
            ylim(0, 1) +
            labs(
                y = "Average Precision",
                x = str_to_sentence(sub("avg", "", sub("_", " ", predictor)))) +
            theme_classic()}
    
    else {
        instances_plot <- ggplot(predictions, aes_string(x = predictor, y = "AP_pred")) +
            geom_point(data = performance, aes_string(x = predictor, y = "AP"), size = 0.2) +
            geom_point() +
            geom_errorbar(aes(ymin = lwr_ci, ymax = upr_ci)) +
            ylim(0, 1) +
            theme_classic() +
            theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
    }

    filename = paste0("figures/analysis/", predictor, ".jpg")
    ggsave(filename, instances_plot)
}

generate_plot("train_instances")
generate_plot("area_avg")
generate_plot("pixel_avg")
generate_plot("plumage")
generate_plot("resting_proportion")
generate_plot("similar_classes")
