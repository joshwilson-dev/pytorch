# Title:        CS3 Dataset Overview
# Description:  Metrics summarising the context of the dataset.
# Author:       Anonymous
# Date:         05/06/2024

# Load required libraries
library(tidyverse)
library(readxl)
library(maps)

# Clear the R environment
rm(list = ls())

# Import data from excel
root <- "Supporting Tables TS1-6.xlsx"
dataset_raw <- read_excel(
    path = root,
    sheet = "TS2-dataset")

# Process data
dataset <- dataset_raw %>%
    filter(
      overlap >= 0.7,
      obscured == 'no',
      dataset != "test",
      dataset != "validation",
      species != 'unknown',
      species != "background")

# Class

# Count the number of instances per class
class_counts <- dataset %>%
  group_by(dataset, class, order, genus, commonname) %>%
  count() %>%
  ungroup() %>%
  pivot_wider(names_from = dataset, values_from = n) %>%
  filter(total >= 10) %>%
  mutate(
    total = log(total),
    train = log(train),
    total = total - train) %>%
  ungroup() %>%
  pivot_longer(
    !c(class, order, genus, commonname),
    names_to = 'dataset',
    values_to = 'count') %>%
  arrange(class, order, genus, commonname)

# Order class factor by ored, family, genus, species
class_order <- c(unique(class_counts$commonname))
class_counts$class <- factor(
  class_counts$commonname,
  levels = class_order,
  ordered = TRUE)

# Add vertical lines
annotation <- class_counts %>%
  filter(dataset == "total") %>%
  mutate(vline = row_number() + 0.5) %>%
  group_by(order) %>%
  slice(n())

# Make y axis non-scientific
options(scipen = 999)
breaks <- log(c(5, 50, 500, 5000))
labels <- c(5, 50, 500, 5000)

# Create and save bar plot
class_bias <- ggplot(class_counts, aes(x = class, y = count, fill = dataset)) +
    geom_bar(stat = "identity", colour = "black") +
    labs(x = "", y = "Count", fill = "Dataset") +
    scale_fill_manual(values = c("gray60", "gray40")) +
    scale_y_continuous(
      breaks = breaks,
      labels = labels,
      expand = c(0, 0, 0, 1)) +
    geom_segment(
      data = annotation,
      aes(x = vline, xend = vline, yend = 9.5),
      linetype = "dashed") +
    theme_classic() +
    theme(
      plot.margin = margin(20, 20, 20, 20),
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

ggsave(
  filename = "figures/f5a-class.jpg",
  plot = class_bias,
  height = 5,
  width = 10)

# Posture

# Count the number of instances per posture
posture_counts <- dataset %>%
    group_by(posture, dataset) %>%
    count() %>%
    pivot_wider(names_from = dataset, values_from = n) %>%
    mutate(total = total - train) %>%
    pivot_longer(!posture, names_to = 'dataset', values_to = 'count') %>%
    mutate(dataset = factor(
      dataset,
      levels = c("total", "train"),
      ordered = TRUE))

# Create and save posture bar plot
posture_bias <- ggplot(
  posture_counts,
  aes(x = posture, y = count, fill = dataset)) +
    geom_bar(stat = "identity", colour = "black") +
    labs(x = "Posture", y = "Count", fill = "Dataset") +
    scale_fill_manual(values = c("gray60", "gray40")) +
    scale_y_continuous(limits = c(0, 30000)) +
    theme_minimal() +
    theme(
        axis.title = element_text(size = 30),
        axis.text = element_text(size = 30, colour = "black"),
        legend.position = "none",
        plot.margin = margin(50, 0, 0, 10))

ggsave(
  filename = "figures/f5b-posture.jpg",
  plot = posture_bias,
  height = 8,
  width = 8)

# GSD

# Count the number of instances per GSD
gsd_breaks <- c(0, seq(2, 20, by = 2), Inf)
gsd_counts <- dataset %>%
    mutate(
      gsd = gsd * 1000,
      gsd_cats = cut(gsd, breaks = gsd_breaks)) %>%
    group_by(gsd_cats, dataset) %>%
    count() %>%
    pivot_wider(names_from = dataset, values_from = n) %>%
    mutate(total = total - train) %>%
    pivot_longer(!gsd_cats, names_to = 'dataset', values_to = 'count')

# Create and save plot
gsd_bias <- ggplot(gsd_counts, aes(x = gsd_cats, y = count, fill = dataset)) +
    geom_bar(stat = "identity", colour = "black") +
    labs(x = "GSD [mm/pix]", y = "Count", fill = "Dataset") +
    scale_fill_manual(values = c("gray60", "gray40")) +
    theme_minimal() +
    theme(
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        axis.title = element_text(size = 30),
        axis.text = element_text(size = 30, colour = "black"),
        legend.position = "none",
        plot.margin = margin(50, 0, 0, 10))

ggsave(
  filename = "figures/f5c-gsd.jpg",
  plot = gsd_bias,
  height = 8,
  width = 8)

# Age

# Count instances per age
age_counts <- dataset %>%
    group_by(age, dataset) %>%
    count() %>%
    pivot_wider(names_from = dataset, values_from = n) %>%
    mutate(total = total - train) %>%
    pivot_longer(!age, names_to = 'dataset', values_to = 'count')

# Create and save plot
age_bias <- ggplot(age_counts, aes(x = age, y = count, fill = dataset)) +
    geom_bar(stat = "identity", colour = "black") +
    labs(x = "Age", y = "Count", fill = "Dataset") +
    scale_fill_manual(values = c("gray60", "gray40")) +
    theme_minimal() +
    theme(
        axis.title = element_text(size = 30),
        axis.text = element_text(size = 30, colour = "black"),
        legend.position = "none",
        plot.margin = margin(50, 0, 0, 10))

ggsave(
  filename = "figures/f5d-age.jpg",
  plot = age_bias,
  height = 8,
  width = 8)

# Location

# Reverse geocode to get country for each latitude and longitude
location_data <- dataset %>%
  mutate(
    location = paste0(round(latitude, 0), ", ", round(longitude, 0)),
    country = map.where(database = "world", longitude, latitude),
    country = case_when(
      location == "-62, -59" ~ "Antarctica",
      location == "-51, -61" ~ "Falkland Islands",
      location == "-28, 153" ~ "Australia",
      location == "52, 143" ~ "Russia",
      location == "54, 14" ~ "Poland",
      location == "56, 8" ~ "Denmark",
      location == "-27, 153" ~ "Australia",
      location == "59, -140" ~ "Alaska",
      location == "-63, -61" ~ "Antarctica",
      TRUE ~ country))

# Filter and count instances per country
country_count <- location_data %>%
  # group_by(country, dataset) %>%
  group_by(country) %>%
  summarise(count = n(), .groups = "drop")

# Calculate mean latitude and longitude per country
country_center <- location_data %>%
  group_by(country) %>%
  summarise(
    mean_latitude = mean(latitude),
    mean_longitude = mean(longitude),
    .groups = "drop")
country_data <- merge(country_count, country_center, by = "country")

# Plot world map with instances per location
world_map <- map_data("world")

location_bias <- ggplot() +
  geom_polygon(
    data = world_map,
    aes(x = long, y = lat, group = group),
    colour = "gray85",
    fill = "gray80") +
  coord_fixed(1.3) +
  geom_point(
    data = country_data,
    aes(
      x = mean_longitude,
      y = mean_latitude,
      # colour = dataset,
      size = count)) +
  # scale_colour_manual(values = c('gray60', 'gray40'), name = 'Dataset') +
  scale_size_continuous(
    range = c(3, 15),
    breaks = c(1250, 5000, 10000),
    name = "Count") +
  theme_void() +
  theme(legend.position = c(0.15, 0.4)) +
  guides(colour = guide_legend(override.aes = list(size=10)))

# ggsave(filename = "figures/f5e-location.jpg", plot = location_bias)
ggsave(filename = "figures/presentation.jpg", plot = location_bias)

# Number of images
n_images <- dataset_raw %>%
    filter(dataset == "total") %>%
    distinct(imagepath) %>%
    nrow()

# Number of instances
n_instances <- dataset_raw %>%
    filter(label != "background", dataset == "total") %>%
    distinct(imagepath, points) %>%
    nrow()

# Number of species
n_species <- dataset_raw %>%
    filter(label != "background", dataset == "total") %>%
    mutate(species = paste(genus, species)) %>%
    distinct(species) %>%
    nrow()

# Count of species
species_count <- dataset_raw %>%
    filter(label != "background", dataset == "total") %>%
    mutate(species = paste(genus, species)) %>%
    group_by(species, commonname) %>%
    summarise(count = n())

# Count of postures
posture_count <- dataset_raw %>%
    filter(label != "background", dataset == "total") %>%
    group_by(posture) %>%
    summarise(count = n())

# Count of age
age_count <- dataset_raw %>%
    filter(label != "background", dataset == "total") %>%
    group_by(age) %>%
    summarise(count = n())

# GSD stats
gsd_stats <- dataset_raw %>%
  filter(label != "background", dataset == "total") %>%
  summarise(mean_gsd = mean(gsd), max_gsd = max(gsd), min_gsd = min(gsd))

# Images camera
images_per_camera <- dataset_raw %>%
    filter(dataset == "total") %>%
    distinct(imagepath, camera) %>%
    group_by(camera) %>%
    summarise(count = n())
