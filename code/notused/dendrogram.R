# Title:        
# Description:  
# Author:       Joshua P. Wilson
# Date:         14/02/2024

# Load environment
renv::load("C:/Users/uqjwil54/Documents/Projects/venvs/DBBD")

# Load required libraries
library(tidyverse)
library(igraph)
library(ggraph)
library(rphylopic)
library(readxl)
library(maps)

# Clear the R environment
rm(list = ls())

# Import data from excel
root <- "C:/Users/uqjwil54/Documents/Projects/DBBD/2024_05_10/models/temp/data.xlsx"
dataset_raw <- read_excel(
    path = root,
    sheet = "TS2-dataset")

#### Species Bias ####
# Process data
dataset <- dataset_raw %>%
    filter(
      overlap >= 0.7,
      obscured == 'no',
      dataset != "test",
      dataset != "validation",
      species != 'unknown',
      species != "background") %>%
    mutate(
        species = paste0(toupper(substring(genus, 1, 1)), ". ", species),
        species = paste(class, order, family, genus, species),
        genus = paste(class, order, family, genus),
        family = paste(class, order, family),
        order = paste(class, order))

# Add links between class and order
order <- dataset %>%
    group_by(class, order, dataset) %>%
    summarise(count = n(), .groups = "drop") %>%
    pivot_wider(names_from = dataset, values_from = count, names_prefix = "count_") %>%
    mutate(group = order) %>%
    rename(from = class, to = order)

# Add links between order and family
family <- dataset %>%
    group_by(order, family, dataset) %>%
    summarise(count = n(), .groups = "drop") %>%
    pivot_wider(names_from = dataset, values_from = count, names_prefix = "count_") %>%
    mutate(group = order) %>%
    rename(from = order, to = family)

# Add links between family and genus
genus <- dataset %>%
    group_by(order, family, genus, dataset) %>%
    summarise(count = n(), .groups = "drop") %>%
    pivot_wider(names_from = dataset, values_from = count, names_prefix = "count_") %>%
    rename(group = order, from = family, to = genus)

# Add links between genus and species
species <- dataset %>%
    group_by(order, genus, commonname, dataset) %>%
    summarise(count = n(), .groups = "drop") %>%
    pivot_wider(names_from = dataset, values_from = count, names_prefix = "count_") %>%
    rename(group = order, from = genus, to = commonname) %>%
    mutate(to = str_to_title(to))
max_species = max(species$count_total)

# Create the count for the leaves
count <- distinct(rbind(order, family, genus, species)) %>%
    mutate(
        count_train = case_when(is.na(count_train) ~ 0, T ~ count_train),
        count_total = case_when(is.na(count_total) ~ 0, T ~ count_total),
        percentage_total = count_total/max_species,
        percentage_train = count_train/max_species)

max_species = round(max_species/5000,1)*5000

# Create the edges of dendrogram
edges <- select(count, from, to, group)

# Create the vertices dataframe
vertices <- data.frame(to = unique(c(edges$from, edges$to))) %>%
    mutate(sort = row_number()) %>%
    merge(., count, by = "to", all = TRUE) %>%
    replace(is.na(.), 0) %>%
    filter(from != "unknown") %>%
    arrange(sort) %>%
    mutate(
        label = sapply(strsplit(to, "\\s+"), function(x) paste(tail(x, 2), collapse = " ")),
        group = edges$group[match(to, edges$to)])

# Set up label angle and orientation
idx_leaves <- which(is.na(match(vertices$to, edges$from)))
n_leaves <- length(idx_leaves)
vertices$id[idx_leaves] <- seq(1:n_leaves)
vertices <- vertices %>%
    mutate(
        angle = 90 - 360 * id / n_leaves,
        hjust = ifelse(angle < -90, 1, 0),
        angle = ifelse(angle < -90, angle + 180, angle)
    )

# Add start and end for arc bar
vertices$start <- 2 * pi * (vertices$id - 1) / n_leaves
vertices$end <- vertices$start + 2 * pi / n_leaves

# Add phylopic position
# Calculate the number of species in each order
groupsize <- dataset %>%
  group_by(order) %>%
  summarize(groupsize = n_distinct(species)) %>%
  rename(to = order)

# Determine angle of each order group, convert to x and y to paste phylopic
vertices <- vertices %>%
  merge(., groupsize, by = "to", all = TRUE) %>%
  arrange(sort) %>%
  mutate(groupsize_lag = lag(groupsize, default = 0)) %>%
  mutate(csum_gsl = cumsum(ifelse(is.na(groupsize_lag), 0, groupsize_lag))) %>%
  mutate(phylopic_angle = (csum_gsl + groupsize / 2) * 2 * pi / n_leaves) %>%
  mutate(phylopic_x = 3.9 * sin(phylopic_angle)) %>%
  mutate(phylopic_y = 3.9 * cos(phylopic_angle))

# Create circle points for 100% and 50% arcbars
arcbarmax = 1.2
mod = 0.25
circles <- data.frame(
  x1 = cos(seq(0, 2 * pi, length.out = 100)) * (1 + arcbarmax),
  y1 = sin(seq(0, 2 * pi, length.out = 100)) * (1 + arcbarmax),
  x2 = cos(seq(0, 2 * pi, length.out = 100)) * (1 + arcbarmax/(max_species/(max_species/10))**mod),
  y2 = sin(seq(0, 2 * pi, length.out = 100)) * (1 + arcbarmax/(max_species/(max_species/10))**mod),
  x3 = cos(seq(0, 2 * pi, length.out = 100)) * (1 + arcbarmax/(max_species/(max_species/100))**mod),
  y3 = sin(seq(0, 2 * pi, length.out = 100)) * (1 + arcbarmax/(max_species/(max_species/100))**mod),
  x4 = cos(seq(0, 2 * pi, length.out = 100)) * (1 + arcbarmax/(max_species/(max_species/1000))**mod),
  y4 = sin(seq(0, 2 * pi, length.out = 100)) * (1 + arcbarmax/(max_species/(max_species/1000))**mod),
  x5 = cos(seq(0, 2 * pi, length.out = 100)) * 3.5,
  y5 = sin(seq(0, 2 * pi, length.out = 100)) * 3.5)

# Create a graph object
mygraph <- graph_from_data_frame(edges, vertices = vertices)

# Make the plot
dendrogram <- (
  ggraph(mygraph, layout = "dendrogram", circular = TRUE) +
  geom_edge_diagonal(width = 0.3) +
  geom_node_text(
    aes(
        x = x * (1.1 + arcbarmax),
        y = y * (1.1 + arcbarmax),
        filter = leaf,
        angle = angle,
        label = label,
        hjust = hjust),
    size = 2) +
  geom_node_arc_bar(
    aes(
      r0 = 1,
      r = 1 + (percentage_total**mod) * arcbarmax,
      filter = leaf),
    fill = 'gray60',
    linewidth = 0.2) +
  geom_node_arc_bar(
    aes(
      r0 = 1,
      r = 1 + (percentage_train**mod) * arcbarmax,
      filter = leaf),
    fill = 'gray40',
    linewidth = 0.2) +
  geom_node_arc_bar(
    aes(r0 = 1, r = 1 + arcbarmax, filter = leaf),
    colour = 'black',
    linewidth = 0.1) +
  # Add circle to show 50% arcbars
  geom_path(
    data = circles,
    aes(x = x1, y = y1),
    linetype = "dashed",
    linewidth = 0.2) +
  # Add circle to show 100% arcbars
  geom_path(
    data = circles,
    aes(x = x2, y = y2),
    linetype = "dashed",
    linewidth = 0.2) +
  # Add circle to show 25% arcbars
  geom_path(
    data = circles,
    aes(x = x3, y = y3),
    linetype = "dashed",
    linewidth = 0.2) +
  # Add circle to show order groups
  # geom_path(
  #   data = circles,
  #   aes(x = x5, y = y5),
  #   linewidth = 0.2) +
  # Add text
  geom_node_text(
    aes(x = 0, y = (1 + arcbarmax/(max_species/(max_species/1))**mod)),
    label = as.character(max_species), size = 3) +
  geom_node_text(
    aes(x = 0, y = (1 + arcbarmax/(max_species/(max_species/10))**mod)),
    label = as.character(max_species/10), size = 3) +
  geom_node_text(
    aes(x = 0, y = (1 + arcbarmax/(max_species/(max_species/100))**mod)),
    label = as.character(max_species/100), size = 3) +
  # geom_node_text(
  #   aes(x = 0, y = (1 + arcbarmax/(max_species/(max_species/1000))**mod)),
  #   label = as.character(max_species/1000), size = 3) +
  geom_node_text(
    aes(x = 0, y = 1.0),
    label = "0", size = 3) +
  theme_void() +
  expand_limits(
    x = c(-(3 + arcbarmax), 3 + arcbarmax),
    y = c(-(3 + arcbarmax), 3 + arcbarmax)))

# Add phylopic per order
phylopic_uuid <- list(
  "aves accipitriformes" = "b1c1370b-08cf-4c5e-b390-8fd552f60689",
  "aves anseriformes" = "5e338b5b-0f48-4bf6-b737-90b12876a49b",
  "aves charadriiformes" = "145b43ea-a9d7-4a98-8764-2ebefd6cefe8",
  "aves ciconiiformes" = "8691fc60-e7f2-4a69-af65-08461defac6b",
  "aves columbiformes" = "cbe76bdb-5a89-4577-8472-6af7c2052d70",
  "aves coraciiformes" = "611eeb9e-e167-44b5-bebd-8675d0831d37",
  "aves gruiformes" = "cd5ca1a2-b0a9-4ce9-b163-5c004e327d9d",
  "aves passeriformes" = "70a769d7-ae01-495e-a56d-7889cafdeb88",
  "aves pelecaniformes" = "2a168f51-4016-4a32-ba72-64a53bee31ca",
  "aves phoenicopteriformes" = "a1244226-f2c2-41dc-b113-f1c6545958ce",
  "aves podicipediformes" = "deba1d91-daa8-40a6-8d48-7a9f295bc662",
  "aves procellariiformes" = "4e4df08e-d11a-4250-869c-44ce8ab72dd7",
  "aves psittaciformes" = "2236b809-b54a-45a2-932f-47b2ca4b8b7e",
  "aves sphenisciformes" = "dcd56e52-c6e0-4831-824f-c65bee8875ca",
  "aves suliformes" = "21a3420c-8d7a-4bd2-8ff7-b89fe7f5add5")

for (i in seq(nrow(vertices))) {
    name <- vertices[i + 1, "to"]
    if (name %in% names(phylopic_uuid)) {
        index <- which(names(phylopic_uuid) == name)
        x_pos <- vertices[i + 1, "phylopic_x"]
        y_pos <- vertices[i + 1, "phylopic_y"]
        dendrogram <- (
            dendrogram +
            add_phylopic(uuid = phylopic_uuid[[name]], x = x_pos, y = y_pos, ysize = 0.4))}
}
# Save plot
ggsave(filename = "figures/f5-species_bias.jpg", plot = dendrogram)

#### Pose bias
pose_counts <- dataset %>%
    group_by(pose, dataset) %>%
    count() %>%
    pivot_wider(names_from = dataset, values_from = n) %>%
    mutate(total = total - train) %>%
    pivot_longer(!pose, names_to = 'dataset', values_to = 'count') %>%
    mutate(dataset = factor(
      dataset,
      levels = c("total", "train"),
      ordered = TRUE))

pose_bias <- ggplot(pose_counts, aes(x = pose, y = count, fill = dataset)) +
    geom_bar(stat = "identity", colour = "black") +
    labs(x = "Behaviour", y = "Count", fill = "Dataset") +
    scale_fill_manual(values = c("gray60", "gray40")) +
    theme_minimal() +
    theme(
        axis.title = element_text(size = 30),
        axis.text = element_text(size = 30, colour = "black"),
        legend.position = "none",
        plot.margin = margin(30, 0, 0, 10))

ggsave(filename = "figures/f6b-pose_bias.jpg", plot = pose_bias)

#### Age bias
age_counts <- dataset %>%
    group_by(age, dataset) %>%
    count() %>%
    pivot_wider(names_from = dataset, values_from = n) %>%
    mutate(total = total - train) %>%
    pivot_longer(!age, names_to = 'dataset', values_to = 'count')

age_bias <- ggplot(age_counts, aes(x = age, y = count, fill = dataset)) +
    geom_bar(stat = "identity", colour = "black") +
    labs(x = "Age", y = "Count", fill = "Dataset") +
    scale_fill_manual(values = c("gray60", "gray40")) +
    theme_minimal() +
    theme(
        axis.title = element_text(size = 30),
        axis.text = element_text(size = 30, colour = "black"),
        legend.position = "none",
        plot.margin = margin(0, 0, 0, 10))

ggsave(filename = "figures/f6d-age_bias.jpg", plot = age_bias)

#### GSD bias
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

ggsave(filename = "figures/f6c-gsd_bias.jpg", plot = gsd_bias)

#### Location bias
# Reverse geocode to get country for each latitude and longitude
location_data <- dataset_raw %>%
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
      TRUE ~ country)) %>%
  filter(
    species != "background",
    species != "unknown",
    overlap >= 0.7,
    obscured == 'no',
    dataset != "test",
    dataset != "validation")

# Filter and count instances per country
country_count <- location_data %>%
  group_by(country, dataset) %>%
  summarise(count = n(), .groups = "drop")

# Calculate mean latitude and longitude per country
country_center <- location_data %>%
  group_by(country) %>%
  summarise(
    mean_latitude = mean(latitude),
    mean_longitude = mean(longitude),
    .groups = "drop")

country_data <- merge(country_count, country_center, by = "country")

# Get world map data
world_map <- map_data("world")

# Plot world map
location_bias <- ggplot() +
  geom_polygon(
    data = world_map,
    aes(x = long, y = lat, group = group),
    colour = "gray85",
    fill = "gray80") +
  coord_fixed(1.3) +
  geom_point(
    data = country_data,
    aes(x = mean_longitude, y = mean_latitude, size = count, colour = dataset)) +
  scale_colour_manual(values = c('gray40', 'gray60'), name = 'Dataset') +
  scale_size_continuous(
    range = c(3, 15),
    breaks = c(1250, 5000, 10000),
    name = "Count") +
  theme_void() +
  theme(legend.position = c(0.15, 0.4)) +
  guides(colour = guide_legend(override.aes = list(size=10)))

# Save the plot
ggsave(filename = "figures/f6a-location_bias.jpg", plot = location_bias)

#### Number of images
n_images <- dataset_raw %>%
    filter(dataset == "total") %>%
    distinct(imagepath) %>%
    nrow()

#### Number of instances
n_instances <- dataset_raw %>%
    filter(label != "background", dataset == "total") %>%
    distinct(imagepath, points) %>%
    nrow()

#### Number of species
n_species <- dataset_raw %>%
    filter(label != "background", dataset == "total") %>%
    mutate(species = paste(genus, species)) %>%
    distinct(species) %>%
    nrow()

#### Images camera
images_per_camera <- dataset_raw %>%
    filter(dataset == "total") %>%
    distinct(imagepath, camera) %>%
    group_by(camera) %>%
    summarise(count = n())
  
### images per species
class_bias_data <- dataset %>%
  group_by(dataset, order, commonname, age) %>%
  count() %>%
  pivot_wider(names_from = dataset, values_from = n) %>%
  filter(total >= 10) %>%
  mutate(
    total = log(total),
    train = log(train)) %>%
  mutate(total = total - train, class = paste0(commonname, " - ", age)) %>%
  ungroup() %>%
  select(-commonname, -age) %>%
  pivot_longer(!class, names_to = 'dataset', values_to = 'count')

options(scipen = 999)
breaks <- log(c(5, 50, 500, 5000))
labels <- c(5, 50, 500, 5000)

class_bias <- ggplot(class_bias_data, aes(x = class, y = count, fill = dataset)) +
    geom_bar(stat = "identity", colour = "black") +
    labs(x = "", y = "Count", fill = "Dataset") +
    scale_fill_manual(values = c("gray60", "gray40")) +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    scale_y_continuous(breaks = breaks, labels = labels)
ggsave(filename = "figures/f6a-class-bias-simple.jpg", plot = class_bias, height = 5, width = 10)

class_order <- c(unique(class_bias_data$class))
performance$class <- factor(performance$class, levels = class_order, ordered = TRUE)
