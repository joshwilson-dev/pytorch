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
root = "models/bird-2024_04_14/data.xlsx"
dataset_raw <- read_excel(
    path = root,
    sheet = "ST1-dataset")

#### Species Bias
# Process data
dataset <- dataset_raw %>%
    filter(
      overlap >= 0.75,
      obscured == 'no',
      species != 'unknown',
      order != "background") %>%
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
    mutate(group = "aves") %>%
    rename(from = class, to = order)

# Add links between order and family
family <- dataset %>%
    group_by(order, family, dataset) %>%
    summarise(count = n(), .groups = "drop") %>%
    pivot_wider(names_from = dataset, values_from = count, names_prefix = "count_") %>%
    mutate(group = family) %>%
    rename(from = order, to = family)

# Add links between family and genus
genus <- dataset %>%
    group_by(family, genus, dataset) %>%
    summarise(count = n(), .groups = "drop") %>%
    pivot_wider(names_from = dataset, values_from = count, names_prefix = "count_") %>%
    mutate(group = family) %>%
    rename(from = family, to = genus)

# Add links between genus and species
species <- dataset %>%
    group_by(family, genus, species, dataset) %>%
    summarise(count = n(), .groups = "drop") %>%
    pivot_wider(names_from = dataset, values_from = count, names_prefix = "count_") %>%
    rename(group = family, from = genus, to = species)
max_species = max(species$count_total)

# Create the count for the leaves
count <- distinct(rbind(order, family, genus, species)) %>%
    mutate(
        count_test = case_when(is.na(count_test) ~ 0, T ~ count_test),
        count_train = case_when(is.na(count_train) ~ 0, T ~ count_train),
        count_total = case_when(is.na(count_total) ~ 0, T ~ count_total),
        count_total_per = count_total/max_species,
        count_test_per = count_test/max_species,
        count_train_per = (count_test + count_train)/max_species)

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
# Calculate the number of families in each order
groupsize <- dataset %>%
  group_by(family) %>%
  summarize(groupsize = n_distinct(species)) %>%
  rename(to = family)

# Determine angle of each order group, convert to x and y to paste phylopic
vertices <- vertices %>%
  merge(., groupsize, by = "to", all = TRUE) %>%
  arrange(sort) %>%
  mutate(groupsize_lag = lag(groupsize, default = 0)) %>%
  mutate(csum_gsl = cumsum(ifelse(is.na(groupsize_lag), 0, groupsize_lag))) %>%
  mutate(phylopic_angle = (csum_gsl + groupsize / 2) * 2 * pi / n_leaves) %>%
  mutate(phylopic_x = 0.9 * sin(phylopic_angle)) %>%
  mutate(phylopic_y = 0.9 * cos(phylopic_angle))

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
  y4 = sin(seq(0, 2 * pi, length.out = 100)) * (1 + arcbarmax/(max_species/(max_species/1000))**mod))

# Create a graph object
mygraph <- graph_from_data_frame(edges, vertices = vertices)

# Make the plot
dendrogram <- (
  ggraph(mygraph, layout = "dendrogram", circular = TRUE) +
  geom_edge_diagonal(colour = "black") +
  geom_node_text(
    aes(
        x = x * (1.05 + arcbarmax),
        y = y * (1.05 + arcbarmax),
        filter = leaf,
        angle = angle,
        label = label,
        hjust = hjust),
    size = 4) +
  geom_node_arc_bar(
    aes(r0 = 1, r = 1 + (count_total_per**mod) * arcbarmax, filter = leaf),
    fill = 'gray80',
    linewidth = 0.2) +
  geom_node_arc_bar(
    aes(r0 = 1, r = 1 + (count_train_per**mod) * arcbarmax, filter = leaf),
    fill = 'gray60',
    linewidth = 0.2) +
  geom_node_arc_bar(
    aes(r0 = 1, r = 1 + (count_test_per**mod) * arcbarmax, filter = leaf),
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
  # Add circle to show 25% arcbars
  geom_path(
    data = circles,
    aes(x = x4, y = y4),
    linetype = "dashed",
    linewidth = 0.2) +
  geom_node_text(
    aes(x = 0, y = (1 + arcbarmax/(max_species/(max_species/1))**mod)),
    label = as.character(max_species), size = 4) +
  geom_node_text(
    aes(x = 0, y = (1 + arcbarmax/(max_species/(max_species/10))**mod)),
    label = as.character(max_species/10), size = 4) +
  geom_node_text(
    aes(x = 0, y = (1 + arcbarmax/(max_species/(max_species/100))**mod)),
    label = as.character(max_species/100), size = 4) +
  geom_node_text(
    aes(x = 0, y = (1 + arcbarmax/(max_species/(max_species/1000))**mod)),
    label = as.character(max_species/1000), size = 4) +
  geom_node_text(
    aes(x = 0, y = 1.0),
    label = "0", size = 4) +
  theme_void() +
  expand_limits(
    x = c(-(1.7 + arcbarmax), 1.7 + arcbarmax),
    y = c(-(1.7 + arcbarmax), 1.7 + arcbarmax)))

# phylopic_uuid <- list(
# "aves accipitriformes accipitridae" = "93e8bd48-f8e0-457a-8551-670f2684b7f0",
# "aves anseriformes anatidae" = "cf522e02-35cc-44f5-841c-0e642987c2e4",
# "aves anseriformes anseranatidae" = "e470e9bc-869f-46f7-885a-cac1dc724b32",
# "aves charadriiformes charadriidae" = "69575d6b-4e9e-42d4-a4fc-2888b87384c8",
# "aves charadriiformes haematopodidae" = "65ed3e2d-38f5-421f-976c-be4df6ac73fa",
# "aves charadriiformes jacanidae" = "bf5fe2c5-1247-4ed9-93e2-d5af255ec462",
# "aves charadriiformes laridae" = "d7c59a44-774f-4191-8623-07c77ca77851",
# "aves charadriiformes recurvirostridae" = "1c9fb51b-5615-40fb-a2a8-f201fd61ac02",
# "aves charadriiformes scolopacidae" = "1c9fb51b-5615-40fb-a2a8-f201fd61ac02",
# "aves charadriiformes stercorariidae" = "b828fa30-3190-4198-b304-d6599c1e2ae1",
# "aves ciconiiformes ciconiidae" = "b415196d-355a-408d-a73b-1567c87a1721",
# "aves columbiformes columbidae" = "3644ff17-3cd1-4c98-afdb-91113d4e2cca",
# "aves coraciiformes meropidae" = "086d2b94-f13c-47ab-9538-c9b71d97f934",
# "aves gruiformes gruidae" = "7f02b605-c87b-4ec2-9e14-011f813c23a4",
# "aves gruiformes rallidae" = "d5194798-0b50-4a79-8b3d-e6efe468c5ab",
# "aves passeriformes artamidae" = "ac6920cc-61a3-4d33-86b6-e8da07011f87",
# "aves passeriformes corvidae" = "2db31c7c-b0a9-460f-807e-da9181f21cf6",
# "aves passeriformes hirundinidae" = "cf38be70-f0a8-4a47-b387-e797ec16c7c1",
# "aves passeriformes meliphagidae" = "d42be981-b1ba-494a-8047-7ec93d0db31d",
# "aves passeriformes monarchidae" = "87d6a89a-9044-49cd-b708-086c847735f2",
# "aves pelecaniformes ardeidae" = "a2a64845-7c7b-4e4a-b1f5-8fd03b673796",
# "aves pelecaniformes pelecanidae" = "2a168f51-4016-4a32-ba72-64a53bee31ca",
# "aves pelecaniformes threskiornithidae" = "ad11bfb7-4ab8-47d2-8e25-0e11ab947e20",
# "aves phoenicopteriformes phoenicopteridae" = "a1244226-f2c2-41dc-b113-f1c6545958ce",
# "aves podicipediformes podicipedidae" = "deba1d91-daa8-40a6-8d48-7a9f295bc662",
# "aves procellariiformes diomedeidae" = "4e4df08e-d11a-4250-869c-44ce8ab72dd7",
# "aves psittaciformes cacatuidae" = "2236b809-b54a-45a2-932f-47b2ca4b8b7e",
# "aves sphenisciformes spheniscidae" = "dcd56e52-c6e0-4831-824f-c65bee8875ca",
# "aves suliformes anhingidae" = "c8d80b8f-a6b3-4d4f-9f2d-ca6bea033d2c",
# "aves suliformes phalacrocoracidae" = "21a3420c-8d7a-4bd2-8ff7-b89fe7f5add5"
# )

# Add phylopics
# for (i in seq(nrow(vertices))) {
#     name <- vertices[i + 1, "to"]
#     if (name %in% names(phylopic_uuid)) {
#         index <- which(names(phylopic_uuid) == name)
#         x_pos <- vertices[i + 1, "phylopic_x"]
#         y_pos <- vertices[i + 1, "phylopic_y"]
#         dendrogram <- (
#             dendrogram +
#             add_phylopic(uuid = phylopic_uuid[[name]], x = x_pos, y = y_pos, ysize = 0.1))}
# }

# Save plot
ggsave(filename = "figures/f5a-species_bias.jpg", plot = dendrogram)

#### Pose bias
pose_counts <- dataset %>%
    group_by(pose, dataset) %>%
    count() %>%
    pivot_wider(names_from = dataset, values_from = n) %>%
    mutate(total = total - train - test) %>%
    pivot_longer(!pose, names_to = 'dataset', values_to = 'count') %>%
    mutate(dataset = factor(
      dataset,
      levels = c("total", "test", "train"),
      ordered = TRUE))

pose_bias <- ggplot(pose_counts, aes(x = pose, y = count, fill = dataset)) +
    geom_bar(stat = "identity", colour = "black") +
    labs(x = "Pose", y = "Count", fill = "Dataset") +
    scale_fill_manual(values = c("gray80", "gray60", "gray40")) +
    theme_minimal() +
    theme(
        axis.title = element_text(size = 30),
        axis.text = element_text(size = 30, colour = "black"),
        legend.position = "none",
        plot.margin = margin(30, 0, 0, 10))

ggsave(filename = "figures/f6b-pose_bias.jpg", plot = pose_bias)

#### Age bias
age_counts <- dataset_raw %>%
    filter(
        age != "unknown",
        label != "background",
        dataset != "test") %>%
    group_by(age, dataset) %>%
    count()

age_bias <- ggplot(age_counts, aes(x = age, y = n, fill = dataset)) +
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
gsd_counts <- dataset_raw %>%
    mutate(
      gsd = gsd * 1000,
      gsd_cats = cut(gsd, breaks = gsd_breaks)) %>%
    filter(
        label != "background",
        dataset != "test") %>%
    group_by(gsd_cats, dataset) %>%
    count()

gsd_bias <- ggplot(gsd_counts, aes(x = gsd_cats, y = n, fill = dataset)) +
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
    country = map.where(database = "world", longitude, latitude),
    country = case_when(
      location == "-62.0, -59.0" ~ "Antarctica",
      location == "-51.0, -61.0" ~ "Falkland Islands",
      location == "-28.0, 153.0" ~ "Australia",
      location == "52.0, 143.0" ~ "Russia",
      location == "54.0, 14.0" ~ "Poland",
      location == "56.0, 8.0" ~ "Denmark",
      location == "-27.0, 153.0" ~ "Australia",
      location == "59.0, -140.0" ~ "Alaska",
      location == "-63.0, -61.0" ~ "Antarctica",
      TRUE ~ country)) %>%
  filter(label != "background", dataset != "test")

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
  scale_colour_manual(values = c('gray40', 'gray60')) +
  # scale_size(range = c(5, 15)) +
  scale_size_continuous(
    range = c(3, 20),
    breaks = c(1250, 5000, 20000)) +
  theme_void()

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