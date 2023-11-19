##########################################################################################
# Filename: Labor_mobility_study.R
# Author: Yukun (Edward) Zhang
# Email: ykzhang1211@g.ucla.edu
# Date Created: 2018-11-23
# Last Updated: 2021-05-02
# Description: This script performs an analysis of the US labor mobility patterns on wages. 
##########################################################################################

library(plm)
library(knitr)
library(dplyr)
library(ggplot2)
library(broom)
knitr::opts_chunk$set(echo = TRUE)

mydata <- read.csv("wbis-nlsy79-extended.csv")       # Urban takes value from 0-2; Region takes value from 1 to 4;

df <- plm.data(mydata, index = c("i", "year"))     # set panel data
summary(df)

# Create new variables
df$employment <- df$wage > 0
df$urban[df$urban == 2] <- 1
df$age <- as.numeric(as.character(df$year)) - (df$birth + 1900)

# Create lag columns to detect moves
df <- df[!is.na(df$region) & !is.na(df$urban) & !is.na(df$wage), ] %>% group_by(i) %>% arrange(year) %>%
  # Create lagged versions of the region column and the urban column to compare year t with year t-1 for each individual
  mutate(lag_region = lag(region),
         lag_urban = lag(urban),
         # Create columns to track region moves and urban moves
         region_move = if_else(!is.na(lag_region) & region != lag_region, 1, 0),
         urban_move = if_else(!is.na(lag_urban) & urban != lag_urban, 1, 0))

# Report the mean wages, employment and education in each region.
by_region <- as.data.frame(aggregate(df[, c("wage", "employment", "educ")], by=list(df$region), FUN=mean)) %>%
  rename(regions= Group.1, education = educ) %>%
  round(2)
kable(by_region, caption = "The Mean of Wages, Employment and Education Across Regions", align = 'c')

# Report the mean wages, employment and education attainment in urban/non-urban areas.
by_urban <- aggregate(df[, c("wage", "employment", "educ")], by=list(df$urban), FUN=mean) %>%
  rename(urban= Group.1, education = educ) %>%
  round(2)
kable(by_urban, caption = "The Mean of Wages, Employment and Education for Urban/Non-urban", align = 'c')

# Build a transition matrix using the table function
region_transition_matrix <- table(df$lag_region[df$region_move == 1], df$region[df$region_move == 1])

# Give appropriate names to the rows and columns of the transition matrix
rownames(region_transition_matrix) <- c("Region 1", "Region 2", "Region 3", "Region 4")
colnames(region_transition_matrix) <- c("Region 1", "Region 2", "Region 3", "Region 4")

# Print the transition matrix
kable(region_transition_matrix, caption = "Counts of Moves Across 4 US Regions", align = 'c')

# Build a transition matrix using the table function for urban moves
urban_transition_matrix <- table(df$lag_urban[df$urban_move == 1], df$urban[df$urban_move == 1], dnn = c("From", "To"))

# Give appropriate names to the rows and columns of the transition matrix
rownames(urban_transition_matrix) <- c("Non-Urban", "Urban")
colnames(urban_transition_matrix) <- c("Non-Urban", "Urban")

# Print the transition matrix
kable(urban_transition_matrix, caption = "Counts of Moves for Urban/Non-urban Areas", align = 'c')

# Create an event time column
calculate_event_time <- function(move_col) {
  event_time <- rep(NA, length(move_col))
  indices <- which(move_col == 1)
  for (index in indices) {
    if (index > 2 && index + 2 <= length(move_col)) {
      event_time[(index - 2):(index + 2)] <- -2:2
    }
  }
  return(event_time)
}

df <- df %>%
  mutate(region_event_time = calculate_event_time(region_move),
         urban_event_time = calculate_event_time(urban_move))

# Calculate mean wages for each event time
region_event_study <- df %>% filter(!is.na(region_event_time)) %>%
  group_by(region_event_time) %>%
  summarize(mean_wage = mean(wage, na.rm = TRUE))

urban_event_study <- df %>% filter(!is.na(urban_event_time)) %>%
  group_by(urban_event_time) %>%
  summarize(mean_wage = mean(wage, na.rm = TRUE))

# Plot the results
ggplot(region_event_study, aes(x=region_event_time, y=mean_wage)) +
  geom_line() + 
  labs(title="Event Study for Region Moves",
       x="Years around the Move", y="Mean Wage")

ggplot(urban_event_study, aes(x=urban_event_time, y=mean_wage)) +
  geom_line() + 
  labs(title="Event Study for Urban Moves",
       x="Years around the Move", y="Mean Wage")

# OLS modeling
# simple OLS model
model1 <- lm(wage ~ region_move + urban_move + age, data=df)
tidy(model1) %>%
  mutate_if(is.numeric, round, digits = 2)
glance(model1)

# more complex OLS model
model2 <- lm(wage ~ region_move + urban_move + age + educ + gender + employment, data=df)
tidy(model2) %>%
  mutate_if(is.numeric, round, digits = 2)
glance(model2)

# focus on the effect of urban_move with isolation
model3 <- lm(wage ~ urban_move + age + educ + gender + employment, data=df %>% filter(region_event_time == 0))
tidy(model3) %>%
  mutate_if(is.numeric, round, digits = 2)
glance(model3)

# focus on the effect of region_move with isolation
model4 <- lm(wage ~ region_move + age + educ + gender + employment, data=df %>% filter(urban_event_time == 0))
tidy(model4) %>%
  mutate_if(is.numeric, round, digits = 2)
glance(model4)

