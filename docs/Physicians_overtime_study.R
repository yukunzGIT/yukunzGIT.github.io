################################################################################
# Filename: Physicians_overtime_study.R
# Author: Yukun (Edward) Zhang
# Date Created: 2016-05-17
# Last Updated: 2021-03-01
# Description: This script performs an overtime work analysis for ER physicians.
################################################################################

library(dplyr)
library(lubridate)
library(ggplot2)
library(MASS)
library(broom)

# Read the data
df <- read.csv("ucsd_patient_data.txt")

# Rename columns and calculate the length of stay (LOS) for patients
df <- df %>%
  rename(patient_arrive_time=ed_tc, patient_discharge_time=dcord_tc, log_LOS=xb_lntdc) %>%
  mutate(patient_arrive_time = as.POSIXct(patient_arrive_time, format="%d%b%Y %H:%M:%S"), 
         patient_discharge_time = as.POSIXct(patient_discharge_time, format="%d%b%Y %H:%M:%S"),
         LOS = as.numeric(difftime(patient_discharge_time, patient_arrive_time, units = "secs"))/3600)

# Extract the date
df$shift_schedule <- gsub("noon", "12 p.m.", df$shiftid)
df$shift_schedule <- gsub("\\.", "", df$shift_schedule)

df$date <- gsub("^(\\d{2}[a-z]{3}\\d{4}).*", "\\1", df$shift_schedule)

# Extract the date
df$date <- gsub("^(\\d{2}[a-z]{3}\\d{4}).*", "\\1", df$shift_schedule)

# Extract the starting time
df$start_time <- gsub("^\\d{2}[a-z]{3}\\d{4} (.*?) to .*", "\\1", df$shift_schedule)

# Extract the ending time
df$end_time <- gsub(".* to ([0-9]+ [a\\.m\\.p\\.M\\.]+)$", "\\1", df$shift_schedule)

# Convert start and end times to standard datetime format
df$shift_start_time <- as.POSIXct(paste(df$date, df$start_time), format="%d%b%Y %I %p")
df$shift_end_time <- as.POSIXct(paste(df$date, df$end_time), format="%d%b%Y %I %p")

# Check if the physician worked overtime or not
df$over_time <- as.numeric(difftime(df$patient_discharge_time, df$shift_end_time, units = "secs")) > 0
sum(df$over_time)/length(df$over_time)

# Severity Pattern Plot
ggplot(df, aes(x=as.integer(format(patient_arrive_time, "%H")), y=log_LOS)) +
  stat_summary(geom = "line", fun = mean) +
  scale_x_continuous(name = "Hour of the Day", breaks = 0:23) +
  labs(title="Hourly Average Patient Severity Pattern",
       x="Hours in a day", y="Average expected log length of stay")

df$hour_of_day <- hour(patient_arrive_time)
df$busy_hour <- df$hour_of_day >= 4 & df$hour_of_day <= 17

# Distribution of Patient Severity by Hour
ggplot(df, aes(x = as.factor(hour_of_day), y = log_LOS)) +
  geom_boxplot() +
  labs(x = "Hour of the Day", y = "Expected Log Length of Stay", title = "Distribution of Patient Severity by Hour") +
  theme_minimal()

anova_result <- aov(log_LOS ~ as.factor(hour_of_day), data = df)
summary(anova_result)
tidy(TukeyHSD(anova_result)) %>% filter( adj.p.value < 0.05)

# Patient Arrival Pattern Plot
counts <- df %>%
  mutate(hour = hour(patient_arrive_time)) %>%
  group_by(hour) %>%
  summarize(count = n())

ggplot(counts, aes(x = hour, y = count)) +
  geom_bar(stat = "identity") +
  scale_x_continuous(name = "Hour of the Day", breaks = 0:23) +
  labs(title="Hourly Patient Arrival Pattern",
       x="Hours in a day", y="Patient arrival counts")

# Binary Logistic Regression (BLR) Modeling
cor(df$LOS, df$busy_hour)

# Baseline BLR
binary_log_model1 <- glm(as.factor(over_time) ~ LOS + busy_hour,
                         data = df, family = binomial)
tidy(binary_log_model1, conf.int = TRUE, exponentiate = TRUE) %>% mutate_if(is.numeric, round, 2)
glance(binary_log_model1)

# Full BLR
binary_log_model2 <- glm(as.factor(over_time) ~ LOS + busy_hour + patient_age + physician_age,
                         data = df, family = binomial)
tidy(binary_log_model2, conf.int = TRUE, exponentiate = TRUE) %>% mutate_if(is.numeric, round, 2)
glance(binary_log_model2)

