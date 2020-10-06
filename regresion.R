# Script de regresion----------------------------------
# Instalar paquetes--------------------------------------
#install.packages("hrbrthemes")
#install.packages("ggthemes")
#install.packages("ranger")

# Cargo librerias----------------------------------------------
library(tidyverse)
library(hrbrthemes)



# ingreso los datos----------------------------------------------

attendance <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-04/attendance.csv")
standings <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-04/standings.csv")


# Observo los datos-----------------------------------------------
attendance %>% glimpse()
standings %>%  glimpse()

# Joins-------------------------------------------------------
attendance_joined <- attendance %>%
  left_join(standings,
            by = c("year", "team_name", "team")
  )
attendance_joined


attendance_joined %>%
  filter(!is.na(weekly_attendance)) %>%
  ggplot(aes(fct_reorder(team_name, weekly_attendance),
             weekly_attendance,
             fill = playoffs
  )) +
  geom_boxplot(outlier.alpha = 0.5) +
  coord_flip() +
  labs(
    fill = NULL, x = NULL,
    y = "Asistencia semanal de juegos de la NFL"
  )


attendance_joined %>%
  distinct(team_name, year, margin_of_victory, playoffs) %>%
  ggplot(aes(margin_of_victory, fill = playoffs)) +
  geom_histogram(position = "identity", alpha = 0.7) +
  labs(
    x = "Margien de victoria",
    y = "Númmero de equipos",
    fill = NULL
  )


attendance_joined %>%
  mutate(week = factor(week)) %>%
  ggplot(aes(week, weekly_attendance, fill = week)) +
  geom_boxplot(show.legend = FALSE, outlier.alpha = 0.5) +
  labs(
    x = "Semana de la temporada de la NFL",
    y = "Asistencia semanal de juegos de la NFL"
  )


attendance_df <- attendance_joined %>%
  filter(!is.na(weekly_attendance)) %>%
  select(
    weekly_attendance, team_name, year, week,
    margin_of_victory, strength_of_schedule, playoffs
  )
attendance_df

# Inicia la modelización para regresión-----------------

library(tidymodels)
set.seed(1234)
attendance_split <- attendance_df %>%
  initial_split(strata = playoffs)
nfl_train <- training(attendance_split)
nfl_test <- testing(attendance_split)

lm_spec <- linear_reg() %>%
  set_engine(engine = "lm")
lm_spec



lm_fit <- lm_spec %>%
  fit(weekly_attendance ~ .,
      data = nfl_train
  )
lm_fit


rf_spec <- rand_forest(mode = "regression") %>%
  set_engine("ranger")
rf_spec


library('ranger')
rf_fit <- rf_spec %>%
  fit(weekly_attendance ~ ., data = nfl_train)
rf_fit


results_train <- lm_fit %>%
  predict(new_data = nfl_train) %>%
  mutate(
    truth = nfl_train$weekly_attendance,
    model = "lm"
  ) %>%
  bind_rows(rf_fit %>%
              predict(new_data = nfl_train) %>%
              mutate(
                truth = nfl_train$weekly_attendance,
                model = "rf"
              ))
results_test <- lm_fit %>%
  predict(new_data = nfl_test) %>%
  mutate(
    truth = nfl_test$weekly_attendance,
    model = "lm"
  ) %>%
  bind_rows(rf_fit %>%
              predict(new_data = nfl_test) %>%
              mutate(
                truth = nfl_test$weekly_attendance,
                model = "rf"
              ))


results_train %>%
  group_by(model) %>%
  rmse(truth = truth, estimate = .pred)


results_test %>%
  group_by(model) %>%
  rmse(truth = truth, estimate = .pred)


results_test %>%
  mutate(train = "testing") %>%
  bind_rows(results_train %>%
              mutate(train = "training")) %>%
  ggplot(aes(truth, .pred, color = model)) +
  geom_abline(lty = 2, color = "gray80", size = 1.5) +
  geom_point(alpha = 0.5) +
  facet_wrap(~train) +
  labs(
    x = "Valor real",
    y = "Asitencia predicha",
    color = "Tipo de modelo"
  )

