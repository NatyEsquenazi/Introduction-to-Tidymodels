# Script de clasificación---------------------------------

# Ingreso librerias-------------------------------------

library(tidyverse)
library(tidymodels) 
library(palmerpenguins)

# Preprocesamiento---------------------------------
# Elimino columnas que no son numericas
penguins <- palmerpenguins::penguins %>%
  drop_na() %>% #elimino valores perdidos
  select(-year,-sex, -island) #elimino columnas q no son numéricas
glimpse(penguins) #observo variables restantes


# Hago split de datos---------------------------------------
set.seed(123) #setear la semilla
p_split <- penguins %>%
  initial_split(prop=0.75) # divido en 75%

p_train <- training(p_split)
p_split

# Para hacer validación cruzada estratificada
p_folds <- vfold_cv(p_train, strata = species) 


# Creo la receta----------------------------------------
recipe_dt <- p_train %>%
  recipe(species~.) %>%
  step_corr(all_predictors()) %>% #elimino las correlaciones
  step_center(all_predictors(), -all_outcomes()) %>% #centrado
  step_scale(all_predictors(), -all_outcomes()) %>% #escalado
  prep() 
recipe_dt #ver la receta


# Especifico el modelo-------------------------------- 
set.seed(123)
vanilla_tree_spec <- decision_tree() %>% #arboles de decisión
  set_engine("rpart") %>% #librería rpart
  set_mode("classification") #modo para clasificar
vanilla_tree_spec


# Armo el workflow-------------------------------------
tree_wf <- workflow() %>%
  add_recipe(recipe_dt) %>% #agrego la receta
  add_model(vanilla_tree_spec) #agrego el modelo
tree_wf 


# Modelo vanilla sin tunning----------------------------------
set.seed(123) 
vanilla_tree_spec %>% 
  fit_resamples(species ~ ., 
                resamples = p_folds) %>% 
  collect_metrics() #desanidar las metricas

# Especificar 2 hiperparametros-------------------------
set.seed(123) 
trees_spec <- decision_tree() %>% 
  set_engine("rpart") %>% 
  set_mode("classification") %>% 
  set_args(min_n = 20, cost_complexity = 0.1) # especifico hiperparámetros

trees_spec %>%
  fit_resamples(species ~ ., 
                resamples = p_folds) %>% 
  collect_metrics()

# Ejercicios----------------------------------------
# 1. ¿Por qué es el mismo valor obtenido en los dos casos?----
# 2. Dejando fijo el valor de min_n=20, pruebe C=1, C=0.5 y C=0.-----
## Primer caso min_n = 20 y C = 1
set.seed(123) 
trees_spec_20_1 <- decision_tree() %>% 
  set_engine("rpart") %>% 
  set_mode("classification") %>% 
  set_args(min_n = 20, cost_complexity = 1) #especifico hiperparámetros

trees_spec_20_1 %>%
  fit_resamples(species ~ ., 
                resamples = p_folds) %>% 
  collect_metrics()

# 3. Dejando fijo el valor de C=0, pruebe min_n 1 y 5.------------


# Paso mas importante---------------------------------------------
# Prediccion del modelo------------------------
# Utilizamos la funcion last_fit junto al workflow y al split de datos

final_fit_dt <- last_fit(tree_wf,
                         split = p_split
)
final_fit_dt %>%
  collect_metrics()

# Matriz de confusion------------------------------------
final_fit_dt %>%
  collect_predictions() %>%
  conf_mat(species, .pred_class) #para ver la matriz de confusión


# sensibilidad---------------------------------------------

final_fit_dt %>%
  collect_predictions() %>%
  sens(species, .pred_class) 

# Sensibilidad global del modelo



# Ejercicio 2----------------------------------------
# 1. Repetir estos pasos para el modelo de C=0 y min_n=5


# Random Forest-----------------------------------
# Paso 1: dividimos los datos en el paso anterior
# Paso 2: pequeño pre-procesamiento----------------

p_recipe <- training(p_split) %>%
  recipe(species~.) %>%
  step_corr(all_predictors()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep()
p_recipe

# Paso 3: especificamos el modelo---------------------------
rf_spec <- rand_forest() %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

# Paso 4: entrenamiento de la funcion----------------------
# Veamos cómo funciona sin tunning
set.seed(123)
rf_spec %>% 
  fit_resamples(species ~ ., 
                resamples = p_folds) %>% 
  collect_metrics()
# Mtry = 2-------------------------------------
rf2_spec <- rf_spec %>% 
  set_args(mtry = 2)
set.seed(123)
rf2_spec %>% 
  fit_resamples(species ~ ., 
                resamples = p_folds) %>% 
  collect_metrics()

# Mtry = 3------------------------------
rf3_spec <- rf_spec %>% 
  set_args(mtry = 3)
set.seed(123)
rf3_spec %>% 
  fit_resamples(species ~ ., 
                resamples = p_folds) %>% 
  collect_metrics()

# Mtry = 4-----------------------------------
rf4_spec <- rf_spec %>% 
  set_args(mtry = 4)
set.seed(123)
rf4_spec %>% 
  fit_resamples(species ~ ., 
                resamples = p_folds) %>% 
  collect_metrics()

# Tuneo automatico------------------------
tune_spec <- rand_forest(
  mtry = tune(),
  trees = 1000,
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")
tune_spec

# Workflows--------------------------------
tune_wf <- workflow() %>%
  add_recipe(p_recipe) %>%
  add_model(tune_spec)
set.seed(123)
cv_folds <- vfold_cv(p_train, strata = species)
tune_wf


# Paralelizamos----------------------------------

#doParallel::registerDoParallel()
# esto puede demorar mucho en correr 
set.seed(123)
tune_res <- tune_grid(
  tune_wf,
  resamples = cv_folds,
  grid = 20
)
tune_res


# Elegimos el mejor modelos------------------
best_auc <- select_best(tune_res, "roc_auc")
final_rf <- finalize_model(
  tune_spec,
  best_auc
)
final_rf

# Valores finales-------------------
set.seed(123)
final_wf <- workflow() %>%
  add_recipe(p_recipe) %>%
  add_model(final_rf)
final_res <- final_wf %>%
  last_fit(p_split)
final_res %>%
  collect_metrics()

