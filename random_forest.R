
Globalpath<- "C:/Users/ioliv/Documents/uu/ssml/Spatial_group_project" #change the path to your folder
setwd(Globalpath)
getwd()

#install.packages("easypackages")
easypackages::packages ("sf", "sp", "tmap", "mapview", "car", "RColorBrewer", "tidyverse", "osmdata", "nngeo", "FNN", "rpart", "rpart.plot", "randomForest", "sessioninfo", "caret", "rattle", "ipred", "tidymodels", "ranger", "recipes", "workflows", "themis","xgboost", "modelStudio", "DALEX", "DALEXtra", "vip", "pdp")

#install.packages("rlang")
library(rlang)
library(Metrics)
library(rpart)
library(rpart.plot)
library(ranger)
library(randomForest)

ref_data <- st_read ("data/final_dataset.csv")
#val_test_data <- st_read ("data/final_dataset.csv")

ref_data <- st_read("data/dataset_geo.gpkg")
mapview::mapview(ref_data, zcol = "incomming_ref")

set.seed(555)

#Prepare dataset for modeling
ref_data$unemployment <- as.numeric(ref_data$unemployment)
ref_data$clean_elections <- as.numeric(ref_data$clean_elections)
ref_data$population <- as.numeric(ref_data$population)
ref_data$gdp_capita <- as.numeric(ref_data$gdp_capita)
ref_data$gini_index <- as.numeric(ref_data$gini_index)
ref_data$incomming_ref <- as.numeric(ref_data$incomming_ref)
ref_data$incomming_ref_target <- as.numeric(ref_data$incomming_ref_target)
ref_data$avg_inc_ref_5y <- as.numeric(ref_data$avg_inc_ref_5y)
ref_data$ratio_inc_ref_5y <- as.numeric(ref_data$ratio_inc_ref_5y)
ref_data$outgoing_ref_neighbors <- as.numeric(ref_data$outgoing_ref_neighbors)
ref_data$outgoing_ref_other_weighted <- as.numeric(ref_data$outgoing_ref_other_weighted)

model_data <- ref_data[,c('clean_elections','population','incomming_ref_target',
                          'gdp_capita','incomming_ref','avg_inc_ref_5y','outgoing_ref_other_weighted',
                          'ratio_inc_ref_5y','outgoing_ref_neighbors','ADMIN','ISO_A3')]

#model_data <- model_data[complete.cases(model_data),]
model_data <- model_data[!sf::st_is_empty(model_data), ] %>% na.omit()

split1<- sample(c(rep(0, 0.75 * nrow(model_data)), rep(1, 0.25 * nrow(model_data))))
train <- model_data[split1 == 0, ] 
test <- model_data[split1== 1, ]

#Decision tree

#Note: We are using, method = "poisson". It can also be "a"nova", "poisson", "class" or "exp". Depending on the data type. In this case pedal is a count data so we selected poisson. 
DT0 <- rpart (incomming_ref_target ~ incomming_ref + clean_elections + gdp_capita +
                population + avg_inc_ref_5y + ratio_inc_ref_5y +
                + outgoing_ref_neighbors + outgoing_ref_other_weighted, data= train,  method  = "anova") 

summary (DT0)
rpart.plot(DT0)

pred_tree <- predict(DT0, test)
testRMSE_tree <- rmse(test$incomming_ref, pred_tree)
testRMSE_tree

test$prediction_tree <- pred_tree

#Random Forest

#fit a random forest model, here I am selecting some hyper parameter such as mtry = 6 you can test different numbers but we shall explore these in details in the following session.
#here the rule of thumb is to use p/3 variables for regression trees, here p is the number of predictors in the model, and we had 19 predictors
rf <-randomForest(incomming_ref_target ~ incomming_ref + clean_elections + gdp_capita +
                    population + avg_inc_ref_5y + ratio_inc_ref_5y
                    + outgoing_ref_neighbors + outgoing_ref_other_weighted, data= train, mtry = 6, ntree = 1000) 

print(rf)
#let us plot the Variable importance from the RF model outcome
vip::vip(rf, num_features = 19, idth = 0.5, aesthetics = list(fill = "purple2"), include_type = T)

pred_rf <- predict(object = rf, newdata = test)

#check the RMSE value for the predicted set
testRMSE_rf <- rmse(test$incomming_ref_target, pred_rf)
testRMSE_rf

test$prediction_rf <- pred_rf

rf2 <- ranger(incomming_ref_target ~ incomming_ref +  clean_elections + gdp_capita +
                population + avg_inc_ref_5y + ratio_inc_ref_5y
                + outgoing_ref_neighbors + outgoing_ref_other_weighted, num.trees = 1000, mtry = 6, data = train)

pred_rf2 <- predict(rf2,test)$prediction

testRMSE_rf2 <- rmse(test$incomming_ref_target, pred_rf2)
testRMSE_rf2

test$prediction_rf2 <- pred_rf2

#create model explainer
#explainer_rf <- DALEX::explain(
#  model = rf,
#  data = train,
#  y = train$incomming_ref,
#  label = "Random Forest",
#  verbose = FALSE
#)

#now make an interactive dashboard
#modelStudio::modelStudio(explainer_rf)

#Linear model

mdl = lm(incomming_ref_target ~  incomming_ref + clean_elections + gdp_capita +
                 population + avg_inc_ref_5y + ratio_inc_ref_5y +
                 + outgoing_ref_neighbors + outgoing_ref_other_weighted, data= train)
summary(mdl)

pred_lm <- predict(object = mdl, 
                   newdata = test)

testRMSE_lm <- rmse(test$incomming_ref_target, pred_lm)
testRMSE_lm

test$prediction_lm <- pred_lm

mdl_glm = glm(incomming_ref_target ~  incomming_ref + clean_elections + gdp_capita +
           population + avg_inc_ref_5y + ratio_inc_ref_5y +
           + outgoing_ref_neighbors + outgoing_ref_other_weighted, data= train, family = 'log')
summary(mdl_glm)

pred_glm <- predict(object = mdl_glm, 
                   newdata = test)

testRMSE_glm <- rmse(test$incomming_ref_target, pred_glm)
testRMSE_glm


ggplot(data = test, aes(x = ADMIN))+
  geom_line(aes(y = incomming_ref_target, group = 1), color = 'red',  linetype="dashed")+
  geom_line(aes(y = pred_lm, group = 1), color = 'green')+
  geom_line(aes(y = pred_rf, group = 1), color = 'blue')

test %>% 
  ggplot(aes(pred_lm, incomming_ref_target)) +
  geom_point(colour = "#ff6767", alpha = 0.3) +
  labs(title = "Predicted and observed") + 
  theme_bw(18)

val_test_data_model<-val_test_data
val_test_data_model$unemployment <- as.numeric(val_test_data_model$unemployment)
val_test_data_model$clean_elections <- as.numeric(val_test_data_model$clean_elections)
val_test_data_model$population <- as.numeric(val_test_data_model$population)
val_test_data_model$gdp_capita <- as.numeric(val_test_data_model$gdp_capita)
val_test_data_model$gini_index <- as.numeric(val_test_data_model$gini_index)
val_test_data_model$incomming_ref <- as.numeric(val_test_data_model$incomming_ref)
val_test_data_model$incomming_ref_target <- as.numeric(val_test_data_model$incomming_ref_target)
val_test_data_model$avg_inc_ref_5y <- as.numeric(val_test_data_model$avg_inc_ref_5y)
val_test_data_model$ratio_inc_ref_5y <- as.numeric(val_test_data_model$ratio_inc_ref_5y)
val_test_data_model$outgoing_ref_neighbors <- as.numeric(val_test_data_model$outgoing_ref_neighbors)

val_test_data_model <- val_test_data_model[,c('clean_elections','population','incomming_ref_target',
                          'gdp_capita','incomming_ref','avg_inc_ref_5y',
                          'ratio_inc_ref_5y','outgoing_ref_neighbors','ADMIN','ISO_A3')]
val_test_data_model <- val_test_data_model[complete.cases(val_test_data_model),]

pred_lm_validation <- predict(object = mdl, newdata = val_test_data_model)
pred_rf_validation <- predict(object = rf, newdata = val_test_data_model)
pred_rf2_validation <- predict(rf2,val_test_data_model)$prediction
rmse(val_test_data_model$incomming_ref_target, pred_lm_validation)
rmse(val_test_data_model$incomming_ref_target, pred_rf_validation)
rmse(val_test_data_model$incomming_ref_target, pred_rf2_validation)

val_test_data_model$prediction_lm <- pred_lm_validation
val_test_data_model$prediction_rf <- pred_rf_validation
val_test_data_model$prediction_rf2 <- pred_rf2_validation

ggplot(data = val_test_data_model, aes(x = ADMIN))+
  geom_line(aes(y = incomming_ref_target, group = 1), color = 'red')+
  geom_line(aes(y = prediction_lm, group = 1), color = 'green')+
  geom_line(aes(y = prediction_rf, group = 1), color = 'blue')+
  geom_line(aes(y = prediction_rf2, group = 1), color = 'yellow')
