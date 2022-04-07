

if (!require("easypackages")) install.packages("easypackages") #easy package manager
if (!require("tidyverse")) install.packages("tidyverse") #main data science packages
if (!require("sf")) install.packages("sf") #main GIS package
if (!require("sp")) install.packages("sp") #needed for some GIS operation, will not be in use from 2023
if (!require("spdep")) install.packages("spdep") #neighborhood analysis in R
if (!require("spatialreg")) install.packages("spatialreg") #spatial modelling such as lag, error
if (!require("spgwr")) install.packages("spgwr") #GWR modelling
if (!require("RColorBrewer")) install.packages("RColorBrewer") # getting interesting color
if (!require("tmap")) install.packages("tmap") # Mapping package
if (!require("mapview")) install.packages("mapview") # Mapping package
if (!require("car")) install.packages("car") #some base regression functions
if (!require("cowplot")) install.packages("cowplot") #some base regression functions
if (!require("leafsync")) install.packages("leafsync") #using with mapview
if (!require("leaflet.extras2")) install.packages("leaflet.extras2") #using with mapview

easypackages::packages ("sf", "sp", "spdep", "spatialreg", "spgwr", "tmap", "mapview", "car", "RColorBrewer", "tidyverse", 
                        "cowplot", "leafsync", "leaflet.extras2", "mapview", "caret", "xgboost", "randomForest", "lmvar")

################### READING DATA

Globalpath<- "C:/Users/ioliv/Documents/uu/ssml/Spatial_group_project" #change the path to your folder
setwd(Globalpath)
getwd()

ref_data <- st_read("data/dataset_geo.gpkg")
mapview::mapview(ref_data, zcol = "incomming_ref")

set.seed(555)

################### PREPROCESSING

model_data <- ref_data[,c('clean_elections','population','incomming_ref_target',
                          'gdp_capita','incomming_ref','avg_inc_ref_5y','outgoing_ref_other_weighted',
                          'ratio_inc_ref_5y','outgoing_ref_neighbors','ADMIN','ISO_A3')]

model_data <- model_data[!sf::st_is_empty(model_data), ] %>% na.omit()

set.seed(555)
data_split <- rsample::initial_split(model_data, strata = "incomming_ref_target", prop = 0.75)
train.set_wtID <- rsample::training(data_split)
test.set_wtID  <- rsample::testing(data_split)

train <- train.set_wtID 
test <- test.set_wtID 

################### CHECKING FOR SPATIAL AUTOCORRELATION

linearMod <- lm (incomming_ref_target ~  clean_elections + gdp_capita +
                   population + ratio_inc_ref_5y + # incomming_ref + avg_inc_ref_5y
                   + outgoing_ref_neighbors + outgoing_ref_other_weighted, data= model_data) 
summary(linearMod)

vif(linearMod)

#creating adjacency matrix
sf_use_s2(FALSE)
greendata_nbq <- poly2nb(model_data, queen=TRUE) #Queen’s Contiguity neighborhood
summary(greendata_nbq)
greendata_nbq_w <- nb2listw(greendata_nbq, style="W", zero.policy = TRUE) #Queen’s neighborhood wights
summary(greendata_nbq_w, zero.policy = TRUE)

mc_global <- moran.mc(linearMod$residuals, greendata_nbq_w, 139, alternative="greater", zero.policy = TRUE)
plot(mc_global)
mc_global

##################### RANDOM FOREST

set.seed(555)
rf <-randomForest(incomming_ref_target ~ incomming_ref + clean_elections + gdp_capita +
                    population + ratio_inc_ref_5y #+ avg_inc_ref_5y
                    + outgoing_ref_neighbors + outgoing_ref_other_weighted, data= train, mtry = 7, ntree = 15000) 

print(rf)
vip::vip(rf, num_features = 19, idth = 0.5, aesthetics = list(fill = "purple2"), include_type = T)
set.seed(555)
pred_rf <- predict(object = rf, newdata = test)
testRMSE_rf <- RMSE (pred = pred_rf, obs = test$incomming_ref_target)
testRMSE_rf

################### LINEAR MODEL

#+ clean_elections + gdp_capita +
#  population + ratio_inc_ref_5y + #avg_inc_ref_5y
#  + outgoing_ref_neighbors + outgoing_ref_other_weighted

mdl = lm(incomming_ref_target ~  incomming_ref + outgoing_ref_neighbors, data= train, y=T, x=T)
summary(mdl)

pred_lm <- predict(object = mdl, newdata = test)
testRMSE_lm <- RMSE (pred = pred_lm, obs = test$incomming_ref_target)
testRMSE_lm

cv.lm(mdl, k = 10)

fit <- train(incomming_ref_target ~  incomming_ref + outgoing_ref_neighbors, 
             data = train, 
             method = "lm", 
             trControl = trainControl(method = "cv", number = 10))


#par(mfrow = c(2, 2))
#plot(lm(log(incomming_ref_target) ~  incomming_ref + outgoing_ref_neighbors, data= model_data))

##################### XGBOOST

train<-st_drop_geometry(train)
test<-st_drop_geometry(test)

dtrain <- xgb.DMatrix(data.matrix(train[,!names(train) %in% c("incomming_ref_target","ADMIN","ISO_A3")]), label=train$incomming_ref_target)
dtest <- xgb.DMatrix(data.matrix(test[,!names(test) %in% c("incomming_ref_target","ADMIN","ISO_A3")]), label=test$incomming_ref_target)

set.seed(555)
xgb_model <- xgb.train(data = dtrain,
                     #booster = "gblinear",
                     objective = "reg:squarederror", 
                     eval_metric = "rmse",
                     max.depth =2, 
                     eta = 0.1, 
                     nround = 500, 
                     subsample = 0.5, 
                     colsample_bytree = 1, 
                     min_child_weight = 0,
                     gamma = 0
)

xgb.importance(colnames(dtrain), model = xgb_model)

pred_xgboost <- predict(xgb_model, dtest)
testRMSE_xgboost <- RMSE (pred = pred_xgboost, obs = test$incomming_ref_target)
testRMSE_xgboost

####################### RESULTS TOGETHER

test$pred_xgboost <- pred_xgboost
test$prediction_lm <- pred_lm
test$prediction_rf <- pred_rf

ggplot(data = test, aes(x = ADMIN))+
  geom_line(aes(y = pred_lm, group = 1, colour = 'pred_lm'))+
  geom_line(aes(y = pred_rf, group = 1, colour = 'pred_rf'))+
  geom_line(aes(y = pred_xgboost, group = 1, colour = 'pred_xgboost'))+
  geom_line(aes(y = incomming_ref_target, group = 1, colour = 'incomming_ref_target'), size=1)+
  scale_colour_manual("", 
                      breaks = c("incomming_ref_target", "pred_lm", "pred_rf", "pred_xgboost"),
                      values = c("red", "green", "blue", "yellow")) +
  labs(x = "Country",
       y = "Refugees")

results_df <- data.frame(rmse_lm = testRMSE_lm, 
                         rmse_rf = testRMSE_rf, 
                         rmse_xgboost = testRMSE_xgboost)

############################### TUNING PARAMETERS

### XGBOOST

xgb_caret <- train(x = train[,!names(train) %in% c("incomming_ref_target","ADMIN","ISO_A3")],
                   y = train$incomming_ref_target,
                   method = 'xgbTree',
                   objective = "reg:squarederror",
                   trControl = trainControl(method = "repeatedcv",
                                            number = 3,
                                            repeats = 2,
                                            verboseIter = TRUE),
                   tuneGrid = expand.grid(nrounds = c(500,1000,1500),
                                          eta = c(0.01,0.05,0.1,0.2,0.3),
                                          max_depth = c(2,4,6,8,10),
                                          subsample=c(0.5,1),
                                          colsample_bytree=c(0.5,1),
                                          min_child_weight=c(0,20),
                                          gamma=c(0,50))
                   )

pred_xgboost_caret <- predict(xgb_caret, test[,!names(test) %in% c("incomming_ref_target","ADMIN","ISO_A3",
                                                                   "prediction_lm","pred_xgboost","prediction_rf")])
testRMSE_xgboost_caret <- rmse(test$incomming_ref_target, pred_xgboost_caret)
testRMSE_xgboost_caret

### RANDOM FOREST

set.seed(555)
rf_caret <- train(incomming_ref_target ~ incomming_ref + clean_elections + gdp_capita +
                    population + ratio_inc_ref_5y +#avg_inc_ref_5y
                    + outgoing_ref_neighbors + outgoing_ref_other_weighted, 
                  data = train,
                  method = 'rf',
                  metric = 'RMSE',
                  trControl = trainControl(method = "repeatedcv",#https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Repeated_random_sub-sampling_validation
                                           number = 10,
                                           repeats = 5,
                                           verboseIter = TRUE),
                  tuneGrid = expand.grid(mtry = (1:7))
                  )

pred_rf_caret <- predict(rf_caret, test[,!names(test) %in% c("incomming_ref_target","ADMIN","ISO_A3",
                                                             "prediction_lm","pred_xgboost","prediction_rf")])
testRMSE_rf_caret <- RMSE(test$incomming_ref_target, pred_rf_caret)
testRMSE_rf_caret

#################################### VALIDATION

validation_data <- st_read("data/dataset_geo_validation.gpkg")
mapview::mapview(validation_data, zcol = "incomming_ref")

set.seed(555)

validation_model_data <- validation_data[,c('clean_elections','population','incomming_ref_target',
                          'gdp_capita','incomming_ref','avg_inc_ref_5y','outgoing_ref_other_weighted',
                          'ratio_inc_ref_5y','outgoing_ref_neighbors','ADMIN','ISO_A3')]

validation_model_data <- validation_model_data[!sf::st_is_empty(validation_model_data), ] %>% na.omit()

#Combined predictions

val_data_high <- validation_model_data[validation_model_data$incomming_ref > 50000,]
val_data_low <- validation_model_data[validation_model_data$incomming_ref < 50000,]

pred_lm_comb_high <- predict(object = mdl, newdata = val_data_high)
pred_rf_comb_low <- predict(object = rf, newdata = val_data_low)

val_data_high$pred_comb <- pred_lm_comb_high
val_data_low$pred_comb <- pred_rf_comb_low
comb_pred <- rbind(val_data_high, val_data_low)

validation_model_data <- st_drop_geometry(validation_model_data)
comb_pred <- st_drop_geometry(comb_pred)

RMSE (pred = comb_pred$pred_comb, obs = validation_model_data$incomming_ref_target)

### LINEAR MODEL

pred_lm_val <- predict(object = mdl, newdata = validation_model_data)
testRMSE_lm_val <- RMSE (pred = pred_lm_val, obs = validation_model_data$incomming_ref_target)
testRMSE_lm_val

### RANDOM FOREST

set.seed(555)
pred_rf_val <- predict(object = rf, newdata = validation_model_data)
testRMSE_rf_val <- RMSE (pred = pred_rf_val, obs = validation_model_data$incomming_ref_target)
testRMSE_rf_val

### XGBOOST

set.seed(555)
validation_set <- xgb.DMatrix(data.matrix(validation_model_data[,!names(validation_model_data) 
                                                                %in% c("incomming_ref_target","ADMIN","ISO_A3")]), 
                              label=validation_model_data$incomming_ref_target)

pred_xgboost_val <- predict(xgb_model, validation_set)
testRMSE_xgboost_val <- RMSE (pred = pred_xgboost_val, obs = validation_model_data$incomming_ref_target)
testRMSE_xgboost_val

validation_model_data <- merge(x = validation_model_data, y = comb_pred[,c("ISO_A3","pred_comb")],
                               by = "ISO_A3", all.x = TRUE)

validation_model_data$pred_xgboost <- pred_xgboost_val
validation_model_data$prediction_lm <- pred_lm_val
validation_model_data$prediction_rf <- pred_rf_val

ggplot(data = validation_model_data, aes(x = ADMIN))+
  geom_line(aes(y = prediction_lm, group = 1, colour = 'prediction_lm'))+
  geom_line(aes(y = prediction_rf, group = 1, colour = 'prediction_rf'))+
  geom_line(aes(y = pred_xgboost, group = 1, colour = 'pred_xgboost'))+
  geom_line(aes(y = pred_comb, group = 1, colour = 'pred_comb'), size=1)+
  geom_line(aes(y = incomming_ref_target, group = 1, colour = 'incomming_ref_target'), size=1)+
  scale_colour_manual("", 
                      breaks = c("incomming_ref_target", "prediction_lm", "prediction_rf", "pred_xgboost","pred_comb"),
                      values = c("red", "green", "blue", "yellow", "orange")) +
  labs(x = "Country",
       y = "Refugees")

results_df <- data.frame(rmse_lm = testRMSE_lm_val, 
                         rmse_rf = testRMSE_rf_val, 
                         rmse_xgboost = testRMSE_xgboost_val)

subset <- validation_model_data[,c('ADMIN','ISO_A3','incomming_ref_target',
                                   'prediction_lm','pred_xgboost','prediction_rf','pred_comb')]
subset['diff_lm'] <- subset['incomming_ref_target']-abs(subset['prediction_lm'])
subset['diff_rf'] <- subset['incomming_ref_target']-abs(subset['prediction_rf'])
subset['diff_xgb'] <- subset['incomming_ref_target']-abs(subset['pred_xgboost'])
subset['diff_pred_comb'] <- subset['incomming_ref_target']-abs(subset['pred_comb'])
