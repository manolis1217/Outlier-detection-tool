
Globalpath<- "C:/Users/ioliv/Documents/uu/ssml/Spatial_group_project" #change the path to your folder
setwd(Globalpath)
getwd()

#install.packages("easypackages")
easypackages::packages ("sf", "sp", "tmap", "mapview", "car", "RColorBrewer", "tidyverse", "osmdata", "nngeo", "FNN", "rpart", "rpart.plot", "randomForest", "sessioninfo", "caret", "rattle", "ipred", "tidymodels", "ranger", "recipes", "workflows", "themis","xgboost", "modelStudio", "DALEX", "DALEXtra", "vip", "pdp")

#install.packages("rlang")
library(rlang)

ref_data <- st_read ("data/final_dataset.csv")

set.seed(123)
#First we have to split the data into training and test set
#let us create data frame dropping the geometry filed, for simple handling in the model. Keep in mind we still have the lat, lon column that stored the actual location
ref_data_df <- ref_data %>% st_drop_geometry()

#Split the data
#we are using rsample package of tidymodel environment
data_split <- rsample::initial_split(ref_data, strata = "incomming_ref", prop = 0.75) #where we are splitting the data at 75-25, and stratifying based on dependent variable 
train.set_wtID <- rsample::training(data_split)
test.set_wtID  <- rsample::testing(data_split)

#declare the set explicit
train.set <- train.set_wtID 
test.set <- test.set_wtID 


#Fit the decision tree 
#Note: We are using, method = "poisson". It can also be "a"nova", "poisson", "class" or "exp". Depending on the data type. In this case pedal is a count data so we selected poisson. 
DT0 <- rpart (pedal ~ trafficsigcount + bus_stopscount + restcount + street_lampcount + PoP2015_Number +
                cycle_ln_dist + bikefac_dist + bus_stop_dist + Shop_dist + edu_dist + park_dist + len_per_ar + inter3_4ways + avg_centBC +
                road_speed_median + elevstdev + NDVImean + shdiv, data= train.set,  method  = "anova") 


summary (DT0)

####################################################################

split1<- sample(c(rep(0, 0.75 * nrow(ref_data)), rep(1, 0.25 * nrow(ref_data))))
train <- ref_data[split1 == 0, ] 
test <- ref_data[split1== 1, ]   

library('randomForest')

set.seed(123)

train$incomming_ref = factor(train$incomming_ref) 

#fit a random forest model, here I am selecting some hyper parameter such as mtry = 6 you can test different numbers but we shall explore these in details in the following session.
#here the rule of thumb is to use p/3 variables for regression trees, here p is the number of predictors in the model, and we had 19 predictors
rf <-randomForest(incomming_ref ~ unemployment + clean_elections + gdp_capita +
                    population + gini_index + avg_inc_ref_5y + ratio_inc_ref_5y +
                    + outgoing_ref_neighbors, data= train, mtry = 3, ntree = 1000) 

print(rf)

#create model explainer
explainer_rf <- DALEX::explain(
  model = rf,
  data = train,
  y = as.integer (train$incomming_ref),
  label = "Random Forest",
  verbose = FALSE
)

#now make an interactive dashboard
modelStudio::modelStudio(explainer_rf)
