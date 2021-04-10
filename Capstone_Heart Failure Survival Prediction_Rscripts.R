#############################################################
# HarvardX: PH125.9 - Data Science: Capstone
#############################################################
#
# The following script uses the Heart failure clinical records Data Set
# Disease dataset and tests different models in order to predict
# instances of heart disease in patients from a number of parameters.
# This code was run on Windows 8 OS with RStudio Version 1.1.447.
#
# Find this project online at: https://github.com/kingsleysamkf/HarvardX-PH125.9x-Capstone-project/
# Resource page for the dataset: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
#
#############################################################
# Retrieve and Tidy Dataset
#############################################################

library(dplyr)
library(tidyverse)
library(readxl)
library(tinytex)
library(e1071)
library(randomForest)
library(rsample)
library(xgboost)
library(adabag)
library(data.table)
library(caret)

# Automatically download data to working directory (NOTE: Comment lines if project was cloned from GitHub)
# IN CASE OF FAILURE FOR THE AUTOMATIC DOWNLOAD:
# 1. Go to: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
# 2. Download heart_failure_clinical_records_dataset.csv to your working directory
# OR
# 1. Go to: https://github.com/kingsleysamkf/HarvardX-PH125.9x-Capstone-project/
# 2. Click "Clone or Download" button, then select "Download ZIP".
# 3. Unzip the downloaded folder and copy the file "heart_failure_clinical_records_dataset.csv" to your current working directory.
# Finally: Comment the download.file lines and launch the code again
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv", 
              "heart_failure_clinical_records_dataset.csv")
# Load the dataset with named columns
data_columns <- c("age", "anaemia", "hbp", "CPK", "diabetes", "ef", "platelets", "sex",
                  "serum_creatinine", "serum_Na", "smoking", "time_fu_period", "death")
data<- read.csv ("heart_failure_clinical_records_dataset.csv", sep="," , header = TRUE)
#Visual check the dataset
view(data)

#2.2 Statistical quantitative description of the variables
str(data)
dim(data)
head(data)
summary(data)

# Vizualize the density distributions for Death cases against the variables
# the available continuous features
# Create a function for the density plots
density_plot <- function(column, param_name){
  ggplot(data, aes(x=column, fill=DEATH_EVENT, color=DEATH_EVENT)) +
    geom_density(alpha=0.2) +
    theme(legend.position="bottom") +
    scale_x_continuous(name=param_name) +
    scale_fill_discrete(name='DEATH EVENT',labels=c("No", "Yes")) +
    scale_color_discrete(name='DEATH EVENT',labels=c("No", "Yes"))
}



# Plot for all continuous variables
plotAge <- density_plot(data$age, "Age")
plotAge

plotCreatinine_phosphokinase<-density_plot(data$creatinine_phosphokinase, "level of the creatinine phosphokinase(CPK) enzyme in the blood (mcg/L)")
plotCreatinine_phosphokinase

plotEjection_fraction<-density_plot(data$ejection_fraction, "Ejection fraction %")
plotEjection_fraction

plotPlatelets <- density_plot(data$platelets, "platelets in the blood (kiloplatelets/mL)")
plotPlatelets

plotSerum_creatinine <- density_plot(data$serum_creatinine, "level of serum creatinine in the blood (mg/dL) ")
plotSerum_creatinine

plotSerum_sodium <- density_plot(data$serum_sodium, "Level of Serum sodium in the blood (mEq/L)")
plotSerum_sodium

plotTime <- density_plot(data$time, "Time of follow-up period(days)")
plotTime


# Vizualize the density distributions for Death cases against the variables
# the available categorical features
# Create a function for the barplots
format_barplot <- function(gc, columngroup, param_name, labelling){
  ggplot(gc, aes(x=columngroup, y=n, fill=DEATH_EVENT))+ 
    geom_bar( stat="identity") +
    scale_x_discrete(name=param_name, labels=labelling) +
    scale_fill_discrete(name='DEATH EVENT', labels=c("No", "Yes")) +
    scale_color_discrete(name='DEATH EVENT', labels=c("No", "Yes")) +
    theme(legend.position="top")
}

# Convert categorical variables from numeric to factor.
cols <- c("anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "DEATH_EVENT")
data[cols] <- lapply(data[cols], factor)


# Plot for all categorical features
groupby_sex <- data %>% group_by(sex) %>% count(DEATH_EVENT) %>% as.data.frame()
plotSex <- format_barplot(groupby_sex, groupby_sex$sex, "sex", c("Female", "Male"))
plotSex

groupby_anaemia <- data %>% group_by(anaemia) %>% count(DEATH_EVENT) %>% as.data.frame()
plotAnaemia <- format_barplot(groupby_anaemia, groupby_anaemia$anaemia, "Anaemia", c("Normal", "Patient with Anaemia"))
plotAnaemia

groupby_diabetes <- data %>% group_by(diabetes) %>% count(DEATH_EVENT) %>% as.data.frame()
plotDiabetes <- format_barplot(groupby_diabetes, groupby_diabetes$diabetes, "Diabetes", c("Normal", "Patient with Diabetes"))
plotDiabetes

groupby_smoking <- data %>% group_by(smoking) %>% count(DEATH_EVENT) %>% as.data.frame()
plotSmoking <- format_barplot(groupby_smoking, groupby_smoking$smoking, "smoking", c("Non-smoker", "Smoker"))
plotSmoking

groupby_hbp <- data %>% group_by(high_blood_pressure) %>% count(DEATH_EVENT) %>% as.data.frame()
plotHBP <- format_barplot(groupby_hbp, groupby_hbp$high_blood_pressure, "Hypertension", c("Normal Blood Pressure", "Hypertension"))
plotHBP


# Chi-Square Test for each independent categorical variable to show their association with the dependent variable death event
# sex
table(data$DEATH_EVENT, data$sex)
chisq.test(data$DEATH_EVENT, data$sex, correct=FALSE)                        
# anaemia
table(data$DEATH_EVENT, data$anaemia)
chisq.test(data$DEATH_EVENT, data$anaemia, correct=FALSE)  
# diabetes
table(data$DEATH_EVENT, data$diabetes)
chisq.test(data$DEATH_EVENT, data$diabetes, correct=FALSE) 
# smoking
table(data$DEATH_EVENT, data$smoking)
chisq.test(data$DEATH_EVENT, data$smoking, correct=FALSE)  
# sex
table(data$DEATH_EVENT, data$high_blood_pressure)
chisq.test(data$DEATH_EVENT, data$high_blood_pressure, correct=FALSE)  
#ejection_fraction
data$DEATH_EVENT <- as.numeric(data$DEATH_EVENT)
cor.test(data$ejection_fraction, data$DEATH_EVENT)
# high_blood_pressure
table(data$DEATH_EVENT, data$high_blood_pressure)
chisq.test(data$DEATH_EVENT, data$high_blood_pressure, correct=FALSE) 

# Pearson's correlation for each independent continuous variable to show their association with the dependent variable death event
#age
cor.test(data$age, data$DEATH_EVENT)
#creatinine_phosphokinase
cor.test(data$creatinine_phosphokinase, data$DEATH_EVENT)
#Platelets 
cor.test(data$platelets , data$DEATH_EVENT)
#Serum_creatinine
cor.test(data$serum_creatinine, data$DEATH_EVENT)
#serum_sodium
cor.test(data$serum_sodium, data$DEATH_EVENT)
#time
cor.test(data$time, data$DEATH_EVENT)

# Keep the statistically significant independent variables and the dependent variable the a clean dataset
keep_columns <- c(1, 5, 8, 9, 12, 13) 
data_clean <- data[, keep_columns] 
dim(data_clean)
str(data_clean)
view(data_clean)


# We are now ready to select a machine learning algorithm to create a prediction model for our datasets.
cols <- c(6)
data_clean[cols] <- lapply(data_clean[cols], factor)

# The testing set will be 20% of the orignal dataset.
set.seed(1)
index <- createDataPartition(y = data_clean$DEATH_EVENT, times = 1, p = 0.2, list = FALSE)
trainingSet <- data_clean[-index,]
testingSet <- data_clean[index,]


# 1st model: Knn 
# We train a k-nearest neighbor algorithm with a tunegrid parameter to optimize for k
library(caret)
library(e1071)
set.seed(1000)
train_knn <- train(DEATH_EVENT ~ ., method = "knn",
                   data = trainingSet,
                   tuneGrid = data.frame(k = seq(2, 30, 2)))
train_knn$bestTune
confusionMatrix(predict(train_knn, testingSet, type = "raw"),
                testingSet$DEATH_EVENT)$overall["Accuracy"]

# Visualize and save the optimal value for k
k_plot <- ggplot(train_knn, highlight = TRUE)
k_plot
optim_k <- train_knn$bestTune[1, 1]
# Train and predict using k-nn with optimized k value
knn_fit <- knn3(DEATH_EVENT ~ ., data = trainingSet, k = optim_k)
y_hat_knn <- predict(knn_fit, testingSet, type = "class")
cm_knn <- confusionMatrix(data = y_hat_knn, reference = testingSet$DEATH_EVENT, positive = "2")
# Return optimized k value, Accuracy, Sensitivity and Specificity
Accuracy_knn <- cm_knn$overall["Accuracy"]
Sensitivity_knn <- cm_knn$byClass["Sensitivity"]
Specificity_knn <- cm_knn$byClass["Specificity"]
Accuracy_knn
Sensitivity_knn
Specificity_knn

# 2nd model: Naive Bayes
# Look at correlation between features to verify independance
matrix_data <- matrix(as.numeric(unlist(data_clean)),nrow=nrow(data_clean)) 
correlations <- cor(matrix_data)
correlations

# Train and predict using Naive Bayes
train_nb <- train(DEATH_EVENT ~ ., method = "nb", data = trainingSet)
y_hat_nb <- predict(train_nb, testingSet)
cm_nb <- confusionMatrix(data = y_hat_nb, reference = testingSet$DEATH_EVENT, positive = "2")
cm_nb
# Return Accuracy, Sensitivity and Specificity
Accuracy_nb <- cm_nb$overall["Accuracy"]
Sensitivity_nb <- cm_nb$byClass["Sensitivity"]
Specificity_nb <- cm_nb$byClass["Specificity"]
Accuracy_nb
Sensitivity_nb
Specificity_nb

# 3rd model: Generalized Linear Regression Model
# perform 10-fold cross validation
trCntl <- trainControl(method = "CV",number = 10)
# fit into the generalized linear regression model
glmModel <- train(DEATH_EVENT ~ .,data = trainingSet,trControl = trCntl,method="glm",family = "binomial")
# print the model info
summary(glmModel)
glmModel
confusionMatrix(glmModel)

# 4th model: Random Forest Model with K-Fold Cross-Validation
library(randomForest)
library(rsample)
# Define train control for k-fold (10-fold here) cross validation
set.seed(1984)
train_control <- trainControl(method="cv", number=10)
# Train and predict using Random Forest
set.seed(1989)
train_rf <- train(DEATH_EVENT ~ ., data = trainingSet,
                    method = "rf",
                    trControl = train_control)
y_hat_rf <- predict(train_rf, testingSet)
cm_rf <- confusionMatrix(data = y_hat_rf, reference = testingSet$DEATH_EVENT, positive = "2")
cm_rf 
# Return Accuracy, Sensitivity and Specificity
Accuracy_rf <- cm_rf$overall["Accuracy"]
Sensitivity_rf <- cm_rf$byClass["Sensitivity"]
Specificity_rf <- cm_rf$byClass["Specificity"]
Accuracy_rf
Sensitivity_rf 
Specificity_rf

# 5th model: Weighted Subspace Random Forest with K-Fold Cross-Validation
set.seed(1989)
train_wsrf <- train(DEATH_EVENT ~ ., data = trainingSet,
                  method = "wsrf",
                  trControl = train_control)
y_hat_wsrf <- predict(train_wsrf, testingSet)
cm_wsrf <- confusionMatrix(data = y_hat_wsrf, reference = testingSet$DEATH_EVENT, positive = "2")
cm_wsrf 
# Return Accuracy, Sensitivity and Specificity
Accuracy_wsrf <- cm_wsrf$overall["Accuracy"]
Sensitivity_wsrf <- cm_wsrf$byClass["Sensitivity"]
Specificity_wsrf <- cm_wsrf$byClass["Specificity"]
Accuracy_wsrf
Sensitivity_wsrf 
Specificity_wsrf


#Model 6: Adaptive Boosting
library(adabag)
train_ada <- train(DEATH_EVENT ~ ., method = "adaboost", data = trainingSet)
y_hat_ada <- predict(train_ada, testingSet)
cm_ada <- confusionMatrix(data = y_hat_ada, reference = testingSet$DEATH_EVENT, positive = "2")
cm_ada 
# Return Accuracy, Sensitivity and Specificity
Accuracy_ada <- cm_ada$overall["Accuracy"]
Sensitivity_ada <- cm_ada$byClass["Sensitivity"]
Specificity_ada <- cm_ada$byClass["Specificity"]
Accuracy_ada
Sensitivity_ada
Specificity_ada

#Model 7: Extreme Gradient Boosting
library(readxl)
library(xgboost)
train_xgb <- train(DEATH_EVENT ~ ., method = "xgbTree", data = trainingSet)
y_hat_xgb <- predict(train_xgb, testingSet)
cm_xgb <- confusionMatrix(data = y_hat_xgb, reference = testingSet$DEATH_EVENT, positive = "2")
cm_xgb 
# Return Accuracy, Sensitivity and Specificity
Accuracy_xgb <- cm_xgb$overall["Accuracy"]
Sensitivity_xgb <- cm_xgb$byClass["Sensitivity"]
Specificity_xgb <- cm_xgb$byClass["Specificity"]
Accuracy_xgb
Sensitivity_xgb
Specificity_xgb

# Table of results comparing the different models.
results <- data_frame(
  Model=c("Model 1: K-Nearest Neighbors","Model 2: Naive Bayes", 
          "Model 3: Generalized Linear Regression Model with K-Fold Cross-Validation", 
          "Model 4: Random Forest and K-Fold Cross-Validation",
          "Model 5: Weighted Subspace Random Forest and K-Fold Cross-Validation", 
          "Model 6: Adaptive Boosting","Model 7: Extreme Gradient Boosting"), 
  Accuracy=c(Accuracy_knn, Accuracy_nb, Accuracy_glm, Accuracy_rf, Accuracy_wsrf, 
             Accuracy_ada, Accuracy_xgb),
  Sensitivity=c(Sensitivity_knn, Sensitivity_nb, "-", Sensitivity_rf, 
                Accuracy_wsrf, Sensitivity_ada, Sensitivity_xgb), 
  Specificity=c(Specificity_knn, Specificity_nb, "-", Specificity_rf, 
                Specificity_wsrf, Specificity_ada, Specificity_xgb)
) 
results
