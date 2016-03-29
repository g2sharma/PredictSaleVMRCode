#Libraries needed for our models
library(pscl) #Needed for the McFadden R2 test
#install.packages("SDMTools")
library(SDMTools)
#install.packages("clusterGeneration")
library(clusterGeneration)
library(MASS)
library(e1071)
library(caret)
library(stats)
library(randomForest) #New library for Random Forest model
library(graphics)


DataProspAndCust<- readRDS("DataProspAndCust.rds")
attach(DataProspAndCust)

smp_size <- floor(0.75 * nrow(DataProspAndCust))
set.seed(1234)
trainIndex <- sample(seq_len(nrow(DataProspAndCust)), size = smp_size)

#Split data based on even representation of Interest in both test and train

#trainIndex <- createDataPartition(DataProspAndCust$interest, p = 0.7, list = FALSE)

#Create Training dataframe
DataPCTrain <- DataProspAndCust[trainIndex, ]

#We will create the vectors containing the means and standard deviations of our training
#dataset so we can apply these values to our test dataset.
DataTrainMeans <- colMeans(DataPCTrain[, c('Rev','EmCnt','CoOppy','CoOppDays','CoMQLCt','CoOppWCt','CoOppLCt',
                                           'CoOppOpCt','CoOppyMo','CoOppyWMo','CoOppyLMo','CoOppyOMo')])

#Use the apply function to calculate the standard deviation of the features included
DataTrainSD <- apply(DataPCTrain[, c('Rev','EmCnt','CoOppy','CoOppDays','CoMQLCt','CoOppWCt','CoOppLCt',
                                     'CoOppOpCt','CoOppyMo','CoOppyWMo','CoOppyLMo','CoOppyOMo')], MARGIN = 2, FUN = sd)

#Normalize training set by calculating the mean and standard deviation
#Then subtracting the mean from individual numeric feature values and dividing
#by the standard deviation.
#Normalizing only columns 6:17. DataPCTrain will have the new normalized columns
#We can use the scale function to standardize the numeric values in our training dataset
DataPCNormalTrain <- cbind(scale(DataPCTrain[,6:17]), DataPCTrain[, -c(6:17)])


#Create Test dataframe
DataPCTest <- DataProspAndCust[-trainIndex, ]

#Repeated 10 fold cross validation on training 
#http://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/

#save
saveRDS(DataPCNormalTrain, file="DataPCNormalTrain.rds")
saveRDS(DataPCTest, file="DataPCTest.rds")
saveRDS(DataPCTrain, file="DataPCTrain.rds")



#Normalize Test using values calculated from the training set
#First transformation of test will be subtracting mean of train from test vals
#We can use the sweep function to help us apply the training data means and SD to the test dataset

DataPCNormalTestTrans1 <- sweep(DataPCTest[,c('Rev','EmCnt','CoOppy','CoOppDays','CoMQLCt','CoOppWCt','CoOppLCt',
                                              'CoOppOpCt','CoOppyMo','CoOppyWMo','CoOppyLMo','CoOppyOMo')], 2, DataTrainMeans, "-")

#Second take the first transformation and divide by training SD
DataPCNormalTestTrans2 <- sweep(DataPCNormalTestTrans1, 2, DataTrainSD, "/")


#Create the normalized dataset used by the model by binding the factor features with the normalized numeric values
DataPCNormalTest <- cbind(DataPCNormalTestTrans2, DataPCTest[,c('interest', 'IsCust','Cntry1','Cntry2','ActSz1',
                                                                'ActSz2','ActSz3','Ind1','Ind2','Ind3','Ind4',
                                                                'Ind5','Ind6','Ind7','Ind8','Ind9','Ind10')] )

saveRDS(DataPCNormalTest, file = "DataPCNormalTest.rds")


#***************MODELS*****************************************

#Run the logistic regression on DataPCNormalTrain with all features without multicollinearity test
Allmod_fitGLM <- glm(interest ~ Rev + EmCnt + CoOppy + CoOppDays +
                       CoMQLCt + CoOppWCt + CoOppLCt + CoOppOpCt + CoOppyMo + CoOppyWMo +
                       CoOppyLMo + CoOppyOMo + Cntry1 + Cntry2 + ActSz1 + ActSz2 +
                       ActSz3 + Ind1 + Ind2 + Ind3 + Ind4 + Ind5 + Ind6 + Ind7 + Ind8 +
                       Ind9 + Ind10,  data=DataPCNormalTrain, family="binomial")
saveRDS(Allmod_fitGLM, file ="Allmod_fitGLM.rdata")


#The difference between the null deviance and the residual deviance shows how 
#our model is doing against the null model (a model with only the intercept). 
#The wider this gap, the better. Analyzing the table we can see the drop in deviance
#when adding each variable one at a time. Again, adding attributes 
#significantly reduces the residual deviance. The other variables seem to 
#improve the model less even with a low p-value. A large p-value here 
#indicates that the model without the variable explains more or less the same amount
#of variation. Ultimately what you would like to see is a significant drop in deviance and the AIC.

#While no exact equivalent to the R2 of linear regression exists, 
#the McFadden R2 index can be used to assess the model fit.Need library pscl
pR2(Allmod_fitGLM)

#Get the odds by exponentiating the estimates
cbind(exp(coef(mod_fit)), exp(confint(mod_fit)))

#The statistic to determine the overall significance of a logistic model is the 
#likelihood ratio test. It compares the likelihood of the full model (with all the predictors included) 
#with the likelihood of the null model (which contains only the intercept). 
#It is analogous to the overall F-test in linear regression. The likelihood ratio 
#test statistic is: G_0^2 = 2ln\frac{L}{L_0} where L is the likelihood of the full 
#model and L_0 is the likelihood of the null model. The likelihood ratio 
#test statistic (G_0^2 = 41.73) can be compared to a \chi^2 distribution with 3 degrees of freedom.


Allmod_fitGLM.null <- glm(interest ~ 1, family = binomial, data = DataPCNormalTrain)
summary(Allmod_fitGLM.null)
lr.Allmod_fitGLM <- -(deviance(Allmod_fitGLM) / 2)
lr.Allmod_fitGLM.null <- -(deviance(Allmod_fitGLM.null) / 2)
(lr <- 2 * (lr.Allmod_fitGLM - lr.Allmod_fitGLM.null))
1 - pchisq(lr, 2)

1-pchisq(deviance(Allmod_fitGLM), df.residual(Allmod_fitGLM))

#Confusion matrix for the logit model without any changes
#Get probabilities on the training data of 0 or 1 class: using stat sig model without multicol variables best AIC Score
probTrainData = predict(Allmod_fitGLM, type = c("response"))
#Adding probtraindata probab column to DataPCNormalTrain dataset
DataPCNormalTrain$interest2 = probTrainData
#confustion matrix on training data
confusion.matrix(DataPCNormalTrain$interest, probTrainData, threshold = 0.5)


#check for multicolinearity on statisticall significant variables
#########test of colinearity
DataProspAndCust<- readRDS("DataProspAndCust.rds")
attach(DataProspAndCust)
colintest<-DataProspAndCust
rm(DataProspAndCust)
colintest$testTarget<-DataProspAndCust$Rev
colin<-lm(testTarget~ CoOppy + CoOppDays +
            CoMQLCt + CoOppWCt + CoOppLCt + CoOppyMo +
            CoOppyLMo + CoOppyOMo, data=colintest)

# Evaluate Collinearity
vif(colin) # variance inflation factors 
colintestall<-vif(colin)
sqrt(vif(colin)) > 2 # problem?

#remove CoOppyMo
colin2<-lm(testTarget~ CoOppy + CoOppDays +
             CoMQLCt + CoOppWCt + CoOppLCt +
             CoOppyLMo + CoOppyOMo, data=colintest)

# Evaluate Collinearity
vif(colin2) # variance inflation factors 
sqrt(vif(colin2)) > 2 # problem?

#remove CoOppyMo
colin2<-lm(testTarget~ CoOppy + CoOppDays +
             CoMQLCt + CoOppWCt + CoOppLCt +
             CoOppyLMo + CoOppyOMo, data=colintest)

#remove CoOppy
colin3<-lm(testTarget~ CoOppDays +
             CoMQLCt + CoOppWCt + CoOppLCt +
             CoOppyLMo + CoOppyOMo, data=colintest)


# Evaluate Collinearity
vif(colin3) # variance inflation factors 
sqrt(vif(colin3)) > 2 # problem?
rm(colin, colin2, colin3)


#Run a logistic regression without multicolin variables
Allmod_fitGLM2 <- glm(interest ~ Rev + EmCnt + CoOppDays +
                        CoMQLCt + CoOppWCt + CoOppLCt + CoOppOpCt + CoOppyWMo +
                        CoOppyLMo + CoOppyOMo + Cntry1 + Cntry2 + ActSz1 + ActSz2 +
                        ActSz3 + Ind1 + Ind2 + Ind3 + Ind4 + Ind5 + Ind6 + Ind7 + Ind8 +
                        Ind9 + Ind10,  data=DataPCNormalTrain, family="binomial")
summary(Allmod_fitGLM2)

#McFadden R2 on the new output
pR2(Allmod_fitGLM2)

#Pearson's goodness of fit.
#goodness fit using pearson and deviance residuals
sum(residuals(Allmod_fitGLM2, type = "pearson")^2)
deviance(Allmod_fitGLM2)
1 - pchisq(deviance(Allmod_fitGLM2), df.residual(Allmod_fitGLM2))

probTrainData2 = predict(Allmod_fitGLM2, type = c("response"))
#Adding probtraindata probab column to DataPCNormalTrain dataset
#DataPCNormalTrain$interest2 = probTrainData
#confustion matrix on training data
confusion.matrix(DataPCNormalTrain$interest, probTrainData2, threshold = 0.5)


#Run a Step wise logit regression removing variable with mulitcolinearity tested

stepmod_fit <- step(glm(interest ~ Rev + EmCnt + CoOppDays +
                       CoMQLCt + CoOppWCt + CoOppLCt + CoOppOpCt + CoOppyWMo +
                       CoOppyLMo + CoOppyOMo + Cntry1 + Cntry2 + ActSz1 + ActSz2 +
                       ActSz3 + Ind1 + Ind2 + Ind3 + Ind4 + Ind5 + Ind6 + Ind7 + Ind8 +
                       Ind9 + Ind10,  data=DataPCNormalTrain, family="binomial"))

summary(stepmod_fit)

#McFadden R2 on the new output
pR2(stepmod_fit)

#Pearson's goodness of fit.
#goodness fit using pearson and deviance residuals
sum(residuals(stepmod_fit, type = "pearson")^2)
deviance(stepmod_fit)
1 - pchisq(deviance(stepmod_fit), df.residual(stepmod_fit))

probTrainData3 = predict(stepmod_fit, type = c("response"))
#Adding probtraindata probab column to DataPCNormalTrain dataset
#DataPCNormalTrain$interest2 = probTrainData
#confustion matrix on training data
confusion.matrix(DataPCNormalTrain$interest, probTrainData3, threshold = 0.5)


#Run a logistic regression with only the without multicolin variables and statist significant variables
mod_fitStatSig <- glm(interest ~ CoOppDays +
                        CoMQLCt + CoOppWCt + CoOppLCt + CoOppyWMo +
                        CoOppyLMo + CoOppyOMo + Cntry2 + ActSz1,  data=DataPCNormalTrain, family="binomial")
summary(mod_fitStatSig)

#McFadden R2 on the new output
pR2(mod_fitStatSig)

#Pearson's goodness of fit.
#goodness fit using pearson and deviance residuals
sum(residuals(mod_fitStatSig, type = "pearson")^2)
deviance(mod_fitStatSig)
1 - pchisq(deviance(mod_fitStatSig), df.residual(mod_fitStatSig))

probTrainData4 = predict(mod_fitStatSig, type = c("response"))
#confustion matrix on training data
confusion.matrix(DataPCNormalTrain$interest, probTrainData4, threshold = 0.5)

#Apply our model to test to predict
probTestData = predict(stepmod_fit, type = c("response"))
predTst <- predict(stepmod_fit, DataPCNormalTest, type="response")

confusion.matrix(DataPCNormalTest$interest, predTst, threshold = 0.5)




#We will now use a Support Vector machine. Default kernel is 
DataSVMTrain <- svm(interest ~ Rev + EmCnt + CoOppy + CoOppDays +
                      CoMQLCt + CoOppWCt + CoOppLCt + CoOppOpCt + CoOppyMo + CoOppyWMo +
                      CoOppyLMo + CoOppyOMo + Cntry1 + Cntry2 + ActSz1 + ActSz2 +
                      ActSz3 + Ind1 + Ind2 + Ind3 + Ind4 + Ind5 + Ind6 + Ind7 + Ind8 +
                      Ind9 + Ind10,  data=DataPCNormalTrain)
predictSVMTrain <- predict(DataSVMTrain, DataPCNormalTrain)
points(DataPCNormalTrain$interest, predictSVMTrain, col = "red", pch=4)

error <- DataPCNormalTrain$interest - predictSVMTrain
svrPredictionRMSE <- RMSE(predictSVMTrain, DataPCNormalTrain$interest)
summary(DataSVMTrain)
confusion.matrix(DataPCNormalTrain$interest, predictSVMTrain, threshold = 0.5)

#Apply SVM to test data
predSVMTest <- predict(DataSVMTrain, DataPCNormalTest, type="response")
confusion.matrix(DataPCNormalTest$interest, predSVMTest)

#Random Forest
DataRandomForestTrain <- randomForest(interest ~ Rev + EmCnt + CoOppy + CoOppDays +
                                        CoMQLCt + CoOppWCt + CoOppLCt + CoOppOpCt + CoOppyMo + CoOppyWMo +
                                        CoOppyLMo + CoOppyOMo + Cntry1 + Cntry2 + ActSz1 + ActSz2 +
                                        ActSz3 + Ind1 + Ind2 + Ind3 + Ind4 + Ind5 + Ind6 + Ind7 + Ind8 +
                                        Ind9 + Ind10, data=DataPCNormalTrain,ntree=50, importance=TRUE)

DataRandomForestTrain500 <- randomForest(interest ~ Rev + EmCnt + CoOppy + CoOppDays +
                                        CoMQLCt + CoOppWCt + CoOppLCt + CoOppOpCt + CoOppyMo + CoOppyWMo +
                                        CoOppyLMo + CoOppyOMo + Cntry1 + Cntry2 + ActSz1 + ActSz2 +
                                        ActSz3 + Ind1 + Ind2 + Ind3 + Ind4 + Ind5 + Ind6 + Ind7 + Ind8 +
                                        Ind9 + Ind10, data=DataPCNormalTrain,ntree=500, importance=TRUE)
#Apply Random Forest to test data
predRandomForestTest <- predict(DataRandomForestTrain500, DataPCNormalTest, type="response")
confusion.matrix(DataPCNormalTest$interest, predRandomForestTest)


