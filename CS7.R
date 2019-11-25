library(mlbench)
library(caret)
library(dplyr)
data(BreastCancer)
#Binary Classification
#Predict whether a tissue sample is malignant or benign given properties
#about the tissue sample.
View(BreastCancer)
head(BreastCancer)
dim(BreastCancer)
glimpse(BreastCancer)
levels(BreastCancer$Class)
BreastCancer<-BreastCancer[,-1]
index<-createDataPartition(BreastCancer$Class, p=0.8, list=FALSE)
trainSet<-BreastCancer[index,]
testSet<-BreastCancer[-index,]
sum(is.na(BreastCancer))
summary(BreastCancer)
sapply(BreastCancer,class)
#Convert to numeric values
for(i in 1:9){
  BreastCancer[,i]=as.numeric(as.character(BreastCancer[,i]))
}
summary(BreastCancer)
#Class Distribution
percentage<-prop.table(table(BreastCancer$Class))*100
cbind(freq=table(BreastCancer$Class),perc=percentage)
#Summarize correlation between attributes
complete_cases<-complete.cases(BreastCancer)
cor(BreastCancer[complete_cases,1:9])

#Unimodal Visualisation(might face trouble due to having bug in this updated version)
#Try with Jupyter R kernel
par(mfrow=c(3,3))
for(i in 1:9){
  hist(BreastCancer[,i],main=names(BreastCancer)[i])
}
par(mfrow=c(3,3))
complete_cases<-complete.cases(BreastCancer)
for(i in 1:9){
  plot(density(BreastCancer[complete_cases,i]), main = names(BreastCancer)[i])
}
par(mfrow=c(3,3))
for(i in 1:9){
  boxplot(BreastCancer[,i],main=names(BreastCancer)[i])
}
#Multimodal Visualisation
jittered_x<-sapply(BreastCancer[,1:9],jitter)
pairs(jittered_x,names(BreastCancer[,1:9]),col=BreastCancer$Class)

#Try with some algorithms
#Linear : LG, GLMNET, LDA
#Non-linear : knn, SVM, NB, RPART
BreastCancer<-na.omit(BreastCancer)
summary(BreastCancer)
control<-trainControl(method = "repeatedcv", number=10, repeats = 3)
fit.lg<-train(Class~., BreastCancer, method = "glm", metric = "Accuracy", trControl=control)
fit.glmnet<-train(Class~., BreastCancer, method = "glmnet", metric = "Accuracy", trControl=control)
fit.rpart<-train(Class~., BreastCancer, method = "rpart", metric = "Accuracy", trControl=control)
fit.knn<-train(Class~., BreastCancer, method = "knn", metric = "Accuracy", trControl=control)
fit.svm<-train(Class~., BreastCancer, method = "svmRadial", metric = "Accuracy", trControl=control)
fit.nb<-train(Class~., BreastCancer, method = "nb", metric = "Accuracy", trControl=control)

#Compare the results
results<-resamples(list(fit.lg,fit.glmnet,fit.rpart,fit.knn,fit.svm,fit.nb))
summary(results)
dotplot(results)

#Evaluating algorithms : Box-Cox transformation
#tune the SVM
#grid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
#tunning knn
#grid <- expand.grid(.k=seq(1,20,by=1))

#Ensemble methods
#Bagging : BAG, RF/ Boosting : C5.0, GBM

#Finalize the model
#prepare parameters for transformation
set.seed(99)
datasetNoMissing <- BreastCancer[complete.cases(BreastCancer),]
x <- datasetNoMissing[,1:9]
preprocessParams <- preProcess(x, method=c("BoxCox"))
x <- predict(preprocessParams, x)

#1. Remove the Id attribute.
#2. Remove those rows with missing data.
#3. Convert all input attributes to numeric.
#4. Apply the Box-Cox transform to the input attributes using parameters prepared on the
#training dataset.
# prepare the validation dataset
set.seed(7)
# remove id column
testSet <- testSet[,-1]
# remove missing values (not allowed in this implementation of knn)
testSet <- testSet[complete.cases(testSet),]
# convert to numeric
for(i in 1:9) {
  testSet[,i] <- as.numeric(as.character(testSet[,i]))
}
# transform the validation dataset
validationX <- predict(preprocessParams, testSet[,1:9])

#Author chose KNN, I would choose RF
predictions <- knn3Train(x, validationX, datasetNoMissing$Class, k=9, prob=FALSE)
confusionMatrix(predictions, testSet$Class)




