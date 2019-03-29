rm(list=ls())
setwd("C:/Users/hp/Desktop/minor-harsh/images")
library("keras")
library("caret")
dataset = read.csv("Imagedataset2.csv")

library(caTools)
set.seed(123)

sam = read.csv('test.csv')
labels = sam[,1]
labels = as.vector(labels)
dataset= cbind(dataset,labels)
dataset$labels = factor(dataset$labels,
                        levels = c('Hernia','Pneumonia','Fibrosis','Edema','Emphysema',
                                   'Cardiomegaly','Pleural_Thickening','Consolidation',
                                   'Pneumothorax','Mass','Nodule','Atelectasis',
                                   'Effusion','Infiltration','No Finding'),
                        labels = c(0,1, 2, 3,4,5,6,7,8,9,10,11,12,13,14))

split = sample.split(dataset$labels, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

X_train = as.matrix(training_set[-16385])
Y_train = unlist(training_set[16385])
Y_train=as.vector(Y_train)
#Y_train=as.factor(Y_train)

X_test = as.matrix(test_set[-16385])
Y_test = unlist(test_set[16385])
Y_test=as.vector(Y_test)
#Y_test=as.factor(Y_test)

#Y_train = to_categorical(Y_train, num_classes = 15
#Y_test = to_categorical(Y_test, num_classes = 15)

#XGBoost
#install.packages('xgboost')
#library(xgboost)
#classifier = xgboost(data = X_train, label = Y_train, nrounds = 5)
saveRDS(classifier , file = "xboost.rds")
classifier <- readRDS('./xboost.rds')


# Predict the test set
y_pred = predict(classifier, newdata = X_test)
y_pred=round(y_pred)
y_pred=as.factor(y_pred)

levels(y_pred)=c(0,1, 2, 3,4,5,6,7,8,9,10,11,12,13,14)
Y_test=as.factor(Y_test)
levels(Y_test)=c(0,1, 2, 3,4,5,6,7,8,9,10,11,12,13,14)

cm = confusionMatrix(y_pred, Y_test)

tr1=table(y_pred, Y_test)
sum(diag(tr1))/sum(tr1)



# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$labels, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = xgboost(data = X_train, label = Y_train, nrounds = 5)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-16385]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 16385], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
