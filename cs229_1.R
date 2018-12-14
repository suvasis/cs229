
library(glmnet)
#CS229 Fall 2018, suvasis mukherjee and Minakshi Mukherjee
# Load all the packages required for the analysis
library(dplyr) # Data Manipulation

#dummies
library(dummies)
#real xlsx
library("xlsx")

library(e1071)

library(psych)
#
# Load Data
# IIIIIIIIIIIIIIIIIIIII
###################################################################
setwd("/Users/mina/stats202/R")
#1. DATA LOADING -----
# import data set.  

df <- as.matrix(read.csv("leave_one_out_results_linearregression.csv",1 , stringsAsFactors=F),header = TRUE)
head(df)

n = nrow(df)
glimpse(df)
# Split data into train (2/3) and test (1/3) sets

##################################################################
#### split data 80% train and 20% test
##################################################################
train_rows <- sample(1:n, .8*n)
columns <- colnames(df)
print(columns)
x <- subset(df, select = c(2, 3,4, 5,6,7,14,15,16,17,18))
y <- subset(df, select = c(9))
x.train <- x[train_rows, ]
x.test <- x[-train_rows, ]

#print(train_rows)
y.train <- y[train_rows,]
y.test <- y[-train_rows,]

grid = 10^seq(10,-2,length=100)
fit.lasso <- glmnet(x.train, y.train, family="gaussian", alpha=1,lambda=grid)
fit.ridge <- glmnet(x.train, y.train, family="gaussian", alpha=0,lambda=grid)
fit.elnet <- glmnet(x.train, y.train, family="gaussian", alpha=.5,lambda = grid)

#Ridge
set.seed(1)
cv_ridge.out = cv.glmnet(x.train,y.train,alpha=0)
plot(cv_ridge.out)
bestlam1=cv_ridge.out$lambda.min
bestlam1  #1733.05
ridge.pred = predict(fit.ridge,s=bestlam,newx=x.test)
mean((ridge.pred-y.test)^2) #124.5299
out_ridge=glmnet(x,y,alpha=0)
predict(out_ridge,type="coefficients",s=bestlam1)

#Lasso
set.seed(1)
cv_lasso.out = cv.glmnet(x.train,y.train,alpha=1)
plot(cv_lasso.out)
bestlam2=cv_lasso.out$lambda.min
bestlam2  #1.73305
lasso.pred = predict(fit.lasso,s=bestlam2,newx=x.test)
mean((lasso.pred-y.test)^2) #124.426
out_lasso=glmnet(x,y,alpha=1)
predict(out_lasso,type="coefficients",s=bestlam2)


#enet
set.seed(1)
cv_elnet.out = cv.glmnet(x.train,y.train,alpha=.5)
plot(cv_elnet.out)
bestlam3=cv_enet.out$lambda.min
bestlam3  #3.4661
elnet.pred = predict(fit.elnet,s=bestlam3,newx=x.test)
mean((elnet.pred-y.test)^2) #124.4395
out_elnet=glmnet(x,y,alpha=1)
predict(out_elnet,type="coefficients",s=bestlam3)




for (i in 0:10) {
  assign(paste("fit", i, sep=""), cv.glmnet(x.train, y.train, type.measure="mse", 
                                            alpha=i/10,family="gaussian"))
}

# Plot solution paths:
par(mfrow=c(3,2))
# For plotting options, type '?plot.glmnet' in R console
plot(fit.lasso, xvar="lambda")
plot(fit10, main="LASSO")

plot(fit.ridge, xvar="lambda")
plot(fit0, main="Ridge")

plot(fit.elnet, xvar="lambda")
plot(fit5, main="Elastic Net")

#SVM

y_svm <- subset(df, select = c(13))
#print(train_rows)
y.train <- y_svm[train_rows,]
y.test <- y_svm[-train_rows,]

svm_train=data.frame(x.train[,11],y.train)
svm_test=data.frame(x.test[,11],y.test)
print(x.train[,11])
model_svm <- svm(y.train ~ x.train[,11] , svm_train, kernel="linear",cost=0.01,scale=FALSE)
# SVM-Kernel:  linear  Number of Support Vectors:  102
summary(model_svm)

#plot(model_svm,svm_train)
#Use the predictions on the data

pred <- predict(model_svm, svm_train)
y.pred = ifelse(pred > 0.5, 1, 0)
table(svm_train$y.train, y.pred)
mean(y.pred!=svm_train$y.train) #0.4215686
test1.pred <- predict(model_svm, svm_test)
fitpred_l = prediction(test1.pred[1:26],svm_test$y.test)
fitperf_l = performance(fitpred_l,"tpr","fpr")
test.pred = ifelse(test1.pred > 0.5, 1, 0)
table(svm_test$y.test, test.pred[1:26])
mean(test.pred!=svm_test$y.test)#0.3137255


#Plot the predictions and the plot to see our model fit

points(svm_train$x.train, pred, col = "blue", pch=4)

plot(fitperf_l,col="green",lwd=2,main="ROC Curve for SVM")
abline(a=0,b=1,lwd=2,lty=2,col="gray")
auc_l <- performance(fitpred_l,"auc")
auc_l@y.values[[1]] #0.4340278

