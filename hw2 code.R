#1. a. 

rm(list=ls())


N=150
P=50
X=matrix(NA,nrow=N,ncol=P)

#correlated predictors

set.seed(1000)
covmat=matrix(rnorm(P^2,sd=2),nrow=P)
covmat=covmat+t(covmat)
U=eigen(covmat)$vectors
D=diag(rexp(P,rate=10))
covmat=U%*%D%*%t(U)

library(mvtnorm)
set.seed(1000)
for(i in 1:N){
      X[i,]=rmvnorm(1,mean=rep(0,P),sigma=covmat)
}
X=data.frame(X)

betas.true=c(1,2,3,4,5,-1,-2,-3,-4,-5,rep(0,P-10))

#simulate the y
sigma=15.7
X=as.matrix(X)
set.seed(1000)
y=X%*%betas.true+rnorm(N,mean=0,sd=sigma)

##split the date to test and train

alldata=data.frame(cbind(y,X))
names(alldata)[1] <- "y"

train=alldata[1:100,]
test=alldata[101:150,]

#fit the ordinary linear regression using training data sets. calculate the variance inflation factor.

fit=lm(y~.,data=train)
summary(fit)

betas.lm=coef(fit)

# yhat.lm=predict(fit,newdata=test)
# mspe.lm=mean((test$y-yhat.lm)^2)
# mspe.lm 


library(car)
vif(fit)


#1.b.

library(glmnet)

## fit ridge (trying 100 different lambda values)
rr=glmnet(x=as.matrix(train[,-1]),y=as.numeric(train[,1]),alpha=0,nlambda=100)
plot(rr,xvar="lambda",main="Ridge Regression Betas for Different Values of the Tuning Parameter")


## use 10-fold crossvalidation to find the best lambda
cv.rr=cv.glmnet(x=as.matrix(train[,-1]),y=as.numeric(train[,1]),alpha=0,nfolds=10,nlambda=100)

## getting cvmspe from best value of lambda
# cvmspe.rr=min(cv.rr$cvm) 

## get lambda and best rr fit
lambda.rr=cv.rr$lambda.min
lambda.rr

## some plots
par(mfrow=c(1,2))
plot(cv.rr)
abline(v=log(lambda.rr))
plot(rr,xvar="lambda",main="Ridge Regression Betas for Different Values of the Tuning Parameter")
abline(v=log(lambda.rr))
 
## beta estimates for best lambda
betas.rr=coef(cv.rr,s="lambda.min")
betas.rr

## compare the ridge coefficients with standard regression coefficients
plot(betas.rr,betas.lm,xlim=c(-6,6),ylim=c(-55,55))
abline(0,1)



#1.c. repeat it for lasso.

## fit ridge (trying 100 different lambda values)
lasso=glmnet(x=as.matrix(train[,-1]),y=as.numeric(train[,1]),alpha=1,nlambda=100)
plot(lasso,xvar="lambda",main="Lasso Regression Betas for Different Values of the Tuning Parameter")


## use 10-fold crossvalidation to find the best lambda
cv.lasso=cv.glmnet(x=as.matrix(train[,-1]),y=as.numeric(train[,1]),alpha=1,nfolds=10)

## getting cvmspe from best value of lambda
# cvmspe.lasso=min(cv.lasso$cvm) 

## get lambda and best rr fit
lambda.lasso=cv.lasso$lambda.min
lambda.lasso

## beta estimates for best lambda
betas.lasso=coef(cv.lasso,s="lambda.min")
betas.lasso

## compare the lasso coefficients with standard coefficients
plot(betas.lasso,betas.lm,xlim=c(-6,6),ylim=c(-55,55))
abline(0,1)

par(mfrow=c(1,2))
plot(cv.lasso)
abline(v=log(lambda.lasso))
plot(lasso,xvar="lambda", main="Lasso Regression Betas for Different Values of the Tuning Parameter")
abline(v=log(lambda.lasso))

#1.d. repeat for alpha=0.5

## fit ridge (trying 100 different lambda values)
elastic=glmnet(x=as.matrix(train[,-1]),y=as.numeric(train[,1]),alpha=0.5,nlambda=100)
plot(elastic,xvar="lambda",main="Elastic Regression Betas for Different Values of the Tuning Parameter")


## use 10-fold crossvalidation to find the best lambda
cv.elastic=cv.glmnet(x=as.matrix(train[,-1]),y=as.numeric(train[,1]),alpha=0.5,nfolds=10,nlambda=100)

## getting cvmspe from best value of lambda
# cvmspe.elastic=min(cv.elastic$cvm) 

## get lambda and best rr fit
lambda.elastic=cv.elastic$lambda.min
lambda.elastic

## beta estimates for best lambda
betas.elastic=coef(cv.elastic,s="lambda.min")
betas.elastic

## compare the elastic coefficients with standard coefficients
plot(betas.elastic,betas.lm,xlim=c(-6,6),ylim=c(-55,55))
abline(0,1)


## 1. e 

# mspe for the linear regrssion
yhat.lm=predict(fit, newdata=test[,-1])
mspe.lm=mean((test$y-yhat.lm)^2)
mspe.lm

# mspe for the lasso
yhat.lasso=predict(cv.lasso,newx=as.matrix(test[,-1]),s="lambda.min")
mspe.lasso=mean((test$y-yhat.lasso)^2)
mspe.lasso

# mspe for the ridge
yhat.rr=predict(cv.rr,s="lambda.min",newx=as.matrix(test[,-1]))
mspe.rr=mean((test$y-yhat.rr)^2)
mspe.rr

# mspe for elastic
yhat.elastic=predict(cv.elastic,s="lambda.min",newx=as.matrix(test[,-1]))
mspe.elastic=mean((test$y-yhat.elastic)^2)
mspe.elastic




rm(list=ls())



##2. (a)
#logistic regression

getwd()
setwd("/Users/himawari/Desktop")
getwd()


#scale the variables
mydata=read.csv("BrainTumor.csv",sep=",", header=TRUE)
mydata12=mydata[,-1]

Class=mydata[,1]

scaled=scale(mydata12, center=TRUE, scale=TRUE)

mynewdata=cbind(Class, scaled)
mynewdata=as.data.frame(mynewdata)

rm(Class)

# split the data into 70, 30.

set.seed(100)
sample=sample.int(n = nrow(mynewdata), size = floor(nrow(mynewdata)*0.70), replace = FALSE)
train=mynewdata[sample, ]
test=mynewdata[-sample, ]


# fit standard regression
attach(train)
fit=glm(Class~Mean+Variance+Standard.Deviation+Entropy+Skewness+Kurtosis+Contrast+Energy+ASM+Homogeneity+Dissimilarity+Correlation,family="binomial",data=train)
detach(train)

summary(fit)


## 2. (b)

## lasso

library(glmnet)

X=as.matrix(train[,-1])
y=train[,1]

lasso=glmnet(x=X,y=y,family="binomial",alpha=1,nlambda=100)
## use 10-fold crossvalidation to find the best lambda
cv.lasso=cv.glmnet(x=X,y=y,family="binomial",alpha=1,nfolds=10)

## get lambda and best lasso fit
lambda.lasso=cv.lasso$lambda.1se
lambda.lasso

## cross validation plots
par(mfrow=c(1,2))
plot(cv.lasso)
abline(v=log(lambda.lasso))
plot(lasso,xvar="lambda")
abline(v=log(lambda.lasso))

## beta estimates for best lambda
# betas.lasso=coef(cv.lasso)
# betas.lasso


#ridge

X2=as.matrix(train[,-1])
y2=train[,1]

rr=glmnet(x=X2,y=y2,family="binomial",alpha=0,nlambda=100)
## use 10-fold crossvalidation to find the best lambda
cv.rr=cv.glmnet(x=X2,y=y2,family="binomial",alpha=0,nfolds=10)

## get lambda and best lasso fit
lambda.rr=cv.rr$lambda.1se
lambda.rr

## cross validation plots
par(mfrow=c(1,2))
plot(cv.rr)
abline(v=log(lambda.rr))
plot(rr,xvar="lambda")
abline(v=log(lambda.rr))

## beta estimates for best lambda
# betas.rr=coef(cv.rr)
# betas.rr


#predict

#ordinary glm
testnew=test[,-1]

eta=predict(fit,newdata=testnew,type="response")


plot(c(1:1129),eta)

tumor_class=eta


for (i in 1:1129) {
      if (eta[i]>=0.5) {
            tumor_class[i]=1
      } else {
            tumor_class[i]=0
      }
}

plot(c(1:1129), tumor_class)

real=test[,1]
prediction=tumor_class

accuracy=real-prediction

accuracy=unname(accuracy)
accuracy

check=0

for (i in 1:1129) {
      if (accuracy[i] != 0) {
            check=check+1
      }
}

calculator=(1129-check)/1129
calculator


#lasso case

lasso_prob=predict(cv.lasso, s="lambda.1se", newx=as.matrix(test[,-1]), type="response")
lasso_prob


plot(c(1:1129),lasso_prob)

tumor_class2=lasso_prob


for (i in 1:1129) {
      if (lasso_prob[i]>=0.5) {
            tumor_class2[i]=1
      } else {
            tumor_class2[i]=0
      }
}

plot(c(1:1129), tumor_class2)

real=test[,1]
prediction=tumor_class2

accuracy=real-prediction

accuracy=unname(accuracy)
accuracy

check=0

for (i in 1:1129) {
      if (accuracy[i] != 0) {
            check=check+1
      }
}

calculator=(1129-check)/1129
calculator


#ridge case


ridge_prob=predict(cv.rr, s="lambda.1se", newx=as.matrix(test[,-1]), type="response")
ridge_prob


plot(c(1:1129),ridge_prob)

tumor_class3=ridge_prob


for (i in 1:1129) {
      if (ridge_prob[i]>=0.5) {
            tumor_class3[i]=1
      } else {
            tumor_class3[i]=0
      }
}

plot(c(1:1129), tumor_class3)

real=test[,1]
prediction=tumor_class3

accuracy=real-prediction

accuracy=unname(accuracy)
accuracy

check=0

for (i in 1:1129) {
      if (accuracy[i] != 0) {
            check=check+1
      }
}

calculator=(1129-check)/1129
calculator







