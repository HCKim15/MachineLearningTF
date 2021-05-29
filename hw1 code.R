
#homework 1
#1.

rm(list=ls())

getwd()
setwd("/Users/himawari/Desktop")
getwd()


mydata=read.csv("stock.csv",sep=",", header=TRUE)
attach(mydata)
head(mydata)

Month



#fit the multiple linear regression model 1.(d)

model <- lm(Stock ~ Interest + Unemployment + as.factor(Month)  )
summary(model)


ls(model)

#do it with hand (a)~(c)


month.f=factor(Month)
dummy=model.matrix(~month.f)

dummy<-dummy[,-1]
dim(dummy)


X<- cbind(1, Interest, Unemployment, dummy)
dim(X)
y<-Stock

#Error function = (y-Xbeta)'(y-Xbeta)
#beta.hat=(X'X)^{-1}X'y)


beta.hat <-solve(t(X)%*%X)%*%t(X)%*%y
beta.hat
dim(beta.hat)
model$coefficients

sigmasq.hat <- as.numeric(  t(y-X%*%beta.hat)%*%(y-X%*%beta.hat)/(24-14)  )
sigmasq.hat

#sqrt( (X'X)^{-1}sigmasq.hat )

standard_error=sqrt( diag(solve(t(X)%*%X))*sigmasq.hat ) #standard error of regression coefficient

standard_error

detach(mydata)




#2. (a) ~ (c)

rm(list=ls())

#logistic regression

getwd()
setwd("/Users/himawari/Desktop")
getwd()

mydata=read.csv("BrainTumor.csv",sep=",", header=TRUE)
mydata12=mydata[,-1]

tumor=mydata[,1]

#(a)
scaled=scale(mydata12, center=TRUE, scale=TRUE)
pairs(scaled)



#(b)
mynewdata=cbind(tumor, scaled)


set.seed(100)
sample=sample.int(n = nrow(mydata), size = floor(nrow(mydata)*0.70), replace = FALSE)
train=mydata[sample, ]
test=mydata[-sample, ]


attach(train)
fit=glm(Class~Mean+Variance+Standard.Deviation+Entropy+Skewness+Kurtosis+Contrast+Energy+ASM+Homogeneity+Dissimilarity+Correlation,family="binomial",data=train)
detach(train)


set.seed(100)
sample=sample.int(n = nrow(mynewdata), size = floor(nrow(mynewdata)*0.70), replace = FALSE)
train2=mydata[sample, ]
test2=mydata[-sample, ]

attach(train2)
fit2=glm(Class~Mean+Variance+Standard.Deviation+Entropy+Skewness+Kurtosis+Contrast+Energy+ASM+Homogeneity+Dissimilarity+Correlation,family="binomial",data=train)
detach(train2)

summary(fit)
summary(fit2)

#predicting
testnew=test[,-1]
testnew2=test2[,-1]

eta=predict(fit,newdata=testnew,type="link")
eta2=predict(fit2, newdata=testnew2, type="link")

eta3=predict(fit,newdata=testnew,type="response")
eta4=predict(fit2, newdata=testnew2, type="response")

eta3
eta4

plot(c(1:1129),eta3)

tumor_class3=eta3
tumor_class4=eta4

for (i in 1:1129) {
      if (eta3[i]>=0.5) {
            tumor_class3[i]=1
      } else {
            tumor_class3[i]=0
      }
}

for (i in 1:1129) {
      if (eta3[i]>=0.5) {
            tumor_class4[i]=1
      } else {
            tumor_class4[i]=0
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
