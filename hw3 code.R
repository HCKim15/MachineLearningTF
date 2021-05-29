rm(list=ls())


#1 Using Kohonen SOM on wine data

library("kohonen")
data("wines")
str(wines)
head(wines)


#for 2x2
set.seed(10)
som.wines = som(scale(wines), grid=somgrid(2,2,"hexagonal"))
som.wines
dim(getCodes(som.wines))

plot(som.wines, main="Wine data Kohonen SOM")

par(mfrow=c(1,1))
plot(som.wines, type="changes", main="Wine data: SOM")

#for 4x4
set.seed(10)
som.wines = som(scale(wines), grid=somgrid(4,4,"hexagonal"))
som.wines
dim(getCodes(som.wines))

plot(som.wines, main="Wine data Kohonen SOM")

par(mfrow=c(1,1))
plot(som.wines, type="changes", main="Wine data: SOM")


#for 6x6
set.seed(10)
som.wines = som(scale(wines), grid=somgrid(6,6,"hexagonal"))
som.wines
dim(getCodes(som.wines))

plot(som.wines, main="Wine data Kohonen SOM")

par(mfrow=c(1,1))
plot(som.wines, type="changes", main="Wine data: SOM")

#for 10x10
set.seed(10)
som.wines = som(scale(wines), grid=somgrid(10,10,"hexagonal"))
som.wines
dim(getCodes(som.wines))

plot(som.wines, main="Wine data Kohonen SOM")

par(mfrow=c(1,1))
plot(som.wines, type="changes", main="Wine data: SOM")

#for 14x14 this gives us error as the there are more grids than the number of observations

# set.seed(10)
# som.wines = som(scale(wines), grid=somgrid(14,14,"hexagonal"))
# som.wines
# dim(getCodes(som.wines))
# 
# plot(som.wines, main="Wine data Kohonen SOM")
# 
# par(mfrow=c(1,1))
# plot(som.wines, type="changes", main="Wine data: SOM")



#2 Using Backprop.txt code to do NN

rm(list=ls())

rm(list=setdiff(ls(), c("err", "err1")))

numIter = 1000

#(a) I created forwardProp function based on the neural network in this HW problem as follows:

### Initial settings 

# Initialize parameters
w1 = 0.7
w2 = 0.9
w3 = 0.5
w4 = 0.3
w5 = 0.8
w6 = 0.5

w = c(w1, w2, w3, w4, w5, w6)

# input and target values
input1 = 2.5
input2 = 0.5
input = c(input1, input2)

out1 = 1
out2 = 1
out = c(out1, out2)


### define sigmoid activation functions
sigmoid = function(z){ 
      return( 1/(1+exp(-z)) )
}

forwardProp = function(input, w){
      # input to hidden layer
      neth1 = w[1]*input[1]
      neth2 = w[2]*input[1] + w[3]*input[2]
      outh1 = sigmoid(neth1)
      outh2 = sigmoid(neth2)
      
      # hidden layer to output layer
      neto1 = w[4]*outh1
      neto2 = w[5]*outh1 + w[6]*outh2
      outo1 = sigmoid(neto1)
      outo2 = sigmoid(neto2)
      
      res = c(outh1, outh2, outo1, outo2)
      return(res)
}

error = function(res, out){ 
      err = 0.5*(out[1] - res[3])^2 + 0.5*(out[2] - res[4])^2 
      return(err)
}

#(b) I created back propagation update code based on the neural network in this HW problem as follows:

# set learning rate

gamma = 0.1
gamma = 0.6
gamma = 1.2

### Implement Forward-backward propagation
err = c()
err1 = c()
err2 = c()

#for gamma = 0.1
for(i in 1:numIter){
      
      ### forward
      res = forwardProp(input, w)
      outh1 = res[1]; outh2 = res[2]; outo1 = res[3]; outo2 = res[4]
      
      ### compute error
      err[i] = error(res, out)
      
      ### backward propagation
      ## update w_4, w_5, w_6
      
      # compute dE_dw4
      dE_douto1 = -( out[1] - outo1 )
      douto1_dneto1 = outo1*(1-outo1)
      dneto1_dw4 = outh1
      dE_dw4 = dE_douto1*douto1_dneto1*dneto1_dw4
      
      # compute dE_dw5
      dE_douto2 = -( out[2] - outo2 )
      douto2_dneto2 = outo2*(1-outo2)
      dneto2_dw5 = outh1
      dE_dw5 = dE_douto2*douto2_dneto2*dneto2_dw5
      
      # compute dE_dw6
      dneto2_dw6 = outh2
      dE_dw6 = dE_douto2*douto2_dneto2*dneto2_dw6
      
      ## update w_1, w_2, w_3
      # compute dE_douth1 first
      dneto1_douth1 = w4
      dneto2_douth1 = w5
      dE_douth1 = dE_douto1*douto1_dneto1*dneto1_douth1 + dE_douto2*douto2_dneto2*dneto2_douth1
      
      # compute dE_douth2 first
      dneto2_douth2 = w6
      dE_douth2 = dE_douto2*douto2_dneto2*dneto2_douth2 
      
      # compute dE_dw1    
      douth1_dneth1 = outh1*(1-outh1)
      dneth1_dw1 = input[1]
      dE_dw1 = dE_douth1*douth1_dneth1*dneth1_dw1
      
      # compute dE_dw2
      douth2_dneth2 = outh2*(1-outh2)
      dneth2_dw2 = input[1]
      dE_dw2 = dE_douth2*douth2_dneth2*dneth2_dw2
      
      # compute dE_dw3
      dneth2_dw3 = input[2] 
      dE_dw3 = dE_douth2*douth2_dneth2*dneth2_dw3

      ### update all parameters via a gradient descent 
      w1 = w1 - gamma*dE_dw1
      w2 = w2 - gamma*dE_dw2
      w3 = w3 - gamma*dE_dw3
      w4 = w4 - gamma*dE_dw4
      w5 = w5 - gamma*dE_dw5
      w6 = w6 - gamma*dE_dw6
      
      w = c(w1, w2, w3, w4, w5, w6)
      
      print(i)
      
}


#for gamma = 0.6
for(i in 1:numIter){
      
      ### forward
      res = forwardProp(input, w)
      outh1 = res[1]; outh2 = res[2]; outo1 = res[3]; outo2 = res[4]
      
      ### compute error
      err1[i] = error(res, out)
      
      ### backward propagation
      ## update w_4, w_5, w_6
      
      # compute dE_dw4
      dE_douto1 = -( out[1] - outo1 )
      douto1_dneto1 = outo1*(1-outo1)
      dneto1_dw4 = outh1
      dE_dw4 = dE_douto1*douto1_dneto1*dneto1_dw4
      
      # compute dE_dw5
      dE_douto2 = -( out[2] - outo2 )
      douto2_dneto2 = outo2*(1-outo2)
      dneto2_dw5 = outh1
      dE_dw5 = dE_douto2*douto2_dneto2*dneto2_dw5
      
      # compute dE_dw6
      dneto2_dw6 = outh2
      dE_dw6 = dE_douto2*douto2_dneto2*dneto2_dw6
      
      ## update w_1, w_2, w_3
      # compute dE_douth1 first
      dneto1_douth1 = w4
      dneto2_douth1 = w5
      dE_douth1 = dE_douto1*douto1_dneto1*dneto1_douth1 + dE_douto2*douto2_dneto2*dneto2_douth1
      
      # compute dE_douth2 first
      dneto2_douth2 = w6
      dE_douth2 = dE_douto2*douto2_dneto2*dneto2_douth2 
      
      # compute dE_dw1    
      douth1_dneth1 = outh1*(1-outh1)
      dneth1_dw1 = input[1]
      dE_dw1 = dE_douth1*douth1_dneth1*dneth1_dw1
      
      # compute dE_dw2
      douth2_dneth2 = outh2*(1-outh2)
      dneth2_dw2 = input[1]
      dE_dw2 = dE_douth2*douth2_dneth2*dneth2_dw2
      
      # compute dE_dw3
      dneth2_dw3 = input[2] 
      dE_dw3 = dE_douth2*douth2_dneth2*dneth2_dw3
      
      ### update all parameters via a gradient descent 
      w1 = w1 - gamma*dE_dw1
      w2 = w2 - gamma*dE_dw2
      w3 = w3 - gamma*dE_dw3
      w4 = w4 - gamma*dE_dw4
      w5 = w5 - gamma*dE_dw5
      w6 = w6 - gamma*dE_dw6
      
      w = c(w1, w2, w3, w4, w5, w6)
      
      print(i)
      
}


#for gamma = 1.2
for(i in 1:numIter){
      
      ### forward
      res = forwardProp(input, w)
      outh1 = res[1]; outh2 = res[2]; outo1 = res[3]; outo2 = res[4]
      
      ### compute error
      err2[i] = error(res, out)
      
      ### backward propagation
      ## update w_4, w_5, w_6
      
      # compute dE_dw4
      dE_douto1 = -( out[1] - outo1 )
      douto1_dneto1 = outo1*(1-outo1)
      dneto1_dw4 = outh1
      dE_dw4 = dE_douto1*douto1_dneto1*dneto1_dw4
      
      # compute dE_dw5
      dE_douto2 = -( out[2] - outo2 )
      douto2_dneto2 = outo2*(1-outo2)
      dneto2_dw5 = outh1
      dE_dw5 = dE_douto2*douto2_dneto2*dneto2_dw5
      
      # compute dE_dw6
      dneto2_dw6 = outh2
      dE_dw6 = dE_douto2*douto2_dneto2*dneto2_dw6
      
      ## update w_1, w_2, w_3
      # compute dE_douth1 first
      dneto1_douth1 = w4
      dneto2_douth1 = w5
      dE_douth1 = dE_douto1*douto1_dneto1*dneto1_douth1 + dE_douto2*douto2_dneto2*dneto2_douth1
      
      # compute dE_douth2 first
      dneto2_douth2 = w6
      dE_douth2 = dE_douto2*douto2_dneto2*dneto2_douth2 
      
      # compute dE_dw1    
      douth1_dneth1 = outh1*(1-outh1)
      dneth1_dw1 = input[1]
      dE_dw1 = dE_douth1*douth1_dneth1*dneth1_dw1
      
      # compute dE_dw2
      douth2_dneth2 = outh2*(1-outh2)
      dneth2_dw2 = input[1]
      dE_dw2 = dE_douth2*douth2_dneth2*dneth2_dw2
      
      # compute dE_dw3
      dneth2_dw3 = input[2] 
      dE_dw3 = dE_douth2*douth2_dneth2*dneth2_dw3
      
      ### update all parameters via a gradient descent 
      w1 = w1 - gamma*dE_dw1
      w2 = w2 - gamma*dE_dw2
      w3 = w3 - gamma*dE_dw3
      w4 = w4 - gamma*dE_dw4
      w5 = w5 - gamma*dE_dw5
      w6 = w6 - gamma*dE_dw6
      
      w = c(w1, w2, w3, w4, w5, w6)
      
      print(i)
      
}

ts.plot( err, col="red", main="error rate" )
lines(err1, col="blue")
lines(err2, col="green")

pred = forwardProp(input, w)
pred[3:4]



