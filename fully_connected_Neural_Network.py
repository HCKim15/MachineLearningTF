import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train.shape
y_train.shape

np.min(x_train), np.max(x_train)

plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap='gray')
    plt.xlabel(y_train[i])
    plt.colorbar()
    
x_train=x_train/(255.0-0.0)
x_test=x_test/255.0


rate=0.1
print(rate*x_train.shape[0])


x_valid=x_train[55000:]
x_train=x_train[:55000]
y_valid=y_train[55000:]
y_train=y_train[:55000]




from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Flatten, Dense, Dropout

model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
#dense: fully connected layer.

#prevent overfitting
#L2 Regularization
model.add(Dense(84, kernel_regularizer=regularizers.l2(0.0001), activation='relu'))


#2 drop out
model.add(Dropout(rate=0.3))


model.add(Dense(10, activation='softmax'))

model.summary()


## Model compilation: 학습 방식 정하기
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])



#3. training step

hist=model.fit(x_train, y_train,
               epochs=3,
               batch_size=100,
               validation_data=(x_valid, y_valid))

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.plot(hist.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(122)
plt.plot(hist.history['accuracy'], 'b-', label='training')
plt.plot(hist.history['val_accuracy'], 'r:',label='validation')
plt.legend()
plt.show()


#l1=model.layers[1]
#l2=model.layers[2]
#l3=model.layers[4]
#w1=l1.get_weights()
#w2=l2.get_weights()
#w3=l3.get_weights()

##test step
#test_loss, test_acc=model.evaluate(x_test, y_test, verbose=2)
#print('test accuracy: ', test_acc)

##prediction

#prediction=model.predict(x_test)
#print(predictions[0])
#print([round(p,4) for p in predictions[0]])
#print([np.argmax(predictions[0]),y_text[0]])


#save data
model.json=model.to_json()
#with open('NN_fashion_mnist,json','w') as json_file:
 #   json_file.write(model_json)
#model.save_weights('NN_fashion_mnist.h5')
#print('Saved model to disk')

##load the model.
# from tensorflow.cpmpat.v2.keras.models import model_from_json

#model json file 열기
# json_file=open('NN_fashion_mnist.json','r')
# loaded_model_json=json_file.read()
# json_file.close()

# json 파일로부터 model 로드하기
# loaded_model=model_from_json(loaded_model_json)
#로드한 model에 weight 로드하기
# loaded_model.load_weights('NN_mnist.h5')

# 모델 컴파일
#loaded_model.compile(optimizers='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#모델 evaluation
#test_loss, test_acc=model.evaluate(x_test, y_test)
#print("Accuracy: %,2f%%" %(test_acc*100))









