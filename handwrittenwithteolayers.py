import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(X_train,Y_train),(X_test,Y_test)=keras.datasets.mnist.load_data()
#print(len(X_train))
X_train=X_train/255
X_test=X_test/255
x_train_flatenned=X_train.reshape(len(X_train),28*28)
x_test_flatenned=X_test.reshape(len(X_test),28*28)
model=keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
    ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
model.fit(x_train_flatenned,Y_train,epochs=5)

model.evaluate(x_test_flatenned,Y_test)
y_predicted=model.predict(x_test_flatenned)
print(np.argmax(y_predicted[0]))
print(y_predicted)
y_predicted_labels=[np.argmax(i) for i in y_predicted]
print(y_predicted_labels)
print(Y_test)
cm=tf.math.confusion_matrix(labels=Y_test,predictions=y_predicted_labels)
print(cm)
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')