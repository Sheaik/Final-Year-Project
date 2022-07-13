import tensorflow as tf
import numpy as np
import tensorflow.keras as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dataset = pd.read_csv('ADANIPORTS.csv')

dataset.info()


m = dataset.index
n = dataset['Open']
plt.figure(figsize=(4,4))
plt.title("Line plot graph",fontsize=20)
plt.xlabel("Day",fontsize=15)
plt.ylabel("Open PRICE",fontsize=15)
plt.plot(m,n,label="Line plot")
plt.legend(loc='best')
plt.show()

m = dataset.index
n = dataset['High']
plt.figure(figsize=(4,4))
plt.title("Bar plot graph",fontsize=20)
plt.xlabel("Day",fontsize=15)
plt.ylabel("High PRICE on that day",fontsize=15)
plt.plot(m,n,label="Line plot")
plt.legend(loc='best')
plt.show()


myval = pd.DataFrame()

myval["price"] = np.nan


actual = dataset

print(actual)

val = dataset
print(val)


val = val.iloc[:,8:9]
print(val)


val['Close'] = val['Close'].astype('int')
print(val)

act = val['Close'].values

print(act)

act2 = act

print(max(act))

print(min(act))


for x in range(0,3322):
  if(act2[[x]]<100):
    g=1;
  elif((act2[[x]]>=100)and(act2[[x]]<150)):
    g=2;
  elif((act2[[x]]>=150)and(act2[[x]]<200)):
    g=3;
  elif((act2[[x]]>=200)and(act2[[x]]<250)):
    g=4;
  elif((act2[[x]]>=250)and(act2[[x]]<300)):
    g=5;
  elif((act2[[x]]>=300)and(act2[[x]]<350)):
    g=6;
  elif((act2[[x]]>=350)and(act2[[x]]<400)):
    g=7;
  elif((act2[[x]]>=400)and(act2[[x]]<450)):
    g=8;
  elif((act2[[x]]>=450)and(act2[[x]]<500)):
    g=9;
  elif((act2[[x]]>=500)and(act2[[x]]<550)):
    g=10;
  elif((act2[[x]]>=550)and(act2[[x]]<600)):
    g=11;
  elif((act2[[x]]>=600)and(act2[[x]]<650)):
    g=12;
  elif((act2[[x]]>=650)and(act2[[x]]<700)):
    g=13;
  elif((act2[[x]]>=700)and(act2[[x]]<750)):
    g=14;
  elif((act2[[x]]>=750)and(act2[[x]]<800)):
    g=15;
  elif((act2[[x]]>=800)and(act2[[x]]<850)):
    g=16;
  elif((act2[[x]]>=850)and(act2[[x]]<900)):
    g=17;
  elif((act2[[x]]>=900)and(act2[[x]]<950)):
    g=18;
  elif((act2[[x]]>=950)and(act2[[x]]<1000)):
    g=19;
  elif((act2[[x]]>=1000)and(act2[[x]]<1050)):
    g=20;
  elif((act2[[x]]>=1050)and(act2[[x]]<1100)):
    g=21;
  elif((act2[[x]]>=1100)and(act2[[x]]<1150)):
    g=22;
  elif((act2[[x]]>=1150)and(act2[[x]]<1200)):
    g=23;
  elif((act2[[x]]>=1200)and(act2[[x]]<1250)):
    g=24;
  elif((act2[[x]]>=1250)and(act2[[x]]<1300)):
    g=25;
  elif(act2[[x]]>1300):
    g=26;
  act2[x]=g
  actual["y"] = ""

print("finished")
#actual['y'].append(act2)
dta = pd.DataFrame(act2, columns = ['y'])

print(dta)


training_dataset = pd.concat([val, dta], axis=0)

print(val.shape)

x = dataset.iloc[:,3:7]
print(x)

print(dta)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, dta, test_size = 0.20, random_state = 121)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#### Create the model
model = tf.models.Sequential()
### Add the layers
model.add(tf.layers.Dense(64, input_dim=4, activation='relu'))     ## input and a hidden layer
model.add(tf.layers.Dense(128,activation='relu'))   ## hidden layer
model.add(tf.layers.Dense(256,activation='relu'))   ## hidden layer
model.add(tf.layers.Dense(512,activation='relu'))   ## hidden layer
model.add(tf.layers.Dense(27,activation='softmax')) ## output layer


###compile the model
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000)

model.save('training.h5')
