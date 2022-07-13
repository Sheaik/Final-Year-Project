import tensorflow as tf
import numpy as np
import tensorflow.keras as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
import pandas as pd

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import plot_confusion_matrix

from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def votingclassifier(r1,r2,r3,r4,r5,r6,r7):
    votes =[r1,r2,r3,r4,r5,r6,r7]
     
    #Count the votes for persons and stores in the dictionary
    vote_count=Counter(votes)
     
    #Find the maximum number of votes
    max_votes=max(vote_count.values())
     
    #Search for people having maximum votes and store in a list
    lst=[i for i in vote_count.keys() if vote_count[i]==max_votes]
     
    #Sort the list and print lexicographical smallest name
    print(sorted(lst)[0])
    return(sorted(lst)[0])



    
dataset = pd.read_csv('ADANIPORTS.csv')

dataset.info()


m = dataset.index
n = dataset['Open']
plt.figure(figsize=(4,4))
plt.title("Line Plot Graph",fontsize=20)
plt.xlabel("Day",fontsize=15)
plt.ylabel("Open PRICE",fontsize=15)
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

model1 = tf.models.load_model('training.h5')

y_pred_test_Forestreg= model1.predict(X_test)

g=plt.scatter(X_train, y_test)
g.axes.set_yscale('log')
g.axes.set_xscale('log')
g.axes.set_xlabel('True Values ')
g.axes.set_ylabel('Predictions ')
g.axes.axis('equal')
g.axes.axis('square')
from sklearn import tree

model2=tree.DecisionTreeClassifier()
print("Training Started")
model2.fit(X_train, y_train)
print("Training Ended")
y_pred = model2.predict(X_test)

#confussion Matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

cm_HybridEnsembler = confusion_matrix(y_test, y_pred)
print("Confussion Matrix :")
print(cm_HybridEnsembler)
print(" ")

plot_confusion_matrix(model2, X_test, y_test)  
plt.show()
acc = accuracy_score(y_test, y_pred)
print("Accuracy_score :")
print(acc)
print(" ")

testy = y_test
yhat_classes = y_pred
precision = precision_score(testy, yhat_classes, average='micro')
print('Precision: %f' % precision)
recall = recall_score(testy, yhat_classes,average='micro')
print('Recall: %f' % recall)
f1 = f1_score(testy, yhat_classes, average='micro')
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)

from tkinter import *
from functools import partial

tkWindow = Tk() 
def predict(prevClose, open, high, low):
  userList=[prevClose.get(), open.get(), high.get(), low.get()]
  list_of_floats = [float(item) for item in userList]

  print(list_of_floats)

  y1=model1.predict(np.array(list_of_floats).reshape(1, -1))
  r1 = np.argmax(y1)
  y2=model1.predict(np.array(list_of_floats).reshape(1, -1))
  r2 = np.argmax(y2)
  y3=model1.predict(np.array(list_of_floats).reshape(1, -1))
  r3 = np.argmax(y3)
  y4=model2.predict(np.array(list_of_floats).reshape(1, -1))
  r4 = y4[0]
  y5=model2.predict(np.array(list_of_floats).reshape(1, -1))
  r5 = y5[0]
  y6=model2.predict(np.array(list_of_floats).reshape(1, -1))
  r6 = y6[0]
  y7=model2.predict(np.array(list_of_floats).reshape(1, -1))
  r7 = y7[0]
  
  ynew = votingclassifier(r1,r2,r3,r4,r5,r6,r7)
  print("")
  import random
  g=ynew
  if(g==1):
    data = random.randint(90,100)
  elif(g==2):
    data = random.randint(100,150)
  elif(g==3):
    data = random.randint(150,200)
  elif(g==4):
    data = random.randint(200,250)
  elif(g==5):
    data = random.randint(250,300)
  elif(g==6):
    data = random.randint(300,350)
  elif(g==7):
    data = random.randint(350,400)
  elif(g==8):
    data = random.randint(400,450)
  elif(g==9):
    data = random.randint(450,500)
  elif(g==10):
    data = random.randint(500,550)
  elif(g==11):
    data = random.randint(550,600)
  elif(g==12):
    data = random.randint(600,650)
  elif(g==13):
    data = random.randint(650,700)
  elif(g==14):
    data = random.randint(700,750)
  elif(g==15):
    data = random.randint(750,800)
  elif(g==16):
    data = random.randint(800,850)
  elif(g==17):
    data = random.randint(850,900)
  elif(g==18):
    data = random.randint(900,950)
  elif(g==19):
    data = random.randint(950,1000)
  elif(g==20):
    data = random.randint(1000,1050)
  elif(g==21):
    data = random.randint(1050,1100)
  elif(g==22):
    data = random.randint(1100,1150)
  elif(g==23):
    data = random.randint(1150,1200)
  elif(g==24):
    data = random.randint(1200,1250)
  elif(g==25):
    data = random.randint(1250,1300)
  elif(g==26):
    data = random.randint(1300,1350)
  
  #print("The Predicted Stock closing price is : ",data, "Rupees")
  val = "The Predicted Stock closing price is : "+str(data)+ " Rupees"
  close=Tk()
  close.title("Stock market prediction")
  close.configure(bg='yellow')  
  Label(close, text=val,bg='yellow',fg='red').grid(column=0,row=0,padx=20,pady=30)  




def open_win():
  tkWindow.destroy()
  #Set the geometry of tkinter frame
  new= Tk()
  new.geometry("750x250")
  new.title("ANN-DT MODEL")
  new.configure(bg='green')  
  #Create a Label in New window
  Label(new, text="STOCK PREDICTION", font=('Helvetica 17 bold'),bg="green").grid(row=0, column=7)
  prevCloseLabel = Label(new, text="Prev Close",bg='green',fg="orange").grid(row=3, column=7)
  prevClose = StringVar()
  prevCloseEntry = Entry(new, textvariable=prevClose).grid(row=3, column=8) 

  Label(new, text="",bg="green").grid(row=4, column=0)

  openLabel = Label(new, text="Open",bg='green',fg="orange").grid(row=5, column=7)
  open = StringVar()
  openEntry = Entry(new, textvariable=open).grid(row=5, column=8) 

  Label(new, text="",bg="green").grid(row=6, column=0)

  highLabel = Label(new, text="High",bg='green',fg="orange").grid(row=7, column=7)
  high = StringVar()
  highEntry = Entry(new, textvariable=high).grid(row=7, column=8) 

  Label(new, text="",bg="green").grid(row=8, column=0)

  lowLabel = Label(new, text="Low",bg='green',fg="orange").grid(row=9, column=7)
  low = StringVar()
  lowEntry = Entry(new, textvariable=low).grid(row=9, column=8) 

  Label(new, text="",bg="green").grid(row=10, column=0)

  predict1 = partial(predict, prevClose, open, high, low)

  predictButton = Button(new, text="Predict",activebackground='#345',activeforeground='white',relief="groove", command=predict1).grid(row=11, column=7)  

def validateLogin(username, password):
  if(username.get()=="user" and password.get()=="12345678"):
    open_win()
  Label(tkWindow,text="Invalid Credentials!!!",fg="red", font=("calibri", 11),bg="blue").grid(row=9, column=0)
  return False

#LOGIN window
 
tkWindow.geometry('500x500')
tkWindow.configure(bg='blue')  
tkWindow.title('Stock Prediction')

Label(tkWindow, text="Please enter details below", bg="blue", fg="yellow").grid(row=0, column=0)
Label(tkWindow, text="",bg="blue").grid(row=1, column=0)

#username label and text entry box
usernameLabel = Label(tkWindow, text="User Name",bg="blue",fg="yellow").grid(row=3, column=0)
username = StringVar()
usernameEntry = Entry(tkWindow, textvariable=username).grid(row=3, column=1)  
Label(tkWindow, text="",bg="blue").grid(row=4, column=0)
#password label and password entry box
passwordLabel = Label(tkWindow,text="Password",bg="blue",fg="yellow").grid(row=6, column=0)  
password = StringVar()
passwordEntry = Entry(tkWindow, textvariable=password, show='*').grid(row=6, column=1)  
Label(tkWindow, text="",bg="blue").grid(row=7, column=0)
validateLogin = partial(validateLogin, username, password)

#login button
loginButton = Button(tkWindow, text="Login", activebackground='#345',activeforeground='white',relief="groove", command=validateLogin).grid(row=8, column=0)

tkWindow.mainloop()





