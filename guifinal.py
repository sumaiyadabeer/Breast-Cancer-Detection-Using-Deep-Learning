from tkinter import *
from tkinter import filedialog
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import h5py
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model 
from keras.utils.vis_utils import plot_model
import cv2
import os
from sklearn.model_selection import train_test_split
K.set_image_dim_ordering('tf')

model=1
path="img.png"
window = tk.Tk()
window.title("Graphical Interface")
wd=str(window.winfo_screenwidth()-260)+"x"+str(window.winfo_screenheight()-200)
window.geometry(wd)

# load data
numepochs=5
batchsize=128
folder_path = './data500/'
images = []
labels = []
class_label = 0

def load_images_from_folder(folder,class_label):
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder, filename))
		if img is not None:
			img = cv2.resize(img,(140,92))
			img = img.reshape(92,140,3)
			images.append(img)
			labels.append(class_label)
	class_label=class_label+1
	return class_label


# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding="same",input_shape=(92,140,3), activation='relu'))
	#model.add(Conv2D(32, (3, 3), activation='relu',padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu',padding = 'same'))
	#model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
	#model.add(Conv2D(128, (3, 3), activation='relu',padding = 'same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	#model.add(Dense(50, activation='relu'))
	#model.add(Dropout(0.2))
	model.add(Dense(2, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


class_label = 0
class_label = load_images_from_folder(folder_path+'benign',class_label)
class_label = load_images_from_folder(folder_path+'malignant',class_label)

Data = np.asarray(images)
Labels = np.asarray(labels)

X_train,X_test,y_train,y_test=train_test_split(Data,Labels,test_size=0.2,random_state=2)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

tr="train data shape:"+"\n"
tr=tr+"test data shape:"+"\n"
tr=tr+str(X_test.shape)+"\n"
tr=tr+"train label shape:"+"\n"
tr=tr+str(y_train.shape)+"\n"
tr=tr+"test label shape:"+"\n"
tr=tr+str(y_test.shape)+"\n"

def training(X_train, y_train,X_test, y_test,tr):
	global hist
	# build the model
	model = larger_model()
	# Fit the model
	hist=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=numepochs, batch_size=batchsize)
	model.summary()
	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=1,batch_size=batchsize)
	model.save('28april.h5')
	print("Deep Net Accuracy: %.2f%%" % (scores[1]*100))
	
	#create text field
	greetings_disp =tk.Text(master=window,height=20,width=120, fg="midnight blue")
	greetings_disp.grid(column=0,row=3)
	ly= ""
	for layer in model.layers:
		ly=ly+str(layer.name)+"       "  + "				layer input: "+str(layer.input)+"\n"#str(layer.inbound_nodes)+str(layer.outbound_nodes)      "			<<--inputs-->> \n\n" + tr+
	greetings_disp.insert(tk.END ,"			<<--LAYER ARCHITECTURE-->> \n\n" +ly+"\n\n NETWORK is trained with Accuracy  of"+ str(scores[1]*100)+"%")
	return model
	
def graphh():

	# visualizing losses and accuracy
	train_loss=hist.history['loss']
	val_loss=hist.history['val_loss']
	train_acc=hist.history['acc']
	val_acc=hist.history['val_acc']
	xc=range(numepochs)

	plt.figure(1,figsize=(10,5))
	plt.subplot(121)
	plt.plot(xc,train_loss)
	plt.plot(xc,val_loss)
	plt.xlabel('num of Epochs')
	plt.ylabel('loss')
	plt.title('train_loss vs val_loss')
	plt.grid(True)
	plt.legend(['train','val'])
	plt.style.use(['classic'])

	#plt.figure(2,figsize=(7,5))
	plt.subplot(122)
	plt.plot(xc,train_acc)
	plt.plot(xc,val_acc)
	plt.xlabel('num of Epochs')
	plt.ylabel('accuracy')
	plt.title('train_acc vs val_acc')
	plt.grid(True)

	plt.legend(['train','val'],loc=4)
	plt.style.use(['classic'])
	plt.show()
	
def test_test(model):
	test_image = X_test[0:1]
	pa=model.predict(test_image)
	if(model.predict_classes(test_image)==[0]):
		s="BENIGN with Accuracy: " + str(pa[0][0]*100) + "%\n"
	else:
		s="MALIGNANT with Accuracy: "+ str(pa[0][1]*100) + "%\n"

	return s

def test_random(model,path):
	test_image = cv2.imread(path)
	test_image= cv2.resize(test_image,(140,92))
	test_image = test_image.reshape(92,140,3)
	test_image = np.array(test_image)
	test_image = test_image.astype('float32')
	test_image /= 255
	test_image= np.expand_dims(test_image, axis=0)
	pa=model.predict(test_image)
	if(model.predict_classes(test_image)==[0]):
		s="BENIGN with Accuracy: "+ str(pa[0][0]*100) + "%\n"
	else:
		s="MALIGNANT with Accuracy: "+ str(pa[0][1]*100) + "%\n"
	return s

def b_test_test():
	greetings=test_test(model)
	#create text field
	greetings_disp =tk.Text(master=window,height=1,width=45 ,fg="midnight blue")
	greetings_disp.grid(column=0,row=6)
	greetings_disp.insert(tk.END , greetings)

def b_random_test_show():
	global path1
	path=filedialog.askopenfilename(filetypes=(("JPG", ".jpg"), ("All files", "*.*")))
	path1=path
	img=cv2.imread(path1)
	plt.imshow(img)
	plt.show()
	#greetings_disp =tk.Text(master=window,height=0,width=0 ,fg="midnight blue")
	#greetings_disp.grid(column=0,row=10)
	#greetings_disp.insert(tk.END , greetings)

def b_random_test():
	path=filedialog.askopenfilename(filetypes=(("JPG", ".jpg"), ("All files", "*.*")))
	greetings=test_random(model,path)
	#create text field
	greetings_disp =tk.Text(master=window,height=1,width=45 ,fg="midnight blue")
	greetings_disp.grid(column=0,row=12)
	greetings_disp.insert(tk.END , greetings)
	img=cv2.imread(path)
	plt.imshow(img)
	plt.show()

def b_training():
	global model
	model=training(X_train, y_train,X_test, y_test,tr)

labelfont=('Arial', 40, 'bold')
label1 = tk.Label(text="   Breast Cancer Detection using Deep Learning     ", anchor='n', font=labelfont , fg="midnight blue" , bg="mint cream")
label1.grid(column=0,row=0)

#buttons

button1 = tk.Button(text="Start Training" , command= b_training , bg="powder blue")
button1.grid(column=0,row=2)

button2 = tk.Button(text="Test an Image from Dataset" , command=b_test_test , bg="powder blue")
button2.grid(column=0,row=5)

'''button3 = tk.Button(text="Display an Image" , command= b_random_test_show , bg="powder blue")
button3.grid(column=0,row=8)'''

button4 = tk.Button(text="Select an Image for Testing" , command= b_random_test, bg="powder blue")
button4.grid(column=0,row=11)

button5 = tk.Button(text="See Loss and Accuracy plots" , command= graphh, bg="powder blue")
button5.grid(column=0,row=13)

window.mainloop()






