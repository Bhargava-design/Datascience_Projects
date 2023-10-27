#!/usr/bin/env python
# coding: utf-8

# In[36]:


import keras


# In[37]:


from keras.datasets import mnist


# In[38]:


from keras.models import Sequential


# In[39]:


from keras.layers import Dense,Dropout,Flatten


# In[40]:


from keras.layers import Conv2D,MaxPooling2D


# In[41]:


from keras import backend as b


# In[42]:


#spliting data into test and train it
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[43]:


x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
input_shape=(28,28,1)


# In[44]:


#converting class vectors into binary
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)


# In[45]:


x_train=x_train.astype("float32")
x_test=x_test.astype("float32")


# In[46]:


x_train/=255
x_test/=255


# In[47]:


batch_size=128
num_classes=10
epochs=10


# In[48]:


model=Sequential()


# In[49]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D


# In[50]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D


# In[51]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[52]:


model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))


# In[53]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[54]:


model.add(Flatten())


# In[55]:


model.add(Dense(128,activation="relu"))


# In[56]:


model.add(Dropout(0.3))


# In[57]:


model.add(Dense(64,activation='relu'))


# In[58]:


model.add(Dropout(0.5))


# In[59]:


model.add(Dense(num_classes,activation='softmax'))


# In[60]:


from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta

model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])


# In[61]:


hist=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))


# In[62]:


#now we have to print the loss
score = model.evaluate(x_test, y_test,verbose=0)
print("loss",score[0])


# In[63]:


print("accuracy:",score[1])


# In[74]:


#now save the model
model.save('mnist.h5')


# In[76]:


model.save('my_model.keras')


# In[ ]:




