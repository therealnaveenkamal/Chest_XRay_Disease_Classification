#!/usr/bin/env python
# coding: utf-8

# # Importing Packages

# In[1]:


import pandas as pd
import os
from matplotlib import rcParams
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow.keras.backend as K
from keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
from tensorflow.python.framework.ops import disable_eager_execution
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity
disable_eager_execution()


# # Dataset obtained from: NIH
# https://nihcc.app.box.com/v/ChestXray-NIHCC

# In[2]:


dataframe = pd.read_csv("Data_Entry_2017_v2020.csv")


# In[3]:


dataframe.head()


# In[4]:


#Enumerating all column names
columns = ["Image"]
for i in dataframe["Finding Labels"].values:
    for j in i.split("|"):
        if j not in columns:
            columns.append(j)
labels = columns.copy()
labels.remove("Image")
labels.remove("No Finding")
columns.remove("No Finding")


# In[5]:


#Taking the first 90000 images from the master table as the train dataset
trainset = pd.DataFrame(columns = columns)
for i in range(90000):
    col = [0]*len(columns)
    col[0] = dataframe["Image Index"][i]
    count = 1
    for j in columns[1:]:
        if(j in dataframe["Finding Labels"][i]):
            col[count] = 1
        count+=1
    trainset.loc[len(trainset)] = col


# In[6]:


#Taking the next 10000 images from the master table as the validation dataset
valset = pd.DataFrame(columns = columns)
for i in range(90000, 100000):
    col = [0]*len(columns)
    col[0] = dataframe["Image Index"][i]
    count = 1
    for j in columns[1:]:
        if(j in dataframe["Finding Labels"][i]):
            col[count] = 1
        count+=1
    valset.loc[len(valset)] = col


# In[7]:


#Taking the next 12000 images from the master table as the test dataset
testset = pd.DataFrame(columns = columns)
for i in range(100000, 112000):
    col = [0]*len(columns)
    col[0] = dataframe["Image Index"][i]
    count = 1
    for j in columns[1:]:
        if(j in dataframe["Finding Labels"][i]):
            col[count] = 1
        count+=1
    testset.loc[len(testset)] = col


# In[8]:


testset1 = pd.DataFrame(columns = columns)
for i in range(50000, 50500):
    col = [0]*len(columns)
    col[0] = dataframe["Image Index"][i]
    count = 1
    for j in columns[1:]:
        if(j in dataframe["Finding Labels"][i]):
            col[count] = 1
        count+=1
    testset1.loc[len(testset1)] = col


# In[9]:


img_dir = "images"
plt.figure(figsize = (15,15))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(plt.imread(os.path.join(img_dir, trainset["Image"][i])), cmap = "gray")
    plt.title(dataframe[dataframe["Image Index"] == trainset["Image"][i]].values[0][1])
plt.tight_layout()


# In[10]:


trainset


# # Checking for Patient Overlap
# In order to prevent **Overly optimistic prediction model**

# In[11]:


def isOverlap(s1, s2):
    total = set(s1).intersection(set(s2))
    return [len(total), total]


# In[12]:


def overlapcheck(trainset, valset, testset):
    patid_train = []
    patid_val = []
    patid_test = []
    for name in trainset['Image'].values:
        patid_train.append(int(name.split("_")[0]))

    for name in valset['Image'].values:
        patid_val.append(int(name.split("_")[0]))

    for name in testset['Image'].values:
        patid_test.append(int(name.split("_")[0]))
    trte = isOverlap(patid_train, patid_test)
    teva = isOverlap(patid_test, patid_val)
    trva = isOverlap(patid_train, patid_val)
    print("Patient Overlap - Train and Test: ", trte[0])
    print("Patient Overlap - Test and Validation: ", teva[0])
    print("Patient Overlap - Train and Validation: ", trva[0])
    return trte, teva, trva


# In[13]:


trte, teva, trva = overlapcheck(trainset, valset, testset)


# # Analysis of a Sample Image

# In[18]:


num = np.random.randint(trainset.shape[0])
sample = plt.imread(os.path.join(img_dir,trainset.iloc[[num]]["Image"].values[0]))
plt.figure(figsize=(15, 15))
plt.title(dataframe[dataframe["Image Index"] == trainset.iloc[[num]]["Image"].values[0]].values[0][1])
plt.imshow(sample, cmap = 'gray')
plt.colorbar()
trainset.iloc[[num]]


# In[19]:


num


# In[20]:


print("Maximum Pixel Value: ", sample.max())
print("Minimum Pixel Value: ", sample.min())
print(f"Image dimension: {sample.shape[0]} x {sample.shape[1]} ")


# In[21]:


plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(25, 10))
plt.xlabel("Pixel Values")
print("Mean - Pixel Value: ", sample.mean())
print("Std Deviation Pixel Value: ", sample.std())
sns.histplot(sample.ravel(), ax = ax, kde = True)


# In[22]:


from keras.preprocessing.image import ImageDataGenerator
traingen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization= True)
traingenerator = traingen.flow_from_dataframe(
        dataframe=trainset,
        directory="images",
        x_col="Image",
        y_col= labels,
        class_mode="raw",
        batch_size= 1,
        shuffle=True,
        target_size=(512,512)
)


# In[23]:


imagegen = ImageDataGenerator().flow_from_dataframe(dataframe = trainset, directory = "images", x_col = "Image", y_col = labels, class_mode = "raw", batch_size= 1, shuffle=True, target_size=(512,512))
train_sample = imagegen.next()[0]
imagegen1 = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization= True)
imagegen1.fit(train_sample)

valgenerator = imagegen1.flow_from_dataframe(dataframe = valset, directory = "images", x_col = "Image", y_col = labels, class_mode = "raw", batch_size= 1, shuffle=True, target_size=(512,512))
testgenerator = imagegen1.flow_from_dataframe(dataframe = testset, directory = "images", x_col = "Image", y_col = labels, class_mode = "raw", batch_size= 1, shuffle=True, target_size=(512,512))


# In[24]:


testgenerator1 = imagegen1.flow_from_dataframe(dataframe = testset1, directory = "images", x_col = "Image", y_col = labels, class_mode = "raw", batch_size= 10, shuffle=True, target_size=(512,512))


# In[25]:


item, value = traingenerator.__getitem__(num)


# In[26]:


plt.figure(figsize=(15, 15))
plt.imshow(item[0], cmap = 'gray')
plt.colorbar()


# In[27]:


fig, ax = plt.subplots(figsize=(25, 10))
plt.xlabel("Pixel Values")
print("Mean of Pixel Values - Standardized: ", item[0].mean())
print("Standard Deviation of Pixel Values - Standardized: ", item[0].std())
print("Mean of Pixel Values - Sample: ", sample.mean())
print("Standard Deviation of Pixel Values - Sample: ", sample.std())
sns.histplot(item[0].ravel(), ax = ax, kde = False)
sns.histplot(sample.ravel(), ax = ax, kde = False, color = "red")


# # Build the Model

# In[28]:


plt.rcParams.update({'font.size': 12})


# In[29]:


positive_freqs = np.mean(traingenerator.labels, axis = 0)
negative_freqs = 1 - positive_freqs


# In[30]:


data = {
    'Class': labels,
    'Positive': positive_freqs,
    'Negative':negative_freqs
}


# In[31]:


negative_freqs


# In[32]:


positive_freqs


# In[33]:


X_axis = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(X_axis-0.2, data['Positive'], width=0.4, color='b', label = "Positive")
ax.bar(X_axis+0.2, data['Negative'], width=0.4, color='r', label = 'Negative')
plt.xticks(X_axis, labels, rotation = 90)
plt.legend()
plt.figure(figsize=(20,15))


# In[34]:


data = {
    'Class': labels,
    'Positive': positive_freqs*negative_freqs,
    'Negative':negative_freqs*positive_freqs
}


# In[35]:


X_axis = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(X_axis-0.2, data['Positive'], width=0.4, color='b', label = "Positive")
ax.bar(X_axis+0.2, data['Negative'], width=0.4, color='r', label = 'Negative')
plt.xticks(X_axis, labels, rotation = 90)
plt.legend()
plt.figure(figsize=(20,15))


# In[36]:


def calcloss(positivewt, negativewt, al=1e-7):
    def weighted_loss(y_true, y_pred):
        loss = 0.0
        for i in range(len(positivewt)):
            loss += -((positivewt[i] * K.transpose(y_true)[i] * K.log(K.transpose(y_pred)[i] + al))+(negativewt[i]*(1 - K.transpose(y_true)[i])*K.log(1 - K.transpose(y_pred)[i] + al)))
        return K.mean(loss)
    return weighted_loss


# In[37]:


model = DenseNet121(weights='densenet.hdf5', include_top=False)
model = Model(inputs=model.input, outputs=Dense(len(labels), activation="sigmoid")(GlobalAveragePooling2D()(model.output)))
model.summary()


# In[38]:


model.compile(optimizer='adam', loss=calcloss(negative_freqs, positive_freqs))


# In[39]:


fitter = model.fit(traingenerator, validation_data=valgenerator, steps_per_epoch = 1000, epochs = 50)


# In[40]:


model.save_weights("cxr_naveen.h5")


# In[41]:


plt.plot(fitter.history['loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
plt.show()


# In[42]:


predicted_vals = model.predict(testgenerator, steps = len(testgenerator))


# In[43]:


predicted_vals1 = model.predict(testgenerator1, steps = len(testgenerator1))


# In[44]:


auc_roc_vals = []
for i in range(len(labels)):
    try:
        gt = np.array(testgenerator.labels[:, i])
        pred = predicted_vals[:,i]
        gt = gt.astype('int64')
        gt = gt.reshape(-1, 1)
        auc_roc = roc_auc_score(gt, pred)
        print(auc_roc)
        auc_roc_vals.append(auc_roc)
        fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
        plt.figure(1, figsize=(10, 10))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_rf, tpr_rf,
                 label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
    except ValueError:
        pass
plt.show()


# In[45]:


auc_roc_vals = []
for i in range(len(labels)):
    try:
        gt = np.array(testgenerator1.labels[:, i])
        pred = predicted_vals1[:,i]
        gt = gt.astype('int64')
        gt = gt.reshape(-1, 1)
        auc_roc = roc_auc_score(gt, pred)
        print(auc_roc)
        auc_roc_vals.append(auc_roc)
        fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
        plt.figure(1, figsize=(10, 10))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_rf, tpr_rf,
                 label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
    except ValueError:
        pass
plt.show()


# In[46]:


labels2show = np.take(labels, np.argsort(auc_roc_vals)[::-1])[:4]


# In[47]:


labels2show


# In[48]:


def get_mean_std_per_batch(image_path, df, H=400, W=400):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image"].values):
        # path = image_dir + img
        sample_data.append(
            np.array(image.load_img(image_path, target_size=(H, W))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std

def load_image(img, image_dir, df, preprocess=True, H=400, W=400):
    """Load and preprocess image."""
    img_path = image_dir + img
    mean, std = get_mean_std_per_batch(img_path, df, H=H, W=W)
    x = image.load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x


def grad_cam(input_model, image, cls, layer_name, H=400, W=400):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def compute_gradcam(model, img, image_dir, df, labels, selected_labels,
                    layer_name='bn'):
    preprocessed_input = load_image(img, image_dir, df)
    predictions = model.predict(preprocessed_input)

    print("Loading original image")
    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(load_image(img, image_dir, df, preprocess=False), cmap='gray')

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            plt.subplot(151 + j)
            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
            plt.axis('off')
            plt.imshow(load_image(img, image_dir, df, preprocess=False),
                       cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
            j += 1


# In[50]:


compute_gradcam(model, '00000001_001.png', "images/", trainset, labels, labels2show)


# In[52]:


compute_gradcam(model, '00000013_024.png', "images/", trainset, labels, labels2show)


# In[ ]:




