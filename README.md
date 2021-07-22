# Chest X-Ray Multi Class Disease Classification using Transfer Learning!
#### First things first, go through my repository and my other resources and appreciate me with your sweet nominal donation :)

<a href="https://www.buymeacoffee.com/therealnavzz" target="_blank"><img src="https://miro.medium.com/max/250/1*_2U75b7qjGfxk8AN6UO0FA.png" alt="Buy Me A Coffee" height="41" width="174"></a><br/>

#### Second,

<a href="https://therealnavzz.medium.com/chest-x-ray-based-multi-class-disease-classification-using-densenet121-transfer-learning-approach-985d734924fa" target="_blank"><img src="https://miro.medium.com/max/8976/1*Ra88BZ-CSTovFS2ZSURBgg.png" alt="My Medium Article" height="41" width="174"></a><br/>

#### Connect with me on:
<a href="https://www.linkedin.com/in/navz/" target="_blank"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS5BVNv--geO9Pok2S_1X6Fyz_I81rbIo55Dw&usqp=CAU" alt="My LinkedIn Profile" height="41" width="174"></a><br/>

Before I start, let me give credits to all my resources :)

**Dataset Link: https://nihcc.app.box.com/v/ChestXray-NIHCC**

**Stanford ML Group: https://stanfordmlgroup.github.io/projects/chexnext/***

**AI for Medical Diagnosis: https://www.coursera.org/learn/ai-for-medical-diagnosis**

In this project, we will be exploring the features of a Chest X-Ray by using various Image Segmentation techniques. Apart from the Exploratory Data Analysis of the Images, we will be focussing on data augmentation, class imbalance and Over Optimistic Model Performance Problem. Moving on, we will be building a model to predict the probability of the given X-Ray to be one among the disease classes. This project uses Transfer Learning - DenseNet121 Pretrained Model for class prediction. 

WIDTH = 512px

HEIGHT = 512px

BATCH_SIZE = 1

### **DenseNet Architecture**

![enter image description here](https://miro.medium.com/max/875/1*04TJTANujOsauo3foe0zbw.jpeg)

![enter image description here](https://miro.medium.com/max/875/1*SSn5H14SKhhaZZ5XYWN3Cg.jpeg)

### **Sample Image Analysis - Gradient Computation**

![enter image description here](https://miro.medium.com/max/875/1*UZEHcLbDoZjksvVRmwSveA.png)

### **Weighed Cross-Entropy Loss Function**
<br/>

![enter image description here](https://miro.medium.com/max/875/1*sLx6K5AIJIOPbEpUf7xU9g.png)

<br/>

![enter image description here](https://miro.medium.com/max/875/1*dpGb-Shqj3097ecn7oGx9w.png)
<br/>

### **ROC Curves for Partially Trained Model**
<br/>

![enter image description here](https://miro.medium.com/max/769/1*s1at0mD1PIE0g8zrtFTT2g.png)


### **ROC Curves using Fully Trained - Using Pretrained Model***

<br/>

![](https://miro.medium.com/max/761/1*bWsojruo2m1s9tj8hzttrg.png)
