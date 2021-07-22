from keras.preprocessing.image import ImageDataGenerator

#To standardize the images, we use samplewise_center = True and samplewize_std_normalization = True
traingen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization= True)
traingenerator = traingen.flow_from_dataframe(
        dataframe=trainset,
        directory="images",
        x_col="Image",
        y_col= labels,
        class_mode="raw",
        batch_size= 1,
        shuffle=False,
        target_size=(512,512)
)

#Standardizing featurewise as we don't process images as groups rather one-by-one
imagegen = ImageDataGenerator().flow_from_dataframe(dataframe = trainset, directory = "images", x_col = "Image", y_col = labels, class_mode = "raw", batch_size= 1, shuffle=False, target_size=(512,512))
train_sample = imagegen.next()[0]
imagegen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization= True)
imagegen.fit(train_sample)
valgenerator = imagegen.flow_from_dataframe(dataframe = valset, directory = "images", x_col = "Image", y_col = labels, class_mode = "raw", batch_size= 1, shuffle=False, target_size=(512,512))
testgenerator = imagegen.flow_from_dataframe(dataframe = testset, directory = "images", x_col = "Image", y_col = labels, class_mode = "raw", batch_size= 1, shuffle=False, target_size=(512,512))

#Taking a random sample standardized image
item, value = traingenerator.__getitem__(num)
plt.figure(figsize=(15, 15))
plt.imshow(item[0], cmap = 'gray')
plt.colorbar()

#Plotting the histogram of original and standardized pixel intensities
fig, ax = plt.subplots(figsize=(25, 10))
plt.xlabel("Pixel Values")
print("Mean of Pixel Values - Standardized: ", item[0].mean())
print("Standard Deviation of Pixel Values - Standardized: ", item[0].std())
print("Mean of Pixel Values - Sample: ", sample.mean())
print("Standard Deviation of Pixel Values - Sample: ", sample.std())
sns.histplot(item[0].ravel(), ax = ax, kde = False)
sns.histplot(sample.ravel(), ax = ax, kde = False, color = "red")
{"mode":"full","isActive":false}