model = DenseNet121(weights='densenet.hdf5', include_top=False)
model = Model(inputs=model.input, outputs=Dense(len(labels), activation="sigmoid")(GlobalAveragePooling2D()(model.output)))
model.compile(optimizer='adam', loss=calcloss(negative_freqs, positive_freqs))
fitter = model.fit(traingenerator, validation_data=valgenerator, steps_per_epoch = 1000, epochs = 50)
model.save_weights("cxr_naveen.h5")

plt.plot(fitter.history['loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
plt.show()

predicted_vals = model.predict(testgenerator, steps = len(testgenerator))
{"mode":"full","isActive":false}