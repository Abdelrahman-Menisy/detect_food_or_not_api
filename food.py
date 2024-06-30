import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import regularizers
from pathlib import Path

# Food images
plt.figure(figsize=(15, 15))
food_images = r"C:\Users\abdel\Downloads\archive (4)\training\food"
for i in range(12):
    file = random.choice(os.listdir(food_images))
    food_image_path = os.path.join(food_images, file)
    img = plt.imread(food_image_path)
    ax = plt.subplot(3, 4, i + 1)
    # plt.imshow(img)
# plt.show()

# Non Food images
plt.figure(figsize=(15, 15))
non_food_images = r"C:\Users\abdel\Downloads\archive (4)\training\non_food"
for i in range(12):
    file = random.choice(os.listdir(non_food_images))
    non_food_image_path = os.path.join(non_food_images, file)
    img = plt.imread(non_food_image_path)
    ax = plt.subplot(3, 4, i + 1)
    # plt.imshow(img)
# plt.show()

# Read Data
image_path = Path(r"C:\Users\abdel\Downloads\archive (4)\training")
img_path = r"C:\Users\abdel\Downloads\archive (4)\training"

label_name = ["Food", "Non_Food"]
image_size = (224, 224)
class_names = os.listdir(image_path)

numberof_images = {}
for class_name in class_names:
    numberof_images[class_name] = len(os.listdir(img_path + "/" + class_name))
images_each_class = pd.DataFrame(numberof_images.values(), index=numberof_images.keys(), columns=["Number of images"])
# print(images_each_class)

# Train DataFrame
train_data_path = Path(r"C:\Users\abdel\Downloads\archive (4)\training")
image_data_path = list(train_data_path.glob(r"**/*.jpg"))
train_label_path = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], image_data_path))
final_train_data = pd.DataFrame({"image_data": image_data_path, "label": train_label_path}).astype("str")
final_train_data = final_train_data.sample(frac=1).reset_index(drop=True)
# print(final_train_data.head())

# Valid DataFrame
valid_data_path = Path(r"C:\Users\abdel\Downloads\archive (4)\validation")
image_data_path = list(valid_data_path.glob(r"**/*.jpg"))
valid_label_path = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], image_data_path))
final_valid_data = pd.DataFrame({"image_data": image_data_path, "label": valid_label_path}).astype("str")
final_valid_data = final_valid_data.sample(frac=1).reset_index(drop=True)
# print(final_valid_data.head())

# Test DataFrame
test_data_path = Path(r"C:\Users\abdel\Downloads\archive (4)\evaluation")
image_data_path = list(test_data_path.glob(r"**/*.jpg"))
test_label_path = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], image_data_path))
final_test_data = pd.DataFrame({"image_data": image_data_path, "label": test_label_path}).astype("str")
final_test_data = final_test_data.sample(frac=1).reset_index(drop=True)
# print(final_test_data.head())

batch_size = 30

traindata_generator = ImageDataGenerator(rescale=1./255, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2,
                                         shear_range=0.2, horizontal_flip=True, validation_split=0.2, fill_mode='nearest')

validdata_generator = ImageDataGenerator(rescale=1./255)
testdata_generator = ImageDataGenerator(rescale=1./255)

train_data_generator = traindata_generator.flow_from_dataframe(dataframe=final_train_data,
                                                               x_col="image_data",
                                                               y_col="label",
                                                               batch_size=batch_size,
                                                               class_mode="categorical",
                                                               target_size=(224, 224),
                                                               color_mode="rgb",
                                                               shuffle=True)

valid_data_generator = validdata_generator.flow_from_dataframe(dataframe=final_valid_data,
                                                               x_col="image_data",
                                                               y_col="label",
                                                               batch_size=batch_size,
                                                               class_mode="categorical",
                                                               target_size=(224, 224),
                                                               color_mode="rgb",
                                                               shuffle=True)

test_data_generator = testdata_generator.flow_from_dataframe(dataframe=final_test_data,
                                                             x_col="image_data",
                                                             y_col="label",
                                                             batch_size=batch_size,
                                                             class_mode="categorical",
                                                             target_size=(224, 224),
                                                             color_mode="rgb",
                                                             shuffle=False)

class_dict = train_data_generator.class_indices
class_list = list(class_dict.keys())
# print(f'class_list : {class_list}')

train_number = train_data_generator.samples
valid_number = valid_data_generator.samples

# print("Train number : ", train_number)
# print("Valid number : ", valid_number)

# Create DenseNet121 Model
dense121_model = tf.keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = dense121_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
x = Dropout(0.5)(x)
prediction = Dense(2, activation='softmax')(x)
model = Model(inputs=dense121_model.input, outputs=prediction)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train Model
tensor_board = TensorBoard(log_dir="logs")
check_point = ModelCheckpoint("denseNet121.h5", monitor="val_accuracy", mode="auto", verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.3, patience=5, min_delta=0.01, mode="auto", verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = model.fit(train_data_generator,
                    steps_per_epoch=train_number // batch_size,
                    validation_data=valid_data_generator,
                    validation_steps=valid_number // batch_size,
                    shuffle=True,
                    epochs=50,
                    callbacks=[tensor_board, check_point, reduce_lr, early_stopping])

# Accuracy Graph
plt.figure(figsize=(10, 10))
plt.style.use("classic")
plt.plot(history.history['accuracy'], marker=">", color="red")
plt.plot(history.history['val_accuracy'], marker="*", color="green")
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Loss Graph
plt.figure(figsize=(10, 10))
plt.style.use("classic")
plt.plot(history.history['loss'], marker="P", color="navy")
plt.plot(history.history['val_loss'], marker="H", color="crimson")
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Model Prediction
prediction = model.predict(test_data_generator)
prediction = np.argmax(prediction, axis=1)
map_label = dict((m, n) for n, m in (test_data_generator.class_indices).items())
final_predict = pd.Series(prediction).map(map_label).values
y_test = list(final_test_data.label)

# Save the trained model
model.save('food_classifier_model.h5')

# Load the saved model
saved_model = load_model('food_classifier_model.h5')

# Function to predict if an image is food or not
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = saved_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    class_label = class_list[predicted_class[0]]
    return class_label

# Test the function
test_image_path = r"C:\Users\abdel\OneDrive\Desktop\caption.jpg" # Replace with the path to your test image
result = predict_image(test_image_path)
print(f'The image is classified as: {result}')
