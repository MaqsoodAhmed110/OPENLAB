import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNEL = 3
EPOCH = 3

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "D:\PlantVillage\strawberryplant",
    shuffle = True,
    image_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE
)

class_names = dataset.class_names
class_names

len(dataset)

#Visualizing the data
plt.figure(figsize=(10,10))
for image_batch,label_batch in dataset.take(1):
    # print(image_batch[0].shape)
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")
    #every time you run it is shuffling the images and labels

#train test split
# we use 80% of the data for training, 10% for evaluation, and 10% for testing
train_size = 0.8
train_data_size = int(train_size*len(dataset))
print(f"Train Size : {train_data_size}")

train_dataset = dataset.take(train_data_size) #80% of the data
print(len(train_dataset))
#----------------
test_dataset = dataset.skip(train_data_size) #10% of the data
len(test_dataset)

val_size = 0.1
real_val_size = int(len(dataset)*val_size)
print(real_val_size)

val_dataset = test_dataset.take(real_val_size) #5% of the data
len(val_dataset)

#-----------------
test_dataset = test_dataset.skip(real_val_size)
len(test_dataset)

#---------------
def get_dataset_partition_tf(ds, train_split = 0.8, val_split = 0.1 , test_split = 0.1, shuffle = True, shuffle_size =10000):
    dataset_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed = 12)

    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    # test_size = int(test_split * dataset_size)
    train_dataset = ds.skip(train_size)
    val_dataset = ds.skip(train_size).take(val_size)
    test_dataset = ds.skip(train_size).skip(val_size)

    return train_dataset , val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = get_dataset_partition_tf(dataset)

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))


train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

#Data Augmentation (Scaling)
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.2)
])

#CNN MODEL
input_shape = (BATCH_SIZE,IMAGE_SIZE, IMAGE_SIZE, CHANNEL)
n_classes = 3
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3,3), padding = "same", activation = "relu", input_shape = input_shape),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), padding = "same", activation = "relu"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), padding = "same", activation = "relu"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), padding = "same", activation = "relu"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), padding = "same", activation = "relu"),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation = "relu"),
    layers.Dense(38, activation = "softmax")
])
model.build(input_shape = input_shape)

model.summary()

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

history = model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = EPOCH,
    batch_size = BATCH_SIZE,
    verbose = 1,
    )

score = model.evaluate(test_dataset)
score

print(history)
print(history.params)
print(history.history.keys())
print(history.history["accuracy"])

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCH), acc, label = "Training Accuracy")
plt.plot(range(EPOCH), val_acc, label = "Validation Accuracy")
plt.legend(loc = "lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1,2,2)
plt.plot(range(EPOCH), loss, label = "Training Loss")
plt.plot(range(EPOCH), val_loss, label = "Validation Loss")
plt.legend(loc = "upper right")
plt.title("Training and Validation Loss")
plt.show()

#make predictions and test dataset
import numpy as np

for image_batch,labels_batch in test_dataset.take(1):
    first_image = image_batch[0].numpy().astype("uint8")
    first_label = labels_batch[0].numpy()

    print("First Image to presict")
    plt.imshow(first_image)
    print("Actual Label: ", class_names[first_label])

    batch_prediction = model.predict(image_batch)
    print("Predicted Label: ", class_names[np.argmax(batch_prediction[0])])

def predict(model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])),2)
    return predicted_class, confidence

plt.figure(figsize=(12,12))
for  images,labels in test_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(model, first_image)
        actual_class = class_names[labels[i]]
        plt.title(f"Actual: {actual_class},\n Predicted {predicted_class} \n Confidence: {confidence}%")
        plt.axis("off")