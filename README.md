import tensorflow as tf 
import matplotlib. pyplot as plt
import tensorflow.keras.preprocessing.image as tkp
import os
import re

def sanitize_filename(filename):
    # Use regular expressions to remove non-UTF-8 characters from filenames
    return re.sub(r'[^\x00-\x7F]+', '_', filename)  # Replaces non-UTF-8 characters with "_"

def rename_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for name in files:
            new_name = sanitize_filename(name)
            if new_name != name:
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed file {old_path} to {new_path}")

# Specify the base directories
train_dir = "D:\\Soybean project\\disease\\Train ds"
test_dir = "D:\\Soybean project\\disease\\Test ds"


# Run the renaming function on each directory
rename_files_in_directory(train_dir)
rename_files_in_directory(test_dir)
img_height, img_width = 32, 32
batch_size = 32
try:
    Train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    print("Dataset loaded successfully.")
except UnicodeDecodeError as e:
    print("UnicodeDecodeError encountered:")
    print(e)
    try:
    Test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    print("Dataset loaded successfully.")
except UnicodeDecodeError as e:
    print("UnicodeDecodeError encountered:")
    print(e)
    class_names = ["bacterial_blight","brown_spot","cercospora_leaf_blight", "ferrugen","powdery_mildew","southern blight","soybean_rust","sudden Death Syndrom","Yellow mosaic",]
plt.figure(figsize=(15,15))
for images, labels in Train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    model = tf.keras.Sequential(
    [
     tf.keras.layers.Rescaling(1./255),
     tf.keras.layers.Conv2D(64, 2, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(128, 2, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(256, 2, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(256, activation="relu"),
     tf.keras.layers.Dense(9)
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)
model.fit(
    Train_ds,
    epochs = 30
)
model.evaluate(Test_ds)
import numpy

plt.figure(figsize=(20,20))
for images, labels in Train_ds.take(1):
  classifications = model(images)
  # print(classifications)

  for i in range(8):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    index = numpy.argmax(classifications[i])
    plt.title("Pred: " + class_names[index] + " | Real: " + class_names[labels[i]])
    
