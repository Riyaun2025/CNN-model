from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import pickle, time

# Set your correct path here:
TrainingImagePath = r'C:\Users\Admin\Desktop\CNN face recognition\My_datasets'

# Data augmentation for training
train_datagen = ImageDataGenerator(
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # Optional: if you want a validation set
)

#  Training data generator
training_set = train_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Only if validation_split is used
)

#  Validation/Test data generator (optional)
test_set = train_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Only if validation_split is used
)

#  Check label mapping
print(test_set.class_indices)

# Get class indices assigned by flow_from_directory
TrainClasses = training_set.class_indices

# Map numeric labels to class names (folder names)
ResultMap = {}
for faceValue, faceName in zip(TrainClasses.values(), TrainClasses.keys()):
    ResultMap[faceValue] = faceName

# Save mapping for later use
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)

# Print the mapping
print("Mapping of Face and its ID:", ResultMap)

# Output layer will have neurons = number of classes
OutputNeurons = len(ResultMap)
print('\nThe Number of output neurons:', OutputNeurons)

# Define CNN model
classifier = Sequential()

classifier.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(OutputNeurons, activation='softmax'))

# Compile model
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
StartTime = time.time()

# Dynamically calculate steps per epoch and validation steps based on dataset size
steps_per_epoch = training_set.samples // training_set.batch_size
validation_steps = test_set.samples // test_set.batch_size

classifier.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=60,
    validation_data=test_set,
    validation_steps=validation_steps
)

EndTime = time.time()
print("###### Total Time Taken: ", round((EndTime - StartTime) / 60), 'Minutes ######')
