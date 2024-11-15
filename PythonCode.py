import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

IMG_WIDTH, IMG_HEIGHT = 200, 200

TRAINING_DIR = r"/Users/rj/Documents/462/train"
TEST_DIR = r"/Users/rj/Documents/462/test"

train_data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=40,
    zoom_range=0.2,
    validation_split=0.1  
)

test_data_gen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_data_gen.flow_from_directory(
    TRAINING_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    class_mode='categorical',
    batch_size=32,
    subset='training'  
)

validation_generator = train_data_gen.flow_from_directory(
    TRAINING_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    class_mode='categorical',
    batch_size=32,
    subset='validation'  
)

test_generator = test_data_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    class_mode='categorical',
    batch_size=32,
    shuffle=False  
)

model_base = InceptionV3(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    include_top=False,
    weights='imagenet'  
)

for layer in model_base.layers[:-50]:  
    layer.trainable = False

last_output = model_base.get_layer('mixed10').output


x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(2056, activation='relu', kernel_regularizer='l2')(x)
x = tf.keras.layers.Dropout(0.3)(x)  # Dropout layer to prevent overfitting
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer='l2')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(9, activation='softmax')(x)


model = tf.keras.Model(inputs=model_base.input, outputs=x)


model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()


history = model.fit(
    train_generator,
    epochs= 50,  
    validation_data=validation_generator
)



model.save('actualModel.keras')

import pickle

with open('modelHistory.binary', 'wb') as file:
    pickle.dump(history.history, file)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")