import numpy as np
from keras import Input, Model
from keras.src.layers import BatchNormalization, Reshape, GRU
from keras.src.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from Sub_Functions.Load_data import Load_data, Load_data2, train_test_split3
from Sub_Functions.Load_data import train_test_split2
from collections import Counter
# Train-test split
X_train, X_test, y_train, y_test = train_test_split3("Drowsy_non_Drowsy",25)

# Normalize data using ImageDataGenerator for train and test
input_shape=(227, 227, 3)
# Define model
inputs = Input(shape=input_shape)

x = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
x = MaxPooling2D(2, 2)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D(2, 2)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D(2, 2)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Flatten()(x)
x = Reshape((1, -1))(x)
x = GRU(64, return_sequences=False)(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Early stopping
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

# Train the model
model.fit( X_train, y_train_cat,epochs=30, batch_size=32,validation_split=0.2,verbose=1)

# Evaluate the model
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

loss,accuracy1=model.evaluate(X_test,y_test_cat)

cm=confusion_matrix(y_true,y_pred_classes)
print("Confusion_matrix:\n",cm)
print(loss)
print(accuracy1)
model.save("proposed_model_3.h5")