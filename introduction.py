import tensorflow as tf  # Import TensorFlow module
mnist = tf.keras.datasets.mnist  # Initialize datasets sources

(x_train, y_train),(x_test, y_test) = mnist.load_data()  # Loading datasets
x_train, x_test = x_train / 255.0, x_test / 255.0  # Setting limits to training sets

model = tf.keras.models.Sequential([  # Declaring a Sequential TensorFlow Keras model
  tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatness of the input image
  tf.keras.layers.Dense(512, activation=tf.nn.relu),  # Density of the input image
  tf.keras.layers.Dropout(0.2),  # Dropout of the input image
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # Density of the input image
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # Compiling model

model.fit(x_train, y_train, epochs=5)  # Training set up for model
model.evaluate(x_test, y_test)  # Train model
