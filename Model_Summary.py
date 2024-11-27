import tensorflow as tf

# Load the model from the file
model = tf.keras.models.load_model('face.h5')

# Print the model summary
model.summary()
