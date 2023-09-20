# This is just the tutorial from Colab with notes

import tensorflow as tf
print("Tensorflow version:", tf.__version__)

# This is a database of handwritten digits that came from the tf website
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# The above command initializes x and y train and test from the mnist database
# Since the values are between 0 and 255, we scale between 0 and 1 by dividing the values by 255
x_train, x_test = x_train / 255.0, x_test / 255.0

# Here's the part where we build an actual machine learning model
model = tf.keras.models.Sequential([
    # It looks like the sequential model has a list of layers and a name for the model as possible arguemts
    # It's very useful for stacking layers where each layer has one input tensor and one output tensor
        # Tensor is the core framework, and all of the computations involve tensors
        # tensors are vectors / matrices of n dimensions
        # They can be from input data, or the output
    # Layers are functions with a known math structure that can be reused, and have trainable variables
    # tensorflow models are made out of layers
    # the flatten, dense, and dropout functions below are layers
  tf.keras.layers.Flatten(input_shape=(28, 28)),
    # flattens the input, but more importantly gives the input shape for the dense layer below
  tf.keras.layers.Dense(128, activation='relu'),
    #  Now the model will take as input arrays of shape (28, 28) and output arrays of shape (None, 128)?
  tf.keras.layers.Dropout(0.2),
    # Initializes BaseRandomLayer. Googled it, have no idea what this does
  tf.keras.layers.Dense(10)
    # Now we change the output array shape again
])

predictions = model(x_train[:1]).numpy()
# For each example, the model returns a vector of logits or log-odds scores, one for each class
# A logit is the vector of raw, non normalized predictions that a classification model generates, which is ordinarily passed to normalize function
# Predictions returns ten logits because it is 10 handwritten numbers that it's being trained on 

# print(predictions)

# This function converts the logits into probabilities for each class
# Also .numpy() converts a tensor object into an numpuy.ndarray object, which means that the converted tensor will be now processed on the cpu
# This vector gives that when it sees a number, it thinks that theres a 10% chance of it being any number
prob_predictions = tf.nn.softmax(predictions).numpy()

# print(prob_predictions)

# this defines a los function for training

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# takes a vector of ground truth values and vector of logits and returns a scalar loss for each example
# equal to neg log prob of true class; loss = 0 if model is sure of class
# Untrained model gives 1/10 for each class, so -tf.math.log(1/10) ~= 2.3


print(loss_fn(y_train[:1], predictions).numpy())


# Before training, we configure and compile the model using Keras Model.compile, set optimizer to adam, set loss to fn above, and specify metric to be evaluated
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# TRAINING THE MODEL

# We use the model.fit method to adjust the model parameters and minimize the loss
# loss closer to 0 = better
model.fit(x_train, y_train, epochs=5)

# The Model.evaluate method checks the model's performance, usually on a validation set or test set.
# This is the test set that we imported along with the dataset
model.evaluate(x_test,  y_test, verbose=2)
# Now this model is trained to 98% accuracy on this dataset!