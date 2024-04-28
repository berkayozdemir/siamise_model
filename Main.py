import args
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from keras import applications
from keras import layers
from keras import losses
from keras import ops
from keras import optimizers
from keras import metrics
from keras import Model
from keras.applications import resnet
from keras import layers, models, optimizers, metrics
import keras.saving
import sys

target_shape = (200, 200)

anchor_images_path = sys.argv[1]
positive_images_path = sys.argv[2]

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])


# We need to make sure both the anchor and positive images are loaded in
# sorted order so we can match them together.
anchor_images = [os.path.join(anchor_images_path, f) for f in os.listdir(anchor_images_path) if os.path.isfile(os.path.join(anchor_images_path, f))]

positive_images = [os.path.join(positive_images_path, f) for f in os.listdir(positive_images_path) if os.path.isfile(os.path.join(positive_images_path, f))]



image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

# To generate the list of negative images, let's randomize the list of
# available images and concatenate them together.
rng = np.random.RandomState(seed=42)
rng.shuffle(anchor_images)
rng.shuffle(positive_images)

negative_images = anchor_images + positive_images
np.random.RandomState(seed=32).shuffle(negative_images)

negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
negative_dataset = negative_dataset.shuffle(buffer_size=4096)

dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

# Let's now split our dataset in train and validation.
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)





visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])
plt.show()

base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = ops.sum(tf.square(anchor - positive), -1)
        an_distance = ops.sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(resnet.preprocess_input(anchor_input)),
    embedding(resnet.preprocess_input(positive_input)),
    embedding(resnet.preprocess_input(negative_input)),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)



##siamise model
@keras.saving.register_keras_serializable()
class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

    @classmethod
    def from_config(cls, config, **kwargs):
        # Retrieve necessary parameters from config
        siamese_network = config.pop('siamese_network')
        margin = config.pop('margin')

        # Create SiameseModel instance with retrieved parameters
        model = cls(siamese_network=siamese_network, margin=margin)

        # Add any additional configurations back to the model if needed
        return model

    def get_config(self):
        config = super().get_config()
        config.update({
            'siamese_network': self.siamese_network,
            'margin': self.margin
        })
        return config

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset)


sample = next(iter(train_dataset))
visualize(*sample)
siamese_model.save("my_model2.keras")
anchor, positive, negative = sample
anchor_embedding, positive_embedding, negative_embedding = (
    embedding(resnet.preprocess_input(anchor)),
    embedding(resnet.preprocess_input(positive)),
    embedding(resnet.preprocess_input(negative)),
)

cosine_similarity = metrics.CosineSimilarity()

positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
print("Positive similarity:", positive_similarity.numpy())

negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
print("Negative similarity", negative_similarity.numpy())


##predict functions
def predict_similarity(siamese_model, x_test_left, x_test_right, threshold=0.5):
    # Normalize pixel values to range [0, 1]
    x_test_left = tf.cast(x_test_left, 'float32') / 255.0
    x_test_right = tf.cast(x_test_right, 'float32') / 255.0

    # Make predictions on test data
    distances = siamese_model.predict([x_test_left, x_test_right, x_test_right])

    # Compute absolute differences between distances
    diff = tf.abs(distances[0] - distances[1])

    # Classify pairs based on distance
    predictions = diff <= threshold
    return predictions


def save_predicted_images(predictions, x_test_left, x_test_right, save_dir, j):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Iterate through the predictions and save similar pairs
    for i, pred in enumerate(predictions):
        if pred:
            left_image_np = x_test_left[i].numpy()
            right_image_np = x_test_right[i].numpy()
            # Save similar pairs
            left_image_path = os.path.join(save_dir, f"left_image_"+ str(j)+"_{i}.png")
            right_image_path = os.path.join(save_dir, f"right_image_"+ str(j)+"_{i}.png")

            plt.imsave(left_image_path, left_image_np[i])
            plt.imsave(right_image_path, right_image_np[i])





# Example usage of predict_similarity function
# Assuming you have loaded or generated your test dataset
# x_test_left and x_test_right should contain the left and right images respectively
# Each pair of images should have the same index in both arrays
sample = next(iter(val_dataset))
x_test_left, x_test_right, _ = sample  # Assuming sample is a triplet (anchor, positive, negative)



# Assuming siamese_network is the trained Siamese network model
predictions = predict_similarity(siamese_network, x_test_left, x_test_right)

for i in range(32):
    sample = next(iter(val_dataset))
    x_test_left1, x_test_right1, _ = sample
    predictions = predict_similarity(siamese_network, x_test_left1, x_test_right1)
    save_predicted_images(predictions, x_test_left1, x_test_right1, save_dir="predicted_images", j=i)
# Print predictions
for i, pred in enumerate(predictions):
    if pred:
        print(f"Images {i} and {i} are similar")
    else:
        print(f"Images {i} and {i} are dissimilar")