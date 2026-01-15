import tensorflow as tf
from keras import layers, models
from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense
import numpy as np
import time
from server import *



#######################################################################################################
############################# Loading Dataset and assigning it to clients #############################

# Loading Dataset: (CIFAR-10)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#============================ Different Datasets to experiment with =======================
# Dataset (FMNIST):
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Dataset (MNIST):
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#==========================================================================================

x_train, x_test = x_train.astype("float32"), x_test.astype("float32")
y_train, y_test = y_train.flatten(), y_test.flatten()


# Split Data among clients

num_clients = 5 # We will do collaborative training using 5 clients
data_per_client = len(x_train) // num_clients

# Cifar-10 has 50k training data; so each client gets 10k training data.
# MNIST and FMNIST datasets have 60k training data. so for 5 clients, each will be assigned 12k training data.
print("Each client has ", data_per_client, "data points")

client_data = [(x_train[i * data_per_client:(i + 1) * data_per_client],
                y_train[i * data_per_client:(i + 1) * data_per_client])
               for i in range(num_clients)]


input_shape = (32, 32, 3) # for Cifar-10 dataset.
#input_shape = (28,28,1) # for mnist and fmnist datasets

num_classes = 10 # All 3 datasets contains 10 image classes/labels
batch_size = 64 # Experiment with different batch size so observe its effect on training results.
epochs = 20

###########################################################################################################
##################################### Neural Network Model for Clients  ###################################

# LeNet (a simple Neural Network Architecture)

def build_model(input_shape=input_shape, num_classes=10):
    with tf.device('/cpu:0'):
    # with tf.device('/gpu:0'):      #Comment previous line and uncomment this one if you want to use GPU (training will be faster)

        inputs = Input(shape=input_shape)
        # C1: 6 filters, 5x5, tanh
        x = Conv2D(6, (5, 5), activation='tanh', padding='valid', name='C1')(inputs)
        # S2: Average pooling 2x2, stride 2
        x = AveragePooling2D(pool_size=(2, 2), name='S2')(x)
        # C3: 16 filters, 5x5, tanh
        x = Conv2D(16, (5, 5), activation='tanh', padding='valid', name='C3')(x)
        # S4: Average pooling 2x2, stride 2
        x = AveragePooling2D(pool_size=(2, 2), name='S4')(x)
        # Flatten
        x = Flatten(name='Flatten')(x)
        # F5: Dense 120, tanh
        x = Dense(120, activation='tanh', name='F5')(x)
        # F6: Dense 84, tanh
        x = Dense(84, activation='tanh', name='F6')(x)
        # Output: Softmax for 10 classes
        outputs = Dense(num_classes, activation='softmax', name='Output')(x)

    return Model(inputs, outputs, name='LeNet')



############################################################################################################
############################################ Training ######################################################

# 5 LeNet model for 5 clients
models = []
for _ in range(num_clients):
    model = build_model()  # build the model without splitting
    models.append(model)

# Loss Function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Learning rate scheduler, for better results
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizers = [tf.keras.optimizers.Adam(learning_rate=lr_schedule) for _ in range(num_clients)]




def evaluate_global_model(model):
    batch_size = 512
    num_samples = len(x_test)
    preds = []
    for i in range(0, num_samples, batch_size):
        batch_x = x_test[i:i + batch_size]
        logits = model(batch_x, training=False)
        batch_preds = tf.argmax(logits, axis=1)
        preds.extend(batch_preds.numpy())
    preds = tf.convert_to_tensor(preds)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_test), tf.float32))
    global_accs.append(acc)
    print(f"Global Test Accuracy: {acc.numpy():.4f}")



client_accuracies = [[] for _ in range(num_clients)]

# Training loop
# In each epoch: 5 clients train the model on their assigned datapoints (10k each)
# Aggregates the training updates. In real-life FL, the clients would send the updates to the server.
# The training updates are aggregated to update the global model (on server) which is used as base in next epoch
# Hence, despite each client only training on 10k datapoints, they receive the benefit of a model trained on 50k.

for epoch in range(epochs):
    print(f"\n -> Epoch {epoch + 1}/{epochs}")

    for client_id, (x_client, y_client) in enumerate(client_data):
        start_time = time.time()

        train_ds = tf.data.Dataset.from_tensor_slices((x_client, y_client)).shuffle(1000).batch(batch_size)
        model = models[client_id]

        for step, (x_batch, y_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = loss_fn(y_batch, logits)
                acc = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.cast(y_batch, tf.int64)), tf.float32))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizers[client_id].apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(f"Client {client_id}, Step {step}, Loss: {loss.numpy():.4f}, Accuracy: {acc.numpy():.4f}")

            client_accuracies[client_id].append(acc.numpy())

        end_time = time.time()
        print(f"Time taken: {end_time - start_time}")

    sample_counts = [len(data[0]) for data in client_data]
    avg_weights = average_trainable_weights_weighted(models, sample_counts)

    for model in models:
        set_trainable_weights(model, avg_weights)


    global_models.append(models[0])


    # An evaluation function to show the global model's accuracy. The global model is updated aggregating the local
    # training updates of the clients
    evaluate_global_model(models[0])




# FL is a distributed learning "Paradigm". A way of decentralized learning.
# In this work, you are seeing ~60% (with epoch > 20). That is becuase we are using simple LeNet Model.
# Replace this model with any complex and more capable one, the results will be better.
# The alexnet.py file contains the AlexNet model.
# Replace the build_model() function here with the model provided in alexnet.py file for better accuracy (However, longer training time)





# In this implementation, we are running 5 clients sequentially, since they are ran on the same device.
# However, in real-life FL, each client is an individual edge-device in the network, training parallelly
# So, in this implementation you are seeing each epoch taking 1 min (~12 seconds * 5 clients)
# But in real-FL, the whole epoch will be done in ~12 seconds (it's a metaphoric amount of time).
# The benefits:
    # A client trains on 10k data, gets the benefit of model that's trained on 50k data
    # Clients never shares their training data in the network. Only the training updates are shared (secured)
    # Global model trained on huge amount of data, within significantly reduced completion time
    # A greatly insightful model as its trained on vast and variety of data
    # and many more...