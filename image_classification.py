import os
import json
import time
import psutil
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')


def estimate_aws_cost(hours, instance_type="c6a.large", pricing_model="On Demand"):
    """
    Estimate AWS cost given the hours of usage, instance type, and pricing model.
    """
    pricing = {
        "c6a.large": {
            "On Demand": 0.0765,
            "Spot": 0.0333
        }
    }
    return hours * pricing.get(instance_type, {}).get(pricing_model, 0)


# Start of the main script
start_time = time.time()

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

assert train_images.shape == (50000, 32, 32, 3)
assert test_images.shape == (10000, 32, 32, 3)
assert train_labels.shape == (50000, 1)
assert test_labels.shape == (10000, 1)

# Split the training data into training and validation (taking last 5000 images)
validation_images = train_images[-5000:]
validation_labels = train_labels[-5000:]

train_images = train_images[:-5000]
train_labels = train_labels[:-5000]
print(f"[DEBUG]: Data Loaded")
print(f"[DEBUG]: Data Normalization Started..")
train_images, validation_images = train_images / 255.0, validation_images / 255.0
print(f"[DEBUG]: Data Normalization Completed..")
# Adjust data type for memory optimization
train_images = train_images.astype(np.float16)
validation_images = validation_images.astype(np.float16)
print(f"[DEBUG]: Model Started..")

baseline_cnn = tf.keras.models.Sequential([
                          tf.keras.layers.Conv2D(filters=10,kernel_size=(5,5),input_shape=(32,32,3),strides=(1,1), activation='relu',kernel_initializer='he_normal'),
                          tf.keras.layers.MaxPool2D((2,2),strides=2),
                          tf.keras.layers.Conv2D(10,kernel_size=(5,5),strides=(1,1),activation='relu',kernel_initializer='he_normal'),
                          tf.keras.layers.MaxPool2D((2,2),strides=2),
                          tf.keras.layers.Flatten(),
                          tf.keras.layers.Dense(20,activation='relu',kernel_initializer='he_normal'),
                          tf.keras.layers.Dense(10,activation=None,kernel_initializer='he_normal')])
print(f"[DEBUG]: Model Compilation Started..")
baseline_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
# Start the timer for training time
training_start_time = time.time()

baseline_cnn_his = baseline_cnn.fit(train_images, train_labels, epochs=200, batch_size=256, validation_data=(validation_images, validation_labels))
print(f"[DEBUG]: Model Fitting Completed..")
# End the timer for training time
training_end_time = time.time()

# Extract and print the last epoch's metrics
final_train_acc = baseline_cnn_his.history['sparse_categorical_accuracy'][-1]
final_val_acc = baseline_cnn_his.history['val_sparse_categorical_accuracy'][-1]
final_train_loss = baseline_cnn_his.history['loss'][-1]
final_val_loss = baseline_cnn_his.history['val_loss'][-1]

print(f"Final Training Accuracy: {final_train_acc*100:.2f}%")
print(f"Final Validation Accuracy: {final_val_acc*100:.2f}%")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")

#save weights
file_name = 'baseline.h5'
baseline_cnn.save_weights(file_name)

#save model history
history_dict = baseline_cnn_his.history

# Save it to a JSON file
with open('baseline_p3_history.json', 'w') as file:
    json.dump(history_dict, file)

end_time = time.time()
print(f"[DEBUG]: Process Completed.")
print(f"[DEBUG]: Calculating Metrics..\n")
# Calculating the metrics
training_time = training_end_time - training_start_time
latency = training_time / len(train_images)  # Time taken per image
throughput = 1 / latency  # Images processed per second
memory_usage = psutil.virtual_memory().used / (1024 ** 3)  # In GB

# Estimated AWS cost (assuming you trained for the full duration of execution)
aws_cost_on_demand = estimate_aws_cost((end_time - start_time) / 3600)  # Convert seconds to hours
aws_cost_spot = estimate_aws_cost((end_time - start_time) / 3600, pricing_model="Spot")
print(f"[DEBUG]: Printing Final Metrics..")
print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Latency: {latency:.6f} seconds per image")
print(f"Throughput: {throughput:.2f} images per second")
print(f"Memory Usage: {memory_usage:.2f} GB")
print(f"Estimated AWS Cost (On Demand): ${aws_cost_on_demand:.2f}")
print(f"Estimated AWS Cost Curent Type -> (Spot): ${aws_cost_spot:.2f}")
