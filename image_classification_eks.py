import os
import ray
import tensorflow as tf
import numpy as np
import boto3
import time
import psutil
import pickle
from datetime import datetime, timedelta
#from kubernetes import client, config
from ray.train.tensorflow import TensorflowTrainer
from ray.train import ScalingConfig

# Function to get node count in the cluster
# def get_node_count():
#     config.load_incluster_config()
#     v1 = client.CoreV1Api()
#     nodes = v1.list_node()
#     return len(nodes.items)

# def get_node_count():
#     # Create EKS and EC2 clients
#     aws_region = "us-east-1"
#     eks_client = boto3.client('eks', region_name=aws_region)
#     ec2_client = boto3.client('ec2', region_name=aws_region)

#     # Get the cluster VPC configuration
#     cluster_info = eks_client.describe_cluster(name="NewEKScluster")
#     vpc_config = cluster_info['cluster']['resourcesVpcConfig']

#     # List all EC2 instances in the VPC
#     instances = ec2_client.describe_instances(
#         Filters=[
#             {'Name': 'vpc-id', 'Values': [vpc_config['vpcId']]},
#             {'Name': 'instance-state-name', 'Values': ['running']}
#         ]
#     )

#     # Count the instances
#     count = sum(len(reservation['Instances']) for reservation in instances['Reservations'])
#     if count == 0:
#         return 0
#     else:
#         return count-2


# Function to load CIFAR-10 batch
def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']

# Function to load CIFAR-10 dataset
def load_cifar10_dataset(dataset_path):
    train_data, train_labels, test_data, test_labels = [], [], [], []
    for i in range(1, 6):
        batch_data, batch_labels = load_cifar10_batch(f'{dataset_path}/data_batch_{i}')
        train_data.append(batch_data)
        train_labels.append(batch_labels)
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    test_data, test_labels = load_cifar10_batch(f'{dataset_path}/test_batch')
    train_data = train_data.reshape((len(train_data), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_data = test_data.reshape((len(test_data), 3, 32, 32)).transpose(0, 2, 3, 1)
    return (train_data, train_labels), (test_data, test_labels)

# Function to estimate AWS cost
def estimate_aws_cost(hours, instance_type="c5a.large", pricing_model="On Demand", eks_cost_per_hour=0.10, efs_cost_per_gb_month=0.16):
    # Pricing information
    pricing = {
        "c5a.large": {"On Demand": 0.0765, "Spot": 0.0333},
        "EBS": {"per_gb_month": 0.08 / 30 / 24},
        "ECR": {"per_gb_month": 0.10 / 30 / 24},
        "CloudWatch": {"ingestion": 0.50 / 30 / 24, "storage": 0.03 / 30 / 24},
        "EKS": {"per_hour": 0.10},
        "EFS": {"per_gb_month": 0.16 / 30 / 24}
    }

    # Calculations for each service
    ebs_used_vol = get_used_ebs_volume_in_gb()
    ecr_image_size = get_ecr_image_size('image-classification-dis')
    cloudwatch_logs_size = get_cloudwatch_log_size()
    efs_metered_size = get_efs_metered_size_gb('01d2ae932027ea024')
    ebs_cost = pricing["EBS"]["per_gb_month"] * ebs_used_vol * hours
    ecr_cost = pricing["ECR"]["per_gb_month"] * ecr_image_size * hours
    cloudwatch_cost = (pricing["CloudWatch"]["ingestion"] * cloudwatch_logs_size + pricing["CloudWatch"]["storage"] * cloudwatch_logs_size) * hours
    eks_cost = pricing["EKS"]["per_hour"] * hours
    efs_cost = pricing["EFS"]["per_gb_month"] * efs_metered_size * hours
    instance_cost = 3 * pricing["c5a.large"]["On Demand"] * hours
    total_cost = sum([instance_cost, ebs_cost, ecr_cost, cloudwatch_cost, eks_cost, efs_cost])
    return total_cost

# Additional functions: get_used_ebs_volume_in_gb, get_cloudwatch_log_size, get_ecr_image_size, get_efs_metered_size_gb

def get_used_ebs_volume_in_gb(path='/'):
    stat = os.statvfs(path)
    block_size = stat.f_frsize
    total_blocks = stat.f_blocks
    free_blocks = stat.f_bfree
    used_blocks = total_blocks - free_blocks
    return (used_blocks * block_size) / (1024 ** 3)

def get_cloudwatch_log_size():
    client = boto3.client('logs')
    log_groups = client.describe_log_groups()
    total_size_bytes = sum(group['storedBytes'] for group in log_groups['logGroups'])
    total_size_gb = total_size_bytes / (1024 ** 3)
    return total_size_gb

def get_ecr_image_size(repository_name):
    client = boto3.client('ecr')
    response = client.describe_images(repositoryName=repository_name, maxResults=1)
    image_size_bytes = response['imageDetails'][0]['imageSizeInBytes']
    image_size_gb = image_size_bytes / (1024 ** 3)
    return image_size_gb

def get_efs_metered_size_gb(file_system_id):
    cloudwatch = boto3.client('cloudwatch')
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)  # adjust based on your needs

    try:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/EFS',
            MetricName='MeteredSize',
            Dimensions=[{'Name': 'FileSystemId', 'Value': file_system_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=86400,  # one day
            Statistics=['Average']
        )
        if response['Datapoints']:
            metered_size_bytes = response['Datapoints'][0]['Average']
            efs_metered_size = metered_size_bytes / (1024 ** 3)  # Convert from bytes to GB
            return efs_metered_size
        return None
    except Exception as e:
        print(f"Error fetching efs metered size: {e}")
        return None

# Function to send metrics to CloudWatch
def send_metric_to_cloudwatch(metric_name, value, namespace="MLTrainingMetrics"):
    cloudwatch = boto3.client('cloudwatch')
    cloudwatch.put_metric_data(
        Namespace=namespace,
        MetricData=[{'MetricName': metric_name, 'Value': value}]
    )
    pass

# Function to create TensorFlow model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(10, (5, 5), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(10, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Training function for Ray

def train_cifar(config):
    # Load dataset
    checkpoint_dir=None
    (train_images, train_labels), (test_images, test_labels) = load_cifar10_dataset(config['dataset_path'])

    # Normalize datasets
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Convert to NumPy arrays and ensure data type
    train_images = np.array(train_images, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
    test_images = np.array(test_images, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int32)

    # Create and compile the model
    model = create_model()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    for epoch in range(config['num_epochs']):
        model.fit(train_images, train_labels, epochs=1, batch_size=config['batch_size'])

        # Save checkpoint at specific epochs
        if epoch % 2 == 0:
            if checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
                model.save(checkpoint_path)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    return {'loss': test_loss, 'accuracy': test_accuracy}


# Main function
def main():
    ray.init(ignore_reinit_error=True)

    config = {
        'dataset_path': "/mnt/efs/cifar-10-batches-py",
        'batch_size': 256,
        'num_epochs': 10  # Adjust the number of epochs as needed
    }

    # trainer = TensorflowTrainer(
    #     train_func=train_cifar,
    #     num_workers=get_node_count(),
    #     use_gpu=False,
    #     config=config
    # )


    trainer = TensorflowTrainer(
        train_loop_per_worker=train_cifar,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
        train_loop_config={
            "num_epochs": 10, 
            "batch_size": 256, 
            "dataset_path": "/mnt/efs/cifar-10-batches-py",
            "checkpoint_dir": "/mnt/efs/checkpoints"
        },
    )
    for i in range(10):  # Adjust the number of epochs as needed
        train_stats = trainer.fit()
        print(f"Train stats: {train_stats}")

    trainer.shutdown()
    training_end_time = time.time()

    # Calculating the metrics
    training_time = training_end_time - training_start_time
    latency = training_time / len_train_images
    throughput = 1 / latency
    memory_usage = psutil.virtual_memory().used / (1024 ** 3)

    aws_cost_on_demand = estimate_aws_cost((training_end_time - training_start_time) / 3600)
    aws_cost_spot = estimate_aws_cost((training_end_time - training_start_time) / 3600, pricing_model="Spot")


    # Printing the metrics
    print(f"Total Execution Time: {training_time:.2f} seconds")
    print(f"Latency: {latency:.6f} seconds per image")
    print(f"Throughput: {throughput:.2f} images per second")
    print(f"Virtual Memory Usage: {memory_usage:.4f} GB")
    print(f"Total EBS Used Volume: {ebs_used_vol:.4f} GB")
    print(f"Total ECR Image Size: {ecr_image_size:.4f} GB")
    if efs_metered_size is not None:
        print(f"EFS Metered Size: {efs_metered_size:.4f} GB")
    else:
        print("Failed to fetch metered size.")
    print(f"Total CloudWatch Logs Size: {cloudwatch_logs_size:.4f} GB")
    print(f"Estimated AWS Cost (On Demand): {aws_cost_on_demand:.4f}")
    print(f"Estimated AWS Cost (Spot): {aws_cost_spot:.4f}")

    # Sending metrics data to CloudWatch
    send_metric_to_cloudwatch("TotalExecutionTime_InSecs.", training_time)
    send_metric_to_cloudwatch("Latency_SecondsPerImage", latency)
    send_metric_to_cloudwatch("Throughput_ImagesPerSecond", throughput)
    send_metric_to_cloudwatch("MemoryUsage_InGB", memory_usage)
    send_metric_to_cloudwatch("EBSTotalUsedVolume_InGB", ebs_used_vol)
    send_metric_to_cloudwatch("ECRImageSize_InGB", ecr_image_size)
    send_metric_to_cloudwatch("EFSMeteredSize_InGB", efs_metered_size)
    send_metric_to_cloudwatch("CloudWatchLogsSize_InGB", cloudwatch_logs_size)
    send_metric_to_cloudwatch("AWSCostOnDemand_InUS$", aws_cost_on_demand)
    send_metric_to_cloudwatch("AWSCostSpot_InUS$", aws_cost_spot)



if __name__ == "__main__":
    start_time = time.time()
    main()
