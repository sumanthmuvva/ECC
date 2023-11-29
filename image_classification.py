import ray
import tensorflow as tf
import numpy as np
import boto3
import time
import psutil
from ray.util.sgd.tf import TFTrainer
from datetime import datetime, timedelta

def estimate_aws_cost(hours, instance_type="c6a.large", pricing_model="On Demand", eks_cost_per_hour=0.10, efs_cost_per_gb_month=0.30):
    # Pricing information
    pricing = {
        "c6a.large": {"On Demand": 0.0765, "Spot": 0.0333},
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
    efs_cost = pricing["EFS"]["per_gb_month"] * efs_metered_size * hours # Assuming EFS usage similar to EBS

    total_cost = sum([instance_cost, ebs_cost, ecr_cost, cloudwatch_cost, eks_cost, efs_cost])
    return total_cost


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

def send_metric_to_cloudwatch(metric_name, value, namespace="MLTrainingMetrics"):
    cloudwatch = boto3.client('cloudwatch')
    cloudwatch.put_metric_data(
        Namespace=namespace,
        MetricData=[{'MetricName': metric_name, 'Value': value}]
    )

def data_creator(config):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, validation_images = train_images / 255.0, test_images / 255.0
    train_images = train_images.astype(np.float16)
    validation_images = validation_images.astype(np.float16)
    global len_train_images 
    len_train_images = len(train_images)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(256)
    val_dataset = tf.data.Dataset.from_tensor_slices((validation_images, test_labels)).batch(256)
    return train_dataset, val_dataset

def create_model(config):
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

def train_example():
    ray.init(ignore_reinit_error=True)
    training_start_time = time.time()

    trainer = TFTrainer(
        model_creator=create_model,
        data_creator=data_creator,
        num_replicas=int(ray.available_resources().get("CPU", 1)),
        use_gpu=False,
        verbose=True
    )

    for i in range(200): 
        train_stats = trainer.train()
        val_stats = trainer.validate()
        print(f"[Epoch {i}] Train stats: {train_stats}, Validation stats: {val_stats}")

    trainer.shutdown()
    training_end_time = time.time()

    # Calculating the metrics
    training_time = training_end_time - training_start_time
    latency = training_time / len_train_images
    throughput = 1 / latency
    memory_usage = psutil.virtual_memory().used / (1024 ** 3)

    aws_cost_on_demand = estimate_aws_cost((training_end_time - training_start_time) / 3600)
    aws_cost_spot = estimate_aws_cost((training_end_time - training_start_time) / 3600, pricing_model="Spot")

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

if __name__ == "__main__":
    start_time = time.time()
    train_example()
