# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import numpy as np
import tensorflow as tf
import time
import boto3
import json

tf.compat.v1.disable_eager_execution()

# Global cache dictionary
PRICE_CACHE = {}

def load_graph(model_file):
    print("[DEBUG] Loading graph from:", model_file)
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    print("[DEBUG] Graph loaded successfully!")
    return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
    print("[DEBUG] Reading tensor from image file:", file_name)
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.io.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.io.decode_png(file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.io.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.io.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.io.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    tensor = sess.run(normalized)
    print("[DEBUG] Tensor successfully read from image!")
    return tensor

def load_labels(label_file):
    print("[DEBUG] Loading labels from:", label_file)
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    labels = [l.rstrip() for l in proto_as_ascii_lines]
    print("[DEBUG] Labels loaded successfully!")
    return labels

def estimate_aws_cost(execution_time, INSTANCE_TYPE='t2.micro', REGION='us-east-1', EBS_TYPE='gp3', EBS_SIZE_GB=30, OUTBOUND_DATA_GB=1):
    def get_aws_price(service_code, region, attribute_filter):
        cache_key = json.dumps([service_code, region, attribute_filter], sort_keys=True)
        if cache_key in PRICE_CACHE:
            return PRICE_CACHE[cache_key]
        client = boto3.client('pricing', region_name='us-east-1')
        try:
            response = client.get_products(
                ServiceCode=service_code,
                Filters=[
                    {
                        'Type': 'TERM_MATCH',
                        'Field': 'location',
                        'Value': region
                    },
                    attribute_filter
                ],
                MaxResults=10
            )
        except Exception as e:
            print(f"Error fetching price: {e}")
            return None

        for product in response['PriceList']:
            product_data = json.loads(product)
            terms = product_data['terms']['OnDemand']
            term_key = list(terms.keys())[0]
            price_dimensions = terms[term_key]['priceDimensions']
            price_dimension_key = list(price_dimensions.keys())[0]
            price = float(price_dimensions[price_dimension_key]['pricePerUnit']['USD'])
            # Cache the result for 24 hours (86400 seconds)
            PRICE_CACHE[cache_key] = price
            print(f"Price: ${price}")
            return price
        # If no price found, return None and cache this result for 10 minutes to avoid frequent checks
        PRICE_CACHE[cache_key] = None
        time.sleep(10)  # Sleep for 10 minutes to prevent frequent retries
        return None

    def get_ec2_price(instance_type, region='US East (N. Virginia)'):
        attribute_filter = {
            'Type': 'TERM_MATCH',
            'Field': 'instanceType',
            'Value': instance_type
        }
        return get_aws_price('AmazonEC2', region, attribute_filter)

    def get_ebs_price(ebs_type, region='US East (N. Virginia)'):
        attribute_filter = {
            'Type': 'TERM_MATCH',
            'Field': 'volumeType',
            'Value': ebs_type
        }
        return get_aws_price('AmazonEC2', region, attribute_filter)

    def data_transfer_cost(data_gb, region='US East (N. Virginia)'):
        attribute_filter = {
            'Type': 'TERM_MATCH',
            'Field': 'transferType',
            'Value': 'DataTransfer-Out-Bytes'  # This is an example, adjust if needed
        }
        price_per_gb = get_aws_price('AmazonEC2', region, attribute_filter)
        return price_per_gb * data_gb if price_per_gb else 0

    # Convert execution time from seconds to hours
    run_time_hours = execution_time / 3600
    # Estimate Cost
    ec2_cost = get_ec2_price(INSTANCE_TYPE, REGION)
    if ec2_cost is None:
        print("Error fetching EC2 price. Setting Manual Value.")
        ec2_cost = 0.0116
    ec2_cost = ec2_cost * run_time_hours
    ebs_cost = get_ebs_price(EBS_TYPE, REGION)
    if ebs_cost is None:
        print("Error fetching EBS price. Setting Manual Value.")
        ebs_cost = 0.004625
    ebs_cost = ebs_cost * run_time_hours
    data_transfer = data_transfer_cost(OUTBOUND_DATA_GB)
    total_cost = ec2_cost + ebs_cost + data_transfer
    return total_cost
'''
def report_metrics_to_cloudwatch(cloudwatch, execution_time, throughput, total_cost):
    cloudwatch.put_metric_data(
        MetricName='Latency',
        Namespace='MyMLModel/Metrics',
        Value=execution_time
    )
    cloudwatch.put_metric_data(
        MetricName='Throughput',
        Namespace='MyMLModel/Metrics',
        Value=throughput
    )
    cloudwatch.put_metric_data(
        MetricName='TrainingTime',
        Namespace='MyMLModel/Metrics',
        Value=execution_time
    )
    cloudwatch.put_metric_data(
        MetricName='EstimatedCost',
        Namespace='MyMLModel/Metrics',
        Value=total_cost
    )
'''
def report_metrics_to_cloudwatch(cloudwatch, execution_time, throughput, total_cost):
    cloudwatch.put_metric_data(
        Namespace='MyMLModel/Metrics', # Replace with your desired namespace
        MetricData=[
            {
                'MetricName': 'ModelExecutionTime',
                'Value': execution_time
            },
            {
                'MetricName': 'ModelThroughput',
                'Value': throughput
            },
            {
                'MetricName': 'TotalCost',
                'Value': total_cost
            }
        ]
    )


if __name__ == "__main__":
    print("[DEBUG] Starting script execution...")
    file_name = "/app/data/grace_hopper.jpg"
    model_file = "/app/data/inception_v3_2016_08_28_frozen.pb"
    label_file = "/app/data/imagenet_slim_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "input"
    output_layer = "InceptionV3/Predictions/Reshape_1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="/app/data/grace_hopper.jpg", help="image to be processed")
    parser.add_argument("--graph", default="/app/data/inception_v3_2016_08_28_frozen.pb", help="graph/model to be executed")
    parser.add_argument("--labels", default="/app/data/imagenet_slim_labels.txt", help="name of file containing labels")
    parser.add_argument("--input_height", default=299, type=int, help="input height")
    parser.add_argument("--input_width", default=299, type=int, help="input width")
    parser.add_argument("--input_mean", default=0, type=int, help="input mean")
    parser.add_argument("--input_std", default=255, type=int, help="input std")
    parser.add_argument("--input_layer", default="input", help="name of input layer")
    parser.add_argument("--output_layer", default="InceptionV3/Predictions/Reshape_1", help="name of output layer")

    args = parser.parse_args()
    cloudwatch = boto3.client('cloudwatch')

    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(file_name, input_height=input_height, input_width=input_width, input_mean=input_mean, input_std=input_std)

    print("[DEBUG] Preparing to run the session...")
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.compat.v1.Session(graph=graph) as sess:
        print("[DEBUG] Running the session...")
        # Start timing
        start_time = time.time()
        results = sess.run(output_operation.outputs[0], { input_operation.outputs[0]: t })
        # end timing
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[DEBUG] Model Execution Time/Latency: {execution_time} seconds")
        # Calculate throughput based on this single prediction
        throughput = 1 / execution_time
        print(f"[DEBUG] Model Throughput: {throughput} predictions/second")
        # Start of our cost estimation logic
        # Constants (customize these based on your setup)
        total_cost = estimate_aws_cost(execution_time)
        report_metrics_to_cloudwatch(cloudwatch, execution_time, throughput, total_cost)
        print(f"[DEBUG] Estimated Cost: ${total_cost}")
    results = np.squeeze(results)
    print("[DEBUG] Session run successfully! Processing results...")
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        print(labels[i], results[i])
    print("[DEBUG] Script execution completed!")