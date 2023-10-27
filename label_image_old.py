
import argparse
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
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

if __name__ == "__main__":
    print("[DEBUG] Starting script execution...")
    file_name = r"C:\Users\muvva\Desktop\ECC\Project\tensorflow\tensorflow\examples\label_image\data\grace_hopper.jpg"
    model_file = r"C:\Users\muvva\Desktop\ECC\Project\tensorflow\tensorflow\examples\label_image\data\inception_v3_2016_08_28_frozen.pb"
    label_file = r"C:\Users\muvva\Desktop\ECC\Project\tensorflow\tensorflow\examples\label_image\data\imagenet_slim_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "input"
    output_layer = "InceptionV3/Predictions/Reshape_1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()

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
        results = sess.run(output_operation.outputs[0], { input_operation.outputs[0]: t })
    results = np.squeeze(results)
    print("[DEBUG] Session run successfully! Processing results...")
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        print(labels[i], results[i])
    print("[DEBUG] Script execution completed!")
