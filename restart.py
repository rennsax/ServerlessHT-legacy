from flask import Flask, request, jsonify
import subprocess
import boto3
import json
import threading
import requests


def restart(event):

    # 具体的参数需要根据实际的训练来更改，暂时把所有可能的参数都复制过来了
    # Setting of k_reduce
    worker_func = event['worker_func']              # Name of worker function
    train_net = event['train_net']                  # Training model (e.g. 'resnet18' or 'resnet101')
    pattern_k = int(event['pattern_k'])             # Number of aggregators
    num_workers = int(event['num_workers'])         # Number of total workers
    batch_size_wor = int(event['batch_size_wor'])   # Batch size of non-aggregators 
    batch_size_agg = int(event['batch_size_agg'])   # Batch size of aggregators
    epochs = int(event['epochs'])                   # Number of epochs
    batches = int(event['batches'])                 # Number of batches
    l_r = float(event['l_r'])                       # Learning rate
    memory = event['memory']                         # Memory of function
    func_time = int(event['func_time'])              # Maximum run-time of function (seconds)
    seed = event['seed']                             # Random seed   
    tmp_bucket = event['tmp_bucket']                 # Bucket storing temporary files
    merged_bucket = event['merged_bucket']           # Bucket storing merged files
    partition_bucket = event['partition_bucket']     # Bucket storing partitioned dataset
    data_bucket = event['data_bucket']               # Bucket storing dataset
    output_bucket = event['output_bucket']           # Bucket storing log files

    instance_url = get_instance_url() # get EC2 URL

    # Lambda payload
    global payload
    payload = {
        'worker_func': worker_func,
        'train_net': train_net,
        'pattern_k': pattern_k,
        'num_workers': num_workers,
        'batch_size_wor': batch_size_wor,
        'batch_size_agg': batch_size_agg,
        'epochs': epochs,
        'batches': batches,
        'l_r': l_r,
        'memory': memory,
        'seed': seed,
        'invoke_round': 1,
        'func_time': func_time,
        'tmp_bucket': tmp_bucket,
        'merged_bucket': merged_bucket,
        'partition_bucket': partition_bucket,
        'output_bucket': output_bucket,
        'instance_url' : instance_url
    }

    flask_thread = threading.Thread(target=run_flask_app) # Create a thread to run the Flask application
    flask_thread.start()


def get_instance_url():
    metadata_url = "http://169.254.169.254/latest/meta-data/public-ipv4"
    response = requests.get(metadata_url)
    if response.status_code == 200:
        instance_ip = response.text
        instance_url = f"http://{instance_ip}"
        return instance_url
    else:
        return None
    
# Invoke Lambda functions
def fun_restart(index):
    lambda_client = boto3.client('lambda')
    payload['worker_index'] = index
    lambda_client.invoke(FunctionName=payload['worker_func'],
                            InvocationType='Event',
                            Payload=json.dumps(payload))
    

def run_flask_app():
    app.run(host='0.0.0.0', port=8080)

        
app = Flask(__name__)

@app.route('/', methods=['POST'])

# Handle HTTP POST request
def handle_post_request():
    data = request.form
    
    if 'time' in data:
        total_time = float(data['time'])
        restart_index = data['index']
        print(f"Received total time: {total_time}")
        print(f"Function index: {restart_index}")
        payload = {
            'worker_index': restart_index
        }
        fun_restart(payload)
        response = {
            'message': 'CLOSE'
        }
    else:
        response = {
            'message': 'Invalid request'
        }

    return jsonify(response)