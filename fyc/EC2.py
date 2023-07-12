from flask import Flask, request, jsonify
import subprocess
import boto3
import json
import threading
import requests
import queue


request_queue = queue.Queue()   # request queue
response_queue = queue.Queue()  # response queue


app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_post_request():
    print("content_type:", request.headers.get("content_type"))
    print("data:", request.data)
    print("form:", request.form)
    print("files:", request.files)
    request_data = request.get_json()
    print(request_data)
    request_queue.put(request_data)  # Put the request data into the request queue

    print("waiting")
    # Wait to get the response data
    response_data = response_queue.get()
    response_queue.task_done()
    print(response_data)

    return jsonify(response_data)


def restart(data):
    # Get instance URL and other necessary data from 'data' dictionary
    instance_url = get_instance_url()
    file_address = data['save_file']
    index = data['index']

    # Prepare payload for Lambda invocation
    payload = {
        'index' : index,
        'file_address' : file_address,
        'instance_url' : instance_url,
        'instance_post' : 5000,
        'if_restart' : True
    }

    # Invoke Lambda function asynchronously
    lambda_client = boto3.client('lambda')
    lambda_client.invoke(FunctionName=payload['worker_func'],
                            InvocationType='Event',
                            Payload=json.dumps(payload))
    
    response = {
        'message' : 'CLOSE'
    }
    
    return response


def step_average(data):
    # Get step data from 'data' dictionary
    step = data.get('step')

    # Calculate average of the step data
    average = sum(step) / len(step)
    response = {
        'average' : average
    }
    print(response,"ave")
    return response


def get_instance_url():
    metadata_url = "http://169.254.169.254/latest/meta-data/public-ipv4"
    response = requests.get(metadata_url)
    if response.status_code == 200:
        instance_ip = response.text
        instance_url = f"http://{instance_ip}"
        return instance_url
    else:
        return None
    

def process_requests():
     while True:
        data = request_queue.get()  # Get the request data from the request queue
        # Process the request and generate the response data
        request_type = data.get('request')
        print(request_type)

        if request_type == 'restart':
            response = restart(data)
    
        if request_type == 'average_step':
            response = step_average(data)

        # Put the response data into the response queue
        response_queue.put(response)
        request_queue.task_done()  # Mark the request as processed


if __name__ == '__main__':
    # Create and start the thread for processing requests
    request_thread = threading.Thread(target=process_requests)
    request_thread.start()

    app.run(host='0.0.0.0', port=5000)
