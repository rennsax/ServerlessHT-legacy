from flask import Flask, request, jsonify
import subprocess
import boto3
import json
import threading
import requests
import queue

step_request_queue = queue.Queue()           # 请求队列
step_response_queue = queue.Queue()          # 响应队列
restart_request_queue = queue.Queue()        # 重启请求队列
restart_response_queue = queue.Queue()       # 重启响应队列

overall_average_events = {}                  # Event objects for overall average calculation
groups_number = []                         

app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_post_request():
    request_data = request.get_json()
    if request_data.get('request') == 'average_step':
        group_id = request_data.get('group_id')
        step_request_queue.put(request_data)  

        print(group_id)

        # Wait for overall average calculation to complete
        overall_average_events[group_id].wait()

        # 获取响应数据
        response_data = step_response_queue.get()

    elif request_data.get('request') == 'restart':
        restart_request_queue.put(request_data)  # Put the restart request data into the restart request queue
        response_data = restart_response_queue.get()  # Wait to get the response data

    print(response_data)

    return jsonify(response_data)


def restart_worker():
    while True:
        request_data = restart_request_queue.get()  # Get the restart request data from the restart request queue
        # Process the restart request and generate the response data
        response = restart(request_data)
        restart_response_queue.put(response)        # Put the response data into the restart response queue


def step_worker(group_id):
    overall_average_event = overall_average_events[group_id]
    response_queue = queue.Queue()  # 创建一个临时的响应队列用于存储当前步骤的响应

    while True:
        request_data = step_request_queue.get()     # Get the step request data from the step request queue

        print("receive one")

        response_queue.put(request_data)             # 将请求数据放入响应队列

        # Check if all Lambda requests have arrived for the group
        if step_request_queue.qsize() == groups_number[group_id]: 
            print(group_id,"done")
            step_average_all = calculate_overall_average(response_queue, group_id)

            response_data = {
                'average_step' : step_average_all,
            }
            
            for _ in range(groups_number[group_id]):
                step_response_queue.put(response_data)
                # response_queue.get()

            overall_average_event.set()

    

def restart(data):
    # Get instance URL and other necessary data from 'data' dictionary
    instance_url = get_instance_url()

    file_name = data.get('file_name')
    request = data.get('request')
    bucket_name = data.get('bucket_name')
    params_file_name = data.get('params_file_name')
    group_id = data.get('group_id')
    epochs_done = data.get('epochs_done')
    ip_address = data.get('ip_address')
    port = data.get('port')

    print(file_name)

    # Prepare payload for Lambda invocation
    payload = {
        'if_restart' : "True",
        'ip_address' : ip_address,
        'port' : port,
        'bucket_name' : bucket_name,
        'params_file_name' : params_file_name,
        'group_id' : group_id,
        'epochs_done' : epochs_done
    }

    # Invoke Lambda function asynchronously
    lambda_client = boto3.client('lambda', region_name='us-west-2')
    lambda_client.invoke(FunctionName=file_name,
                         InvocationType='Event',
                         Payload=json.dumps(payload))

    print(instance_url)

    response = {
        'message': 'CLOSE'
    }

    return response


def calculate_overall_average(response_queue, group_id):
    averages = []

    while not response_queue.empty():
        response = response_queue.get()
        step = response.get('step')
        averages.append(step)

    # 转置操作
    transposed_arrays = zip(*averages)

    # 计算平均值
    overall_average = [sum(column) / len(column) for column in transposed_arrays]

    return overall_average


def get_instance_url():
    metadata_url = "http://169.254.169.254/latest/meta-data/public-ipv4"
    response = requests.get(metadata_url)
    if response.status_code == 200:
        instance_ip = response.text
        instance_url = f"http://{instance_ip}"
        return instance_url
    else:
        return None

if __name__ == '__main__':
    # Create and start the thread for processing restart requests
    restart_thread = threading.Thread(target=restart_worker)
    restart_thread.start()

    total_groups = 2 # set the number of training groups 

    # Create and start the thread for processing step requests for each group
    for group_id in range(total_groups):
        groups_number.append(2)
        overall_average_events[group_id] = threading.Event()
        step_thread = threading.Thread(target=step_worker, args=(group_id,))
        step_thread.start()

    app.run(host='0.0.0.0', port=5000)
