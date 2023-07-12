from flask import Flask, request, jsonify
import json
import time
import requests
import torch
import boto3
import threading


def handler(event, context):

    #time_thread = threading.Thread(target=time_watcher, args=(event, context.function_name))
    #time_thread.start()

    # TODO: train model

    ip_address = event['server_address']
    port = event['instance_port']
    server_address = f"http://{ip_address}:{port}"

    data_value = 1.2  # 初始值
    results = []  # 用于存储结果的列表

    for _ in range(3):
        data = [data_value, data_value + 2, data_value + 4, data_value + 6]

        message = {
            'request': 'average_step',
            'step': data,
        }

        response = requests.post(server_address, json=message)
        response_data = json.loads(response.text)
        results.append(response_data)  # 将结果添加到列表中

        data_value += 1  # 每次迭代递增data的值

        time.sleep(2)

    return {
        'statusCode': 200,
        'body': results
    }


def load_model():
    # TODO: load model's parameters
    pass


def time_watcher(event, function_name):

    func_start = time.time()
    time.sleep(3)

    # Calculate the total time taken by the function
    current_time = time.time()
    total_time = current_time - func_start

    function_index = function_name
    file_name = generate_file_name(function_index)

    # Get server address
    server_address = event['server_address']

    data = {
        "time": total_time,
        "index": function_name,
        'request': 'restart',
        'file_name': file_name
    }

    # Send HTTP request
    response = requests.post(server_address, json=data)

    response_data = json.loads(response.text)

    if response_data['message'] == 'CLOSE':
        print("close")
    
    # Save parameters to S3
    # Load the model
    net = load_model() 
    output_bucket = event['output_bucket']

    # Save the model parameters and checkpoint to a file
    PATH = f'/tmp/{file_name}'
    torch.save({
        'model_parameters': net.state_dict()
    }, PATH)
    
    # Upload the file to the specified output bucket in S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(PATH, output_bucket, file_name)

    return 0


def generate_file_name(index):
    # Generate the file name based on the function index
    file_name = f'{index}.pth'
    return file_name
