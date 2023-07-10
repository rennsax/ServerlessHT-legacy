from flask import Flask, request, jsonify
import json
import time
import requests
import torch
import boto3

def handler(event, context):
    
    func_start = time.time()
    time.sleep(3)

    # Calculate the total time taken by the function
    current_time = time.time()
    total_time = current_time - func_start

    # Get the function name and server address
    fun_name = context.function_name
    server_address = event['Payload']['server_address']

    data = {
        "time": total_time,
        "index": fun_name
    }

    # Send HTTP request
    response = requests.post(server_address, json=data)

    response_data = json.loads(response.text)

    if response_data['message'] == 'CLOSE':
        print("close")
    
    # Save parameters to S3
    worker_index = event['worker_index']
    cur_iter = event['cur_iter']
    time_pre = event['time_pre']
    train_time_pre = event['train_time_pre']
    commu_time_pre = event['commu_time_pre']
    output_bucket = event['output_bucket']
    path = event['path']

    # Load the model
    net = load_model() 

    # Save the model parameters and checkpoint to a file
    PATH = '/tmp/net.pth'
    torch.save({
        'model_state_dict': net.state_dict(),
        'cur_iter': cur_iter,
        'time_pre': time_pre,
        'train_time_pre': train_time_pre,
        'commu_time_pre': commu_time_pre
    }, PATH)
    
    # Upload the file to the specified output bucket in S3
    s3_client = boto3.client('s3')
    file_name = f'net_w{worker_index}'
    s3_client.upload_file(PATH, output_bucket, path + file_name)

    return {"result": "succeed!"}

def load_model():
    pass
