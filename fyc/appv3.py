from flask import Flask, request, jsonify
import json
import time
import requests
import torch
import boto3
import threading
import logging
import torchvision
import torchvision.transforms as transforms
import os

session = boto3.Session(
    aws_access_key_id='AKIAWZDD4R7ZCFG4H75Z',
    aws_secret_access_key='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',
    region_name='ap-northeast-1'  # 替换为实际的AWS区域
)

s3_client = session.client('s3')
logging.basicConfig(level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def handler(event, context):

    time_thread = threading.Thread(target=time_watcher, args=(event, context.function_name))
    time_thread.start()

    # TODO: train model

    ip_address = event['ip_address']
    port = event['port']
    if_restart = event['if_restart']
    bucket_name = event['bucket_name']
    params_file_name = event['params_file_name']
    group_id = event['group_id']
    epochs_done = event['epoch']
    
    server_address = f"http://{ip_address}:{port}"

    data_path = '/tmp/cifar10_data'

    # 加载 CIFAR-10 数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    # 创建 ResNet-18 模型
    model = torchvision.models.resnet18(pretrained=False)

    # if_restart
    if if_restart == "True":
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, params_file_name, '/tmp/model_weights.pth')
        model.load_state_dict(torch.load('/tmp/model_weights.pth'))
    else:
        epochs_done = 0

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs_done, 20):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # 将输入和标签放入设备上
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零参数梯度
            optimizer.zero_grad()

            # 正向传播、反向传播、优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计损失
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/100:.3f}')
                running_loss = 0.0

        # 将模型权重发送给服务器并接收新权重
        model_weights = model.state_dict()
        message = {
            'group_id' : group_id,
            'request': 'average_step',
            'step': model_weights
        }
        response = requests.post(server_address, json=message)
        response_data = json.loads(response.text)

        # 更新模型权重
        new_model_weights = response_data.get('model_weights')
        if new_model_weights:
            model.load_state_dict(new_model_weights)

        # 保存文件
        file_name = '/tmp/model.pth'
        torch.save(model.state_dict(), file_name)
        with open('/tmp/epoch.txt', 'w') as file:
            file.write(str(epoch))

    return {
        'statusCode': 200,
        'body': model_weights
    }


def time_watcher(event, function_name):

    ip_address = event['ip_address']
    port = event['port']
    if_restart = event['if_restart']
    bucket_name = event['bucket_name']
    params_file_name = event['params_file_name']
    group_id = event['group_id']
    epochs_done = event['epoch']
    func_start = time.time()

    time.sleep(60)

    # Calculate the total time taken by the function
    current_time = time.time()
    total_time = current_time - func_start

    while(total_time < 60):
        current_time = time.time()
        total_time = current_time - func_start 

    # Save parameters to S3

    with open('/tmp/epoch.txt', 'r') as file:
        epoch = int(file.read())
    PATH = '/tmp/model.pth'
    #the place in s3
    params_file_name = f'/tmp/{file_name}_{epoch}.pth'
    
    # Upload the file to the specified output bucket in S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(PATH, bucket_name, params_file_name)

    # Get server address
    
    server_address = f"http://{ip_address}:{port}"

    data = {
        'file_name' : function_name,
        'request' : 'restart',
        'ip_address' : ip_address,
        'port' : port,
        'bucket_name' : bucket_name,
        'params_file_name' : params_file_name,
        'group_id' : group_id,
        'epochs_done' : epoch
    }

    # Send HTTP request
    response = requests.post(server_address, json=data)

    response_data = json.loads(response.text)

    if response_data['message'] == 'CLOSE':
        logging.info("CLOSE")
    
    return 0

