import base64
import json
import os
import pickle
import random
import shutil
import tarfile
import threading
import time

import boto3
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image


def handler(event, context):
    print("Handler function started")

    del_file("/tmp")  # clear /tmp

    print("del_file")

    # 创建一个锁对象
    lock = [0, 0, 0]

    time_thread = threading.Thread(
        target=time_watcher, args=(event, context.function_name, lock)
    )
    time_thread.daemon = True  # daemon thread
    time_thread.start()

    train_thread = threading.Thread(target=train_model, args=(event, context, lock))
    train_thread.daemon = True  # daemon thread
    train_thread.start()

    while True:
        if lock[0] == 1:
            break
        time.sleep(0.1)

    lock[2] = 1

    print("real_end")

    return {"statusCode": 200, "body": "end"}


# train model
def train_model(event, context, lock):
    print("start train")

    session = boto3.Session()

    s3_client = session.client("s3")

    print("s3_client")

    device = torch.device("cpu")
    ip_address = event["ip_address"]
    port = event["port"]
    if_restart = int(event["if_restart"])
    bucket_name = event["bucket_name"]
    params_file_name = event["params_file_name"]
    group_id = event["group_id"]
    epochs_done = event["epoch"]
    len_param = int(event["len_param"])
    num_parts = int(event["num_parts"])
    input_lr = float(event["lr"])
    input_momentum = float(event["momentum"])
    total_epoch = int(event["total_epoch"])
    batch_size = int(event["batch_size"])
    model_s3path = event["model_s3path"]
    data_s3path = event["data_s3path"]
    test_data_path = event["test_data_path"]

    server_address = f"http://{ip_address}:{port}"

    # Set random seed
    random_seed = 200
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    data_path = "/tmp/local_data"
    os.makedirs(data_path)

    # write 0
    with open("/tmp/epoch.txt", "w") as file:
        file.write("0")

    print("file.write")

    # 下载数据集文件
    s3_client.download_file(bucket_name, data_s3path, "/tmp/dataset.tar.gz")
    print("数据集文件已下载")

    # 解压缩数据集文件
    with tarfile.open("/tmp/dataset.tar.gz", "r:gz") as tar:
        tar.extractall(data_path)
    print("数据集文件已解压缩")

    # 创建数据集，调整图像尺寸和通道数
    all_train_dataset = create_dataset(data_path)
    print("已创建数据集")

    # split the data
    datasets = split_dataset(all_train_dataset, num_parts)
    train_dataset = datasets[len_param]

    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print("Data loader created")

    # Create model
    model = load_model_from_path(model_s3path, device)
    print("model created")

    # Check if it needs to restart
    if if_restart == 1:
        s3_client = boto3.client("s3")
        s3_client.download_file(bucket_name, params_file_name, "/tmp/model_weights.pth")
        model.load_state_dict(torch.load("/tmp/model_weights.pth"))
        print("Model weights loaded from S3")
    else:
        epochs_done = 0

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=input_lr, momentum=input_momentum
    )
    last_train_info = ""

    for epoch in range(epochs_done, total_epoch):
        print(f"Epoch {epoch+1} started")
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # Move inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward, backward, and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track the loss
            running_loss += loss.item()

            if i % 100 == 99:
                print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/100:.3f}")
                running_loss = 0.0

        # 在 Epoch 完成后，记录最后一次训练信息
        last_train_info = (
            f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/100:.3f}"
        )

        # Send the model weights to the server and receive new weights
        model_weights = model.state_dict()

        # Convert model weights to NumPy arrays
        model_weights_numpy = {}
        for key, tensor in model_weights.items():
            model_weights_numpy[key] = tensor.detach().cpu().numpy().tolist()

        # Convert model weights to binary data
        model_weights_binary = pickle.dumps(model_weights_numpy)

        # Convert byte data to Base64-encoded string
        model_weights_base64 = base64.b64encode(model_weights_binary).decode("utf-8")

        epochs = epoch + 1
        message = {
            "group_id": group_id,
            "len_param": len_param,
            "request": "average_step",
            "step": model_weights_base64,
            "loss": last_train_info,
            "epoch": epochs,
        }

        if lock[1] == 1:
            return 0

        response = requests.post(server_address, json=message)
        response_data = json.loads(response.text)

        row_new_model_weights = response_data.get("model_weights")

        if not row_new_model_weights:
            lock[0] = 1
            return 0

        # Update model weights
        model_weights_bytes = base64.b64decode(row_new_model_weights)
        new_model_weights = pickle.loads(model_weights_bytes)

        if new_model_weights:
            new_state_dict = {}
            for key, value in new_model_weights.items():
                new_state_dict[key] = torch.tensor(value)
            model.load_state_dict(new_state_dict)

        # Save the model file
        file_name = "/tmp/model.pth"
        torch.save(model.state_dict(), file_name)
        with open("/tmp/epoch.txt", "w") as file:
            file.write(str(epoch + 1))
        print(f"Epoch {epoch+1} completed")

        if lock[1] == 1:
            return 0

    # 训练循环结束后，将最后一次训练信息写入 out.txt 文件
    with open("/tmp/out.txt", "w") as file:
        file.write(last_train_info)

    # 在训练循环完成后
    # 加载测试数据集并创建数据加载器
    print("accuracy启动")
    local_data_path = "/tmp/local_acc"
    os.makedirs(data_path)
    # 下载数据集文件
    s3_client.download_file(bucket_name, test_data_path, "/tmp/datasetacc.tar.gz")
    print("数据集文件已下载")

    # 解压缩数据集文件
    with tarfile.open("/tmp/datasetacc.tar.gz", "r:gz") as tar:
        tar.extractall(local_data_path)
    print("数据集文件已解压缩")

    # 创建数据集，调整图像尺寸和通道数
    test_dataset = create_dataset(local_data_path)
    print("已创建数据集")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print("test_loader")
    # 计算模型在测试数据集上的准确率
    accuracy = calculate_accuracy(model, test_loader)
    print("模型在测试数据集上的准确率：", accuracy)

    message = {
            "group_id": group_id,
            "len_param": len_param,
            "request": "accuracy",
            "accuracy": accuracy,
        }
    
    if lock[1] == 1:
        return 0

    response = requests.post(server_address, json=message)

    # 将 epoch.txt 文件上传到指定的 S3 桶中
    params_file_name = f"/output_{context.function_name}_{epoch}_{group_id}_{len_param}.pth"  # 替换为您希望存储的 S3 对象的路径
    s3_client.upload_file("/tmp/out.txt", bucket_name, params_file_name)

    print("Handler function finished")

    lock[0] = 1
    return 0


def time_watcher(event, function_name, lock):
    print("Time watcher function started")

    session = boto3.Session()

    s3_client = session.client("s3")
    ip_address = event["ip_address"]
    port = event["port"]
    bucket_name = event["bucket_name"]
    params_file_name = event["params_file_name"]
    group_id = event["group_id"]
    len_param = event["len_param"]
    num_parts = event["num_parts"]
    reinvoke_time = int(event["reinvoke_time"])
    input_lr = float(event["lr"])
    input_momentum = float(event["momentum"])
    total_epoch = int(event["total_epoch"])
    batch_size = int(event["batch_size"])
    model_s3path = event["model_s3path"]
    data_s3path = event["data_s3path"]
    test_data_path = event["test_data_path"]

    func_start = time.time()

    # Calculate the total time taken by the function
    current_time = time.time()
    total_time = current_time - func_start

    while total_time < reinvoke_time:
        current_time = time.time()
        time.sleep(0.1)
        if lock[2] == 1:
            return 0
        total_time = current_time - func_start

    lock[1] = 1
    # Save parameters to S3
    with open("/tmp/epoch.txt", "r") as file:
        epoch = int(file.read())
    if epoch == 0:
        print("Lambda time limit")
        return 0

    PATH = "/tmp/model.pth"
    # The place in S3
    params_file_name = f"/tmp/{function_name}_{epoch}_{group_id}_{len_param}.pth"

    # Upload the file to the specified output bucket in S3
    s3_client = boto3.client("s3")
    s3_client.upload_file(PATH, bucket_name, params_file_name)

    # Get server address
    server_address = f"http://{ip_address}:{port}"

    data = {
        "file_name": function_name,
        "request": "restart",
        "ip_address": ip_address,
        "port": port,
        "bucket_name": bucket_name,
        "params_file_name": params_file_name,
        "group_id": group_id,
        "epochs_done": epoch,
        "num_parts": num_parts,
        "len_param": len_param,
        "lr": input_lr,
        "momentum": input_momentum,
        "total_epoch": total_epoch,
        "reinvoke_time": reinvoke_time,
        "batch_size": batch_size,
        "model_s3path": model_s3path,
        "data_s3path": data_s3path,
        "test_data_path": test_data_path   
    }

    # Send HTTP request
    response = requests.post(server_address, json=data)

    response_data = json.loads(response.text)

    if response_data["message"] == "CLOSE":
        print("CLOSE")

    lock[0] = 1

    print("Time watcher function finished")

    return 0


def del_file(filepath):
    # Clear the filepath
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def split_dataset(dataset, num_parts):
    total_length = len(dataset)
    len_per_part = total_length // num_parts
    datasets = []
    for i in range(num_parts):
        start_idx = i * len_per_part
        end_idx = start_idx + len_per_part
        sub_dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
        datasets.append(sub_dataset)
    return datasets


def load_model_from_path(path, device):
    model = torch.load(path, map_location=device)
    model.eval()  # 设置为评估模式
    return model


def create_dataset(data_path):
    """
    创建数据集，不调整图像尺寸和通道数

    参数：
    data_path (str)：数据集路径

    返回：
    dataset (List)：包含原始图像的数据集列表，每个元素是一个元组 (image, filename)
    """

    dataset = []

    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)

        # 检查是否为文件（不是目录）
        if os.path.isfile(filepath):
            img = Image.open(filepath)

            # 将图像转换为张量
            transform = transforms.ToTensor()
            img = transform(img)

            # 添加到数据集中
            dataset.append((img, filename))

    return dataset

def calculate_accuracy(model, test_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
