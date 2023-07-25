import base64
import json
import pickle
import queue
import sys
import threading

import boto3
import numpy as np
import requests
from flask import Flask, jsonify, request

step_request_queue = {}  # 请求队列
step_response_queue = {}  # 响应队列
step_tmp_queue = {}

restart_request_queue = queue.Queue()  # 重启请求队列
restart_response_queue = queue.Queue()  # 重启响应队列

# FIXME 拼错了？
waitrall_average_events = {}
overall_average_events = {}  # Event objects for overall average calculation
groups_number = []
current_number = []
if_conflict = []
# FIXME 这个变量名和 Python 内置函数冲突了，修改一下
sum = []

app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_post_request():
    request_data = request.get_json()

    ip = request.remote_addr
    print("client ip:", ip)

    if request_data.get("request") == "average_step":
        group_id = int(request_data.get("group_id"))
        len_param = request_data.get("len_param")
        loss = request_data.get("loss")
        epoch = request_data.get("epoch")

        if epoch == sum[group_id]:
            # (group_id) can get data
            waitrall_average_events[group_id].set()

            step_request_queue[group_id].put(request_data)

            print(group_id, "_", len_param, "epoch:", epoch, "_", loss)

            # Wait for overall average calculation to complete
            if overall_average_events[group_id].is_set():
                overall_average_events[group_id].clear()
            overall_average_events[group_id].wait()

            print(group_id, "go")

            # 获取响应数据
            response_data = step_response_queue[group_id].get()

        else:
            response_data = {"model_weights": " "}

    elif request_data.get("request") == "restart":
        restart_request_queue.put(
            request_data
        )  # Put the restart request data into the restart request queue
        response_data = restart_response_queue.get()  # Wait to get the response data

    return jsonify(response_data)


def restart_worker():
    while True:
        request_data = (
            restart_request_queue.get()
        )  # Get the restart request data from the restart request queue
        # Process the restart request and generate the response data
        response = restart(request_data)
        restart_response_queue.put(
            response
        )  # Put the response data into the restart response queue


def step_worker(group_id):
    overall_average_event = overall_average_events[group_id]
    waitrall_average_event = waitrall_average_events[group_id]

    while True:
        if waitrall_average_event.is_set():
            waitrall_average_event.clear()
        waitrall_average_event.wait()  # wait for call

        while not step_request_queue[group_id].empty():
            request_data = step_request_queue[
                group_id
            ].get()  # Get the step request data from the step request queue

            print("receive one")
            current_number[group_id] += 1

            step_tmp_queue[group_id].put(request_data)  # 将请求数据放入响应队列

            # Check if all Lambda requests have arrived for the group
            if current_number[group_id] == groups_number[group_id]:
                sum[group_id] += 1
                print(group_id, "done")
                step_average_all = calculate_overall_average(step_tmp_queue[group_id])

                print("calculate done")

                if step_average_all is None:
                    print("step_average_all is None")

                response_data = {"model_weights": step_average_all, "restart": 1}

                for _ in range(groups_number[group_id]):
                    step_response_queue[group_id].put(response_data)
                    # response_queue.get()

                while if_conflict[group_id] > 0:
                    step_response_queue[group_id].put(response_data)
                    if_conflict[group_id] -= 1

                current_number[group_id] = 0

                overall_average_event.set()


def restart(data):
    # Get instance URL and other necessary data from 'data' dictionary

    file_name = data.get("file_name")
    bucket_name = data.get("bucket_name")
    params_file_name = data.get("params_file_name")
    group_id = int(data.get("group_id"))
    epochs_done = data.get("epochs_done")
    ip_address = data.get("ip_address")
    port = data.get("port")
    len_param = data.get("len_param")
    input_lr = data.get("lr")
    input_momentum = data.get("momentum")
    total_epoch = data.get("total_epoch")
    reinvoke_time = data.get("reinvoke_time")
    batch_size = data.get("batch_size")
    model_s3path = data.get("model_s3path")
    data_s3path = data.get("data_s3path")

    temp_queue = queue.Queue()

    # Remove requests with the same len_param from step_tmp_queue[group_id]
    while not step_tmp_queue[group_id].empty():
        step_request_data = step_tmp_queue[group_id].get()
        if step_request_data.get("len_param") != len_param:
            step_request_data["if_handle"] = True
            temp_queue.put(step_request_data)

        else:
            step_request_data["if_handle"] = False
            temp_queue.put(step_request_data)
            current_number[group_id] -= 1
            if_conflict[group_id] += 1

    # Put the remaining requests back to step_tmp_queue[group_id]
    while not temp_queue.empty():
        step_request_data = temp_queue.get()
        step_tmp_queue[group_id].put(step_request_data)

    epochs = data.get("epochs_done")
    print("epochs_done:", epochs)
    print(file_name)

    # Prepare payload for Lambda invocation
    payload = {
        "if_restart": True,
        "ip_address": ip_address,
        "port": port,
        "bucket_name": bucket_name,
        "params_file_name": params_file_name,
        "group_id": group_id,
        "epoch": epochs_done,
        "num_parts": groups_number[group_id],
        "len_param": len_param,
        "lr": input_lr,
        "momentum": input_momentum,
        "total_epoch": total_epoch,
        "reinvoke_time": reinvoke_time,
        "batch_size": batch_size,
        "model_s3path": model_s3path,
        "data_s3path": data_s3path,
    }

    # Invoke Lambda function asynchronously
    lambda_client = boto3.client("lambda", region_name="ap-northeast-1")
    lambda_client.invoke(
        FunctionName=file_name, InvocationType="Event", Payload=json.dumps(payload)
    )

    print("restart success:")

    response = {"message": "CLOSE"}

    return response


def calculate_overall_average(response_queue):
    params_dict = {}

    while not response_queue.empty():
        response = response_queue.get()
        if_handle = response.get("if_handle")
        if if_handle == False:
            continue

        model_weights_base64 = response.get("step")
        model_weights_bytes = base64.b64decode(model_weights_base64)
        step = pickle.loads(model_weights_bytes)

        for key, value in step.items():
            if key in params_dict:
                params_dict[key].append(value)
            else:
                params_dict[key] = [value]

    overall_average = {}

    for key, values in params_dict.items():
        num_arrays = len(values)  # 数组的个数
        array_shape = np.array(values[0]).shape  # 获取数组的形状
        average_array = np.zeros(array_shape)  # 创建一个全零数组作为初始平均值
        for i in range(num_arrays):
            array = np.array(values[i])  # 当前数组
            average_array += array  # 逐元素相加
        average_array = np.divide(average_array, num_arrays)
        overall_average[key] = average_array.tolist()

    row_overall_average = pickle.dumps(overall_average)
    model_weights_base64 = base64.b64encode(row_overall_average).decode("utf-8")

    return model_weights_base64


def get_instance_url():
    metadata_url = "http://169.254.169.254/latest/meta-data/public-ipv4"
    response = requests.get(metadata_url)
    if response.status_code == 200:
        instance_ip = response.text
        return instance_ip
    else:
        return None


def main():
    # Create and start the thread for processing restart requests
    restart_thread = threading.Thread(target=restart_worker)
    restart_thread.start()

    ip_address = get_instance_url()
    # FIXME 如果有空，用 argparse 库重写一下
    args = sys.argv  # get input parameters

    total_groups = args[1] if len(args) > 1 else None  # the number of training groups
    num_parts = args[2] if len(args) > 2 else None
    port = args[3] if len(args) > 3 else None
    bucket_name = args[4] if len(args) > 4 else None
    batch_size = args[5] if len(args) > 5 else None
    lr = args[6] if len(args) > 6 else None
    momentum = args[7] if len(args) > 7 else None
    epoch = args[8] if len(args) > 8 else None
    reinvoke_time = args[9] if len(args) > 9 else None
    model_s3path = args[10] if len(args) > 10 else None
    data_s3path = args[11] if len(args) > 11 else None

    # Create and start the thread for processing step requests for each group
    for group_id in range(int(total_groups)):
        groups_number.append(int(num_parts))
        current_number.append(0)
        overall_average_events[group_id] = threading.Event()
        waitrall_average_events[group_id] = threading.Event()
        step_thread = threading.Thread(target=step_worker, args=(group_id,))
        step_thread.start()

        step_request_queue[group_id] = queue.Queue()
        step_response_queue[group_id] = queue.Queue()
        step_tmp_queue[group_id] = queue.Queue()

        if_conflict.append(0)
        sum.append(1)  # FIXME 这里的 1 是什么意思？

        payload = {
            "if_restart": 0,
            "ip_address": ip_address,
            "port": port,
            "bucket_name": bucket_name,
            "params_file_name": "none",
            "group_id": group_id,
            "epoch": 0,
            "num_parts": num_parts,
            "lr": lr,
            "momentum": momentum,
            "total_epoch": epoch,
            "reinvoke_time": reinvoke_time,
            "batch_size": batch_size,
            "model_s3path": model_s3path,
            "data_s3path": data_s3path,
        }

        lambda_client = boto3.client("lambda", region_name="ap-northeast-1")

        for i in range(int(num_parts)):
            # Invoke Lambda function asynchronously
            payload["len_param"] = i
            print(json.dumps(payload))
            lambda_client.invoke(
                FunctionName="Hyperparameter_optimization_group_rfm",
                InvocationType="Event",
                Payload=json.dumps(payload),
            )

    app.run(host="0.0.0.0", port=int(port))


if __name__ == "__main__":
    main()
