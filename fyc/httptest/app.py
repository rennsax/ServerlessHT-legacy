import requests
import json

def handler(event, context):
    ip_address = event['ip_address']
    port = event['port']
    data = [1.2, 3.4, 5.6, 7.8]
    server_address = "http://{ip_address}:{port}"
    response = requests.post(server_address,json=data)
    response_data = json.loads(response.text)
    
    print(response_data)
    return {
        'statusCode': 200,
        'body': json.loads(response.text)
    }
