from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def calculate_average():
    data = request.get_json()
    print("connected")
    if not isinstance(data, list) or not all(isinstance(num, float) for num in data):
        return jsonify({'error': 'Invalid input. Expected a list of floats.'}), 400
    average = sum(data) / len(data)
    return jsonify({'average': average})

if __name__ == '__main__':
    print("connecting")
    app.run(host='0.0.0.0', port=5000)