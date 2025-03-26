from flask import Flask, jsonify

app = Flask(__name__)

# 示例数据
data = {
    "message": "Hello from Flask!",
    "data": [1, 2, 3, 4, 5]
}

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
