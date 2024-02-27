from flask import Flask, jsonify
from flask_cors import CORS
from Path_generator import RandomPathGenerator

app = Flask(__name__)
CORS(app)

@app.route('/generate-path', methods=['GET'])
def generate_path():
    generator = RandomPathGenerator(num_steps=200)
    path_data = generator.generate_path()
    path_data_list = path_data.tolist() if hasattr(path_data, 'tolist') else path_data
    return jsonify(path_data_list)

if __name__ == "__main__":
    app.run(debug=True)
