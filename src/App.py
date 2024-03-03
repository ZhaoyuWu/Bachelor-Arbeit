from flask import Flask, jsonify, request
from flask_cors import CORS
from Path_generator import RandomPathGenerator
from Preprocess import preprocess_data
from DDPG_apply import apply_model_to_processed_data

app = Flask(__name__)
CORS(app)

global_path_data = None

@app.route('/generate-path', methods=['GET'])
def generate_path():
    global global_path_data
    generator = RandomPathGenerator(num_steps=200)
    path_data = generator.generate_path()
    path_data_list = path_data.tolist() if hasattr(path_data, 'tolist') else path_data
    global_path_data = path_data_list
    return jsonify(path_data_list)

@app.route('/submit-path', methods=['POST'])
def submit_path():
    global global_path_data
    frontend_data = request.json
    if global_path_data is not None:
        # preprocessing
        processed_data = preprocess_data(frontend_data, global_path_data)
        # apply DDPG to processed data
        global_path_data = apply_model_to_processed_data(processed_data)
        # print(global_path_data)
        return jsonify({"status": "success", "message": "Path data received and processed"})
    else:
        return jsonify({"status": "error", "message": "Backend path data not available"})

@app.route('/get-processed-path', methods=['GET'])
def get_processed_path():
    global global_path_data
    if global_path_data is not None:
        path_data_list = global_path_data.tolist() if hasattr(global_path_data, 'tolist') else global_path_data
        return jsonify(path_data_list)
    else:
        return jsonify({"status": "error", "message": "Processed path data not available"})

if __name__ == "__main__":
    app.run(debug=True)
