from flask import Flask, jsonify, request
from flask_cors import CORS
from Path_generator import RandomPathGenerator
from Preprocess import preprocess_data
from DDPG_apply import apply_model_to_processed_data
import sqlite3
import subprocess

app = Flask(__name__)
CORS(app)

global_path_data = None

conn = sqlite3.connect('processed_data.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS processed_data (
                    id INTEGER PRIMARY KEY,
                    data TEXT
                )''')
conn.commit()

# Get path from backend
@app.route('/generate-path', methods=['GET'])
def generate_path():
    global global_path_data
    generator = RandomPathGenerator(num_steps=200)
    path_data = generator.generate_path()
    path_data_list = path_data.tolist() if hasattr(path_data, 'tolist') else path_data
    global_path_data = path_data_list
    return jsonify(path_data_list)

# Process data
@app.route('/submit-path', methods=['POST'])
def submit_path():
    global global_path_data
    frontend_data = request.json
    if global_path_data is not None:
        # preprocessing
        processed_data = preprocess_data(frontend_data, global_path_data)


        #==================================================================================================
        #The following code controls whether frontend data is saved in the database.
        # ==================================================================================================
        # cursor.execute("INSERT INTO processed_data (data) VALUES (?)", (str(processed_data),))
        # conn.commit()
        #
        # cursor.execute("SELECT MAX(id) FROM processed_data")
        # max_id = cursor.fetchone()[0]
        #
        # cursor.execute("SELECT COUNT(*) FROM processed_data")
        # count = cursor.fetchone()[0]
        #
        # if count > 20:
        #     cursor.execute("DELETE FROM processed_data WHERE id = (SELECT MIN(id) FROM processed_data)")
        #     conn.commit()
        #
        # if max_id is not None and max_id % 10 == 0:
        #     run_ddpg_script()
        # ==================================================================================================


        # apply DDPG to processed data
        global_path_data = apply_model_to_processed_data(processed_data)
        return jsonify({"status": "success", "message": "Path data received and processed"})
    else:
        return jsonify({"status": "error", "message": "Backend path data not available"})

# Get data from frontend.
@app.route('/get-processed-path', methods=['GET'])
def get_processed_path():
    global global_path_data
    if global_path_data is not None:
        path_data_list = global_path_data.tolist() if hasattr(global_path_data, 'tolist') else global_path_data
        return jsonify(path_data_list)
    else:
        return jsonify({"status": "error", "message": "Processed path data not available"})

def run_ddpg_script():
    subprocess.run(["python", "DDPG.py"], check=True)


if __name__ == "__main__":
    app.run(debug=True)
