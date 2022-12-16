import json
import pandas as pd
from flask import Flask, request, render_template, redirect, flash
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from Coding.Algorithms.IterTD_GlobalBounds \
    import GraphTraverse as GraphTraverseNonProportional
from Coding.Algorithms.IterTD_PropBounds \
    import GraphTraverse as GraphTraverseProportional

from utils_2 import proportional_algorithm, non_proportional_algorithm, from_list_to_shape

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

UPLOAD_FOLDER = '/upload_folder'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSION = {'pdf', 'json'}
pattern_treated_unfairly_lowerbound, attributes, ranked_data, k_min = None


@app.route('/getGroups', methods=['POST'])
@cross_origin()
def getGroups():
    print('in getGroups')
    global pattern_treated_unfairly_lowerbound, attributes, ranked_data, k_min
    task_content = json.loads(request.data.decode())
    # typeOfAlgorithm = task_content['typeOfAlgorithm']
    attributes = ["sex", "age_cat", "race_factor"];
    threshold = int(task_content['threshold'])
    alpha = float(task_content['alpha'])
    k_min = int(task_content['kMin'])
    k_max = int(task_content['kMax'])

    original_data_file = r"file_csv_students"
    ranked_data = pd.read_csv(original_data_file)
    ranked_data_selected_attributes = ranked_data[attributes]
    # if task_content['numOfOption'] == "1":
    #     original_data_file = r"file_csv_students"
    #     ranked_data = pd.read_csv(original_data_file)
    #     ranked_data_selected_attributes = ranked_data[attributes]

    List_k = list(range(k_min, k_max))
    Lowerbounds = [2 for x in List_k]
    Upperbounds = [8 for x in List_k]

    pattern_treated_unfairly_lowerbound, num_patterns_visited, time = GraphTraverseProportional(
        ranked_data_selected_attributes, attributes, threshold, alpha, k_min, k_max, 60 * 10)

    return pattern_treated_unfairly_lowerbound;


@app.route('/getShapes', methods=['POST'])
@cross_origin()
def getShapes():
    print('in getShapes')
    global pattern_treated_unfairly_lowerbound, attributes, ranked_data, k_min
    shapes = from_list_to_shape(pattern_treated_unfairly_lowerbound, ranked_data, attributes, k_min)
    print(shapes)
    return "Hello"


if __name__ == "_main_":
    app.run(host="localhost", port=5001, debug=True)
