import json
import pandas as pd
from flask import Flask, request, render_template, redirect, flash, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from matplotlib import pyplot as plt

from Coding.Algorithms.IterTD_GlobalBounds \
    import GraphTraverse as GraphTraverseNonProportional
from Coding.Algorithms.IterTD_PropBounds \
    import GraphTraverse as GraphTraverseProportional
from from_list_to_shapy_values import string2list, shapley_values_att_value_seperated, get_shaped_values, \
    plot_average_shap_value_of_group, plot_distribution_ratio, get_shap_plot

from utils_2 import from_group_to_shape

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

UPLOAD_FOLDER = '/upload_folder'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSION = {'pdf', 'json'}
pattern_treated_unfairly_lowerbound = None
attributes = None
ranked_data = None
k_min = None
shaped_values = None


@app.route('/getGroups', methods=['POST'])
@cross_origin()
def getGroups():
    print('in getGroups')
    global pattern_treated_unfairly_lowerbound, attributes, ranked_data, k_min
    task_content = json.loads(request.data.decode())
    attributes = task_content['attributes']
    threshold = int(task_content['threshold'])
    alpha = float(task_content['alpha'])
    k_min = int(task_content['kMin'])
    k_max = int(task_content['kMax'])

    #original_data_file = r"file_csv_students"
    original_data_file = r"student-mat_cat_ranked.csv"
    ranked_data = pd.read_csv(original_data_file)
    ranked_data_selected_attributes = ranked_data[attributes]
    # if task_content['numOfOption'] == "1":
    #     original_data_file = r"file_csv_students"
    #     ranked_data = pd.read_csv(original_data_file)
    #     ranked_data_selected_attributes = ranked_data[attributes]

    List_k = list(range(k_min, k_max))
    Lowerbounds = [2 for x in List_k]
    Upperbounds = [8 for x in List_k]

    pattern_treated_unfairly_lowerbound, num_patterns_visited, time, patterns_size_whole, patterns_size_topk = GraphTraverseProportional(
        ranked_data_selected_attributes, attributes, threshold, alpha, k_min, k_max, 60 * 10)
    print("*****!:")
    print(pattern_treated_unfairly_lowerbound,patterns_size_topk[10].pattern_count('M||MS'))
    ans = []
    count_k = k_min
    for k in pattern_treated_unfairly_lowerbound:
        for group in k:
            ans.append([group, patterns_size_whole[group], count_k, patterns_size_topk[count_k].pattern_count(group)])
        count_k += 1
    print("&&&: ", ans)
    return ans

@app.route('/getShapleyValues', methods=['POST'])
@cross_origin()
def getShapes():
    ##########
    global pattern_treated_unfairly_lowerbound, attributes, ranked_data, k_min, shaped_values
    shapes_data = json.loads(request.data.decode())
    group = string2list(shapes_data['group'])
    shaped_values = get_shap_plot(ranked_data, attributes, attributes, attributes, group)
    print(shaped_values)
    shaped_values_per_group = shaped_values.to_dict(orient='records')
    print(shaped_values_per_group)
    result = [[item['Attribute'], item['Shapley values']] for item in shaped_values_per_group]
    return result

#TODO with JinYang
@app.route('/getDistrbution', methods=['POST'])
@cross_origin()
def getDistrbution():
    global pattern_treated_unfairly_lowerbound, attributes, ranked_data, k_min, shaped_values
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    data = json.loads(request.data.decode())
    attribute = str(data['attribute'])
    group = str(data['group'])
    k = int(data['k'])
    original_att = 'final grade'
    group_name = "\{mother's education = 5th to 9th grade\}"
    #def plot_distribution_ratio(ranked_data, attribute, selected_attributes, original_att, group, group_name, k, axis):
    boom = plot_distribution_ratio(ranked_data, attribute, attribute, original_att, group, group_name, k, ax)

    print(boom)
    return "Hello"


if __name__ == "_main_":
    app.run(host="localhost", port=5001, debug=True)
