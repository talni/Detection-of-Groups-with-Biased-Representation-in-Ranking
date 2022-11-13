import json
import pandas as pd
from flask import Flask, request, render_template, redirect, flash
from werkzeug.utils import secure_filename

from Coding.Algorithms.IterTD_GlobalBounds\
    import GraphTraverse as GraphTraverseNonProportional
from Coding.Algorithms.IterTD_PropBounds\
    import GraphTraverse as GraphTraverseProportional



app = Flask(__name__)
UPLOAD_FOLDER = '/upload_folder'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSION = {'pdf', 'json'}

@app.route('/GraphTraverseNonProportional', methods=['POST'])
def index1():
    file = request.files['file']
    file.save(secure_filename(file.filename))

    f = open('upload_folder/test.json')
    data = json.load(f)

    attributes = data['selected_attributes']
    thc = data['threshold']

    k_min = data['k_min']
    k_max = data['k_max']

    List_k = list(range(k_min, k_max))
    Lowerbounds = [2 for x in List_k]
    Upperbounds = [8 for x in List_k]

    original_data_file = r"upload_folder/input.csv"

    ranked_data = pd.read_csv(original_data_file)
    ranked_data = ranked_data[attributes]

    pattern_treated_unfairly_lowerbound, num_patterns_visited, time = GraphTraverseNonProportional(ranked_data, attributes, thc, Lowerbounds, k_min, k_max, 60*10)
    print("in 35  pattern: ", pattern_treated_unfairly_lowerbound, "    ************* num: ", num_patterns_visited)
    return "hello"

@app.route('/GraphTraverseProportional', methods=['POST'])
def index2():
    task_content = request.form['content']
    ranked_data = task_content['task_content']
    attributes = task_content['attributes']
    alpha = task_content['alpha']
    k_min = task_content['k_min']
    k_max = task_content['k_max']
    pattern_treated_unfairly_lowerbound, num_patterns_visited, time = GraphTraverseProportional(ranked_data, attributes, Thc, Lowerbounds, k_min, k_max, 60*10)
    return pattern_treated_unfairly_lowerbound
    return "Hello, world!"


@app.route('/', methods=['GET'])
def index3():
    error = None
    print("tal is king")
    return render_template('home.html')

@app.route('/Graph', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        print("In in post")
        print(request.__dict__)
        file = request.files['file']
        print("1", file.filename)
        file.save(secure_filename(file.filename))
        print("2")
        print(file)
        return "ooga booga" # todo: different template
    return "hello"


if __name__ == "_main_":
    app.run(debug=True)