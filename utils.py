import json
import pandas as pd
from flask import Flask, request, render_template, redirect, flash
from werkzeug.utils import secure_filename

from Coding.Algorithms.IterTD_GlobalBounds \
    import GraphTraverse as GraphTraverseNonProportional
from Coding.Algorithms.IterTD_PropBounds \
    import GraphTraverse as GraphTraverseProportional


def non_proportional_algorithm(request):
    print("in prop!!!!")
    # print("bbom")
    # print(request.form.get("algorithms"))
    json_file = request.files['file_json']
    # TODO : save in uplode_folder
    json_file.save(secure_filename("file_json"))
    csv_file = request.files['file_csv']
    csv_file.save(secure_filename("file_csv"))
    f = open('file_json')
    data = json.load(f)
    attributes = data['selected_attributes']
    thc = data['threshold']

    k_min = data['k_min']
    k_max = data['k_max']

    List_k = list(range(k_min, k_max))
    Lowerbounds = [2 for x in List_k]
    Upperbounds = [8 for x in List_k]

    original_data_file = r"file_csv"

    ranked_data = pd.read_csv(original_data_file)
    ranked_data = ranked_data[attributes]

    pattern_treated_unfairly_lowerbound, num_patterns_visited, \
        time, patterns_size_whole = GraphTraverseNonProportional(ranked_data,
                                                                 attributes, thc,
                                                                 Lowerbounds, k_min,
                                                                 k_max, 60 * 10)
    print("in 35  pattern: ", pattern_treated_unfairly_lowerbound, "    ************* num: ", num_patterns_visited)
    return "hello"


def proportional_algorithm(request):
    print("in non prop!!!!")
    # print("bbom")
    # print(request.form.get("algorithms"))
    json_file = request.files['file_json']
    # TODO : save in uplode_folder
    json_file.save(secure_filename("file_json"))
    csv_file = request.files['file_csv']
    csv_file.save(secure_filename("file_csv"))
    f = open('file_json')
    data = json.load(f)
    attributes = data['selected_attributes']
    thc = data['threshold']
    alpha = data['alpha']

    k_min = data['k_min']
    k_max = data['k_max']

    List_k = list(range(k_min, k_max))
    Lowerbounds = [2 for x in List_k]
    Upperbounds = [8 for x in List_k]

    original_data_file = r"file_csv"

    ranked_data = pd.read_csv(original_data_file)
    ranked_data = ranked_data[attributes]

    # def GraphTraverse(ranked_data, attributes, Thc, alpha, k_min, k_max, time_limit)
    # pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound, num_patterns_visited, running_time
    pattern_treated_unfairly_lowerbound, num_patterns_visited, \
        time, patterns_size_whole = GraphTraverseProportional(ranked_data, attributes,
                                                              thc, alpha, k_min,
                                                              k_max, 60 * 10)

    print("in 77 pattern: ", pattern_treated_unfairly_lowerbound, "    ************* num: ", num_patterns_visited)
    return "hello prop"
