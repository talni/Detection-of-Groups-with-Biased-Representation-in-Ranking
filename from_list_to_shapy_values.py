import copy

import pandas as pd
import sys
import math
from itertools import combinations

from numba import typeof

from Coding.Algorithms import pattern_count
import time
import numpy as np
import copy
import sys
import math
from sklearn.linear_model import LinearRegression
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


sys.path.append('../Coding')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.size'] = '22'

sns.set_palette("Paired")
# sns.set_palette("deep")
sns.set_context("poster", font_scale=2)
sns.set_style("whitegrid")
# sns.palplot(sns.color_palette("deep", 10))
# sns.palplot(sns.color_palette("Paired", 9))

line_style = ['o-', 's--', '^:', '-.p']
color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
# plt_title = ["BlueNile", "COMPAS", "Credit Card"]
#
# label = ["PropBounds", "IterTD"]
line_width = 8
marker_size = 20
# f_size = (18, 16)
FONTSIZE = 50



def string2list(st):
    p = list()
    idx = 0
    item = ''
    i = ''
    for i in st:
        if i == '|':
            if item == '':
                p.append(-1)
            else:
                if item.isnumeric():
                    p.append(int(item))
                else:
                    p.append(item)
                item = ''
            idx += 1
        else:
            item += i
    if i != '|':
        if item.isnumeric():
            p.append(int(item))
        else:
            p.append(item)
    else:
        p.append(-1)
    return p


def P1DominatedByP2(P1, P2):
    length = len(P1)
    for i in range(length):
        if P1[i] == -1:
            if P2[i] != -1:
                return False
        if P1[i] != -1:
            if P2[i] != P1[i] and P2[i] != -1:
                return False
    return True

def idx_of_tuples_in_group(group, data):
    def belong_to_group(row):
        nonlocal group
        if P1DominatedByP2(row, group):
            return True
        else:
            return False
    data["in"] = data.apply(belong_to_group, axis=1)
    return data[data["in"] == True].index

def average_shapley_values_of_group(data, group, all_attributes, shap_values):
    # get all tuples in this group
    data1 = data[all_attributes].copy(deep=True)
    tuples_idx = idx_of_tuples_in_group(group, data1).to_list()
    if len(tuples_idx) == 0:
        # output_file.write("\ngroup {} size {}\n".format(group, len(tuples_idx)))
        print("group {} size {}".format(group, len(tuples_idx)))
        return []
    else:
        avg = np.average(shap_values.values[tuples_idx], axis=0)
        print("group {} size {}".format(group, len(tuples_idx)))
        # print(avg)
        return list(avg)


def idx_of_tuples_in_group_w_attribute(group, data, att, value):
    def belong_to_group(row):
        nonlocal group
        if P1DominatedByP2(row, group):
            if row[att] == value:
                return True
            else:
                return False
        else:
            return False
    data["in"] = data.apply(belong_to_group, axis=1)
    idx = data[data["in"] == True].index
    data.drop(columns=["in"], axis=1, inplace=True)
    return idx


def shapley_values_att_value_seperated(data, group, all_attributes, shap_values, output_file):
    # get all tuples in this group
    data1 = data[all_attributes].copy(deep=True)
    att_idx = 0
    for att in all_attributes:
        for v in range(int(data1.describe()[att]["min"]), int(data1.describe()[att]["max"]) + 1):
            tuples_idx = idx_of_tuples_in_group_w_attribute(group, data1, att, v).to_list()
            if len(tuples_idx) == 0:
                output_file.write("group {} att {} = {} size {}\n".format(group, att, v, len(tuples_idx)))
                print("group {} att {} = {} size {}".format(group, att, v, len(tuples_idx)))
            else:
                values_of_group = shap_values.values[tuples_idx]
                avg = np.average(values_of_group, axis=0)
                print("group {} att {} = {} size {}\n avg {}".format(group, att, v, len(tuples_idx), avg))
                output_file.write("group {} att {} = {} size {} avg {}\n".format(group, att, v, len(tuples_idx), avg[att_idx]))
        att_idx += 1

def tuples_in_group(g, data, selected_attributes):
    tuple_idx = idx_of_tuples_in_group(g, data[selected_attributes].copy(deep=True))
    tuples = data.iloc[tuple_idx]
    return tuples

def tuples_not_in_group(g, data, selected_attributes):
    tuple_idx = idx_of_tuples_in_group(g, data[selected_attributes].copy(deep=True))
    return data.drop(tuple_idx)




def plot_distribution_ratio(ranked_data, attribute, selected_attributes, original_att, group, group_name, k, axis):
    x_list = ranked_data[attribute].unique()
    x_list.sort()
    print("unique values of {} = {}".format(attribute, x_list))

    tuples = tuples_in_group(group, ranked_data, selected_attributes)
    print("num of tuples in group {} = {}".format(len(tuples), group))
    s = tuples[attribute].value_counts().sort_index()
    total = sum(s)
    group_value_dis = [s[i]/total if i in s else 0 for i in x_list]

    s = ranked_data[:k][attribute].value_counts().sort_index()
    total = sum(s)
    topkdis = [s[i]/total if i in s else 0 for i in x_list]

    bar_width = 0.45
    index = np.arange(len(x_list))
    print(index)
    axis.bar(index, group_value_dis, bar_width, color=color[3], label=group_name)
    axis.bar(index + bar_width, topkdis, bar_width,  color=color[7], label="top-k")
    # plt.xticks(index + bar_width, x_list)
    # plt.xticks(range(x_list[0], x_list[-1]+1))
    index2 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    axis.set_xticks([x + bar_width/2 for x in index2], [0, 4, 6, 8, 10, 12, 14, 16, 18, 20], fontsize=FONTSIZE)
    axis.set_yticks([0, 0.1, 0.2, 0.3], [0, 0.1, 0.2, 0.3], fontsize=FONTSIZE)
    axis.set_ylabel('Proportion', fontsize=FONTSIZE)
    axis.set_xlabel('Value of '+ original_att, fontsize=FONTSIZE)
    axis.legend(loc='upper left', fontsize=40, bbox_to_anchor=(-0.02, 1.04))
    # plt.tight_layout()
    # plt.savefig("adult_time.png", bbox_inches='tight')
    # plt.show()
    return axis



def plot_average_shap_value_of_group(data, group, selected_attributes, all_attributes_original, shap_values, axis):
    s = average_shapley_values_of_group(data, group, selected_attributes, shap_values)
    df=pd.DataFrame({'Attribute':all_attributes_original, 'Shapley values':s})

    df.sort_values(by='Shapley values',key=abs, inplace=True,ascending=False)

    small_shap_values = df[6:]
    summary_shap_values = df[:6]

    summary_shap_values = summary_shap_values.append({'Attribute': 'other positive Shapley values', 'Shapley values': sum([x if x > 0 else 0 for x in small_shap_values['Shapley values']])}, ignore_index=True)
    summary_shap_values = summary_shap_values.append({'Attribute': 'other negative Shapley values', 'Shapley values': sum([x if x < 0 else 0 for x in small_shap_values['Shapley values']])}, ignore_index=True)
    print("-------5")
    print(summary_shap_values)
    # print("-------6")
    # summary_shap_values = summary_shap_values[::-1]
    # print("-------7"+summary_shap_values)
    # # summary_shap_values.plot(kind='barh',x='Attribute',y='Shapley values',color=[color[4] if t > 0 else color[0] for t in summary_shap_values['Shapley values']], figsize=(18, 16), legend=False, fontsize=FONTSIZE, ax=axis)
    # summary_shap_values.plot(kind='barh',x='Attribute',y='Shapley values',color=[color[4] if t > 0 else color[0] for t in summary_shap_values['Shapley values']], legend=False, fontsize=FONTSIZE, ax=axis)
    # axis.set_ylabel('Attribute', fontsize=FONTSIZE)
    # print("-------8" + summary_shap_values)
    # plt.show()
    # plt.xlabel('Shapley values', fontsize=FONTSIZE)
    # plt.ylabel('Attribute', fontsize=FONTSIZE)
    # plt.tight_layout()
    # print("-------9" + plt)
    #fig.set(xlabel='Shapley values')
    return summary_shap_values



def get_shap_plot(ranked_data, all_attributes, selected_attributes, all_attributes_original, group):
    x = ranked_data[all_attributes]
    y = ranked_data['rank']
    # have to convert strings to numbers for linear regression
    def convert_string_to_number(df, all_attributes):
        col_idx = 0
        def convert_to_number(column):
            nonlocal col_idx
            if column.dtypes == 'object':
                unique_values = sorted(column.unique())
                print("--------0")
                print(all_attributes[col_idx], unique_values)
                df[all_attributes[col_idx]].replace(to_replace=unique_values,
                                  value=range(1, len(unique_values)+1), inplace=True)
            col_idx += 1
        df.apply(convert_to_number, axis=0)
        return df

    x = convert_string_to_number(x, all_attributes)

    # with sklearn
    model = LinearRegression()
    model.fit(x, y)
    #print("--------1")
    print("Model coefficients:\n")
    for i in range(x.shape[1]):
        print(x.columns[i], "=", model.coef_[i].round(5))
    # compute the SHAP values for the linear model
    explainer = shap.Explainer(model.predict, x)
    shap_values = explainer(x)
    #print("--------2")
    print(shap_values)


    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    print("--------3")
    a = plot_average_shap_value_of_group(ranked_data, group, selected_attributes, all_attributes_original, shap_values, ax)
    print("--------4: a: ")
    print(a)
    return a
    plt.xticks([0, 10, 20, 30], fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(r"student_shap_globalbounds.png", bbox_inches='tight')
    plt.show()
    return shap_values



def get_dis_plot(ranked_data, all_attributes, all_attributes_original, original_att, group, group_name, k):
    fig, ax = plt.subplots(1, 1,figsize=(14, 6))
    att = all_attributes[all_attributes_original.index(original_att)]
    att = att[:len(att)-2]
    plot_distribution_ratio(ranked_data, att, original_att, group, group_name, k, ax)
    # plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35], fontsize=FONTSIZE)
    fig.show()
    plt.savefig(r"student_value_dis_globalbounds.png", bbox_inches='tight')



def get_shaped_values(ranked_data, all_attributes):
    x = ranked_data[all_attributes]
    y = ranked_data['rank']
    # TODO: I chagned all_attributes_original to  all_attributes
    # print("type of all_attri: ",x)
    # print("type of all_attri_original: ", all_attributes_original)
    #todo don think we need it
    #x.set_axis(all_attributes, axis=1, inplace=True)

    # with sklearn
    model = LinearRegression()
    model.fit(x, y)
    print("Model coefficients:\n")
    for i in range(x.shape[1]):
        print(x.columns[i], "=", model.coef_[i].round(5))
    # compute the SHAP values for the linear model
    explainer = shap.Explainer(model.predict, x)
    shap_values = explainer(x)
    return shap_values



