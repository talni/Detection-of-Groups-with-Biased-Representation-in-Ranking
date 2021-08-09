import pandas as pd
import sys
sys.path.append('\\../\\../Coding')
import time
from Coding.Algorithms import NewAlg_1_20210529 as newalg



"""
cox['sex'].replace(to_replace=['Male', 'Female'], value=[0, 1], inplace=True)


cox['age_cat'].replace(to_replace=['Less than 25', '25 - 45', 'Greater than 45'], value=[0, 1, 2], inplace=True)

cox['race'].replace(to_replace=['African-American', 'Asian', 'Caucasian', 'Hispanic', 'Native American', 'Other'], value=[0, 1, 2, 3, 4, 5], inplace=True)


"""

selected_attributes = ["sex", "age_cat", "race"]



original_data_file = r"../../InputData/COMPAS_ProPublica/compas-analysis-master/cox-parsed/cox-parsed_7214rows_cat.csv"
mis_data_file = r"../../InputData/COMPAS_ProPublica/compas-analysis-master/cox-parsed/cox-parsed_mis_cat.csv"


output_path = r'acc_1.txt'
output_file = open(output_path, "w")

output_file.write("selected_attributes: {}\n".format(selected_attributes))


def read_with_att(original_data_file, selected_attributes):
    original_data = pd.read_csv(original_data_file)
    less_attribute_data = original_data[selected_attributes]
    return less_attribute_data


less_attribute_data = read_with_att(original_data_file, selected_attributes)
mis_class_data = read_with_att(mis_data_file, selected_attributes)



tha = 0.05
thc = 1
time_limit = 5 * 60




print("tha = {}, thc = {}".format(tha, thc))
pattern_with_low_accuracy, num_calculation, t_ = newalg.GraphTraverse(less_attribute_data,
                                                                              mis_class_data, tha,
                                                                              thc, time_limit)
print("time = {} s, num_calculation = {}".format(t_, num_calculation))
print("find {} patterns".format(len(pattern_with_low_accuracy)))
print(pattern_with_low_accuracy)



for p in pattern_with_low_accuracy:
    print(p)



output_file.write("newalg, time = {} s, num_calculation = {}\n".format(t_, num_calculation))
output_file.write("num of patterns detected = {}\n".format(len(pattern_with_low_accuracy)))
for p in pattern_with_low_accuracy:
    output_file.write(str(p))
    output_file.write("\n")



