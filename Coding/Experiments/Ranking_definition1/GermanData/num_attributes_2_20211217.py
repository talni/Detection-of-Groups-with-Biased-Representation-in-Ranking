import pandas as pd
from Algorithms import pattern_count
from Algorithms import IterTD_GlobalBounds as newalg
from Algorithms import NaiveAlgRanking_GlobalBounds as naivealg



import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


sns.set_palette("Paired")
# sns.set_palette("deep")
sns.set_context("poster", font_scale=2)
sns.set_style("whitegrid")
# sns.palplot(sns.color_palette("deep", 10))
# sns.palplot(sns.color_palette("Paired", 9))

line_style = ['o-', 's--', '^:', '-.p']
color = ['C0', 'C1', 'C2', 'C3', 'C4']
plt_title = ["BlueNile", "COMPAS", "Credit Card"]

label = ["GlobalBounds", "IterTD"]
line_width = 8
marker_size = 15
f_size = (14, 8)




def ComparePatternSets(set1, set2):
    len1 = len(set1)
    len2 = len(set2)
    if len1 != len2:
        return False
    for p in set1:
        found = False
        for q in set2:
            if newalg.PatternEqual(p, q):
                found = True
                break
        if found is False:
            return False
    return True

def thousands_formatter(x, pos):
    return int(x/1000)

def GridSearch(original_data, all_attributes, thc, Lowerbounds, number_attributes, time_limit, only_new_alg=False):

    selected_attributes = all_attributes[:number_attributes]
    print("{} attributes: {}".format(number_attributes, selected_attributes))

    less_attribute_data = original_data[selected_attributes]


    if only_new_alg:
        pattern_treated_unfairly_lowerbound1, num_patterns_visited1_, t1_ \
            = newalg.GraphTraverse(
            less_attribute_data, selected_attributes, thc,
            Lowerbounds,
            k_min, k_max, time_limit)

        print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
        print("time = {} s, num of pattern_treated_unfairly_lowerbound = {} ".format(
            t1_, len(pattern_treated_unfairly_lowerbound1)))

        return t1_, num_patterns_visited1_, 0, 0, pattern_treated_unfairly_lowerbound1

    pattern_treated_unfairly_lowerbound1, num_patterns_visited1_, t1_ \
        = newalg.GraphTraverse(
        less_attribute_data, selected_attributes, thc,
        Lowerbounds,
        k_min, k_max, time_limit)

    print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
    print("time = {} s, num of pattern_treated_unfairly_lowerbound = {} ".format(
        t1_, len(pattern_treated_unfairly_lowerbound1)))

    pattern_treated_unfairly_lowerbound2, \
    num_patterns_visited2_, t2_ = naivealg.NaiveAlg(less_attribute_data, selected_attributes, thc,
                                                    Lowerbounds,
                                                    k_min, k_max, time_limit)

    print("naive alg, num_patterns_visited = {}".format(num_patterns_visited2_))
    print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}".format(
            t2_, len(pattern_treated_unfairly_lowerbound2)))


    if t1_ > time_limit:
        raise Exception("new alg exceeds time limit")
    if t2_ > time_limit:
        raise Exception("naive alg exceeds time limit")

    for k in range(0, k_max - k_min):
        if ComparePatternSets(pattern_treated_unfairly_lowerbound1[k], pattern_treated_unfairly_lowerbound2[k]) is False:
            raise Exception("sanity check fails! k = {}".format(k + k_min))



    return t1_, num_patterns_visited1_, t2_, num_patterns_visited2_, \
           pattern_treated_unfairly_lowerbound2


all_attributes = ['StatusExistingAcc', 'DurationMonth_C', 'CreditHistory', 'Purpose', 'CreditAmount_C',
                  'SavingsAccount', 'EmploymentLength', 'InstallmentRate', 'MarriedNSex', 'Debtors',
                  'ResidenceLength', 'Property', 'Age_C', 'InstallmentPlans', 'Housing',
                  'ExistingCredit', 'Job', 'NumPeopleLiable', 'Telephone', 'ForeignWorker']

thc = 50

original_data_file = r"../../../../InputData/GermanCredit/GermanCredit_ranked.csv"

original_data = pd.read_csv(original_data_file)[all_attributes]



time_limit = 10*60



num_att_max_naive = 21
num_att_min = 3
num_att_max = 21
execution_time1 = list()
execution_time2 = list()
num_calculation1 = list()
num_calculation2 = list()
num_pattern_skipped_mis_c1 = list()
num_pattern_skipped_mis_c2 = list()
num_pattern_skipped_whole_c1 = list()
num_pattern_skipped_whole_c2 = list()
num_patterns_found_lowerbound = list()
patterns_found_lowerbound = list()
num_loops = 1


k_min = 10
k_max = 50
List_k = list(range(k_min, k_max))
Lowerbounds = [10] * 10 + [20] * 10 + [30] * 10 + [40] * 10


for number_attributes in range(num_att_min, num_att_max_naive):
    print("\n\nnumber of attributes = {}".format(number_attributes))

    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, calculation1_,  t2_, calculation2_, pattern_treated_unfairly_lowerbound = \
            GridSearch(original_data, all_attributes, thc, Lowerbounds, number_attributes, time_limit)
        t1 += t1_
        t2 += t2_
        calculation1 += calculation1_
        calculation2 += calculation2_
        if l == 0:
            patterns_found_lowerbound.append(pattern_treated_unfairly_lowerbound)
            num_patterns_found_lowerbound.append(len(pattern_treated_unfairly_lowerbound))
    t1 /= num_loops
    t2 /= num_loops
    calculation1 /= num_loops
    calculation2 /= num_loops

    execution_time1.append(t1)
    num_calculation1.append(calculation1)
    execution_time2.append(t2)
    num_calculation2.append(calculation2)



for number_attributes in range(num_att_max_naive, num_att_max):
    print("\n\nnumber of attributes = {}".format(number_attributes))
    t1, calculation1 = 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, calculation1_, t2_, calculation2_, pattern_treated_unfairly_lowerbound = \
            GridSearch(original_data, all_attributes, thc, Lowerbounds, number_attributes, time_limit, only_new_alg=True)
        t1 += t1_
        calculation1 += calculation1_
        if l == 0:
            patterns_found_lowerbound.append(pattern_treated_unfairly_lowerbound)
            num_patterns_found_lowerbound.append(len(pattern_treated_unfairly_lowerbound))

    t1 /= num_loops
    calculation1 /= num_loops

    execution_time1.append(t1)
    num_calculation1.append(calculation1)




output_path = r'../../../../OutputData/Ranking_definition1_1/GermanData/num_att_urb_german.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)


output_file.write("execution time\n")
for n in range(num_att_min, num_att_max_naive):
    output_file.write('{} {} {}\n'.format(n, execution_time1[n-num_att_min], execution_time2[n-num_att_min]))
for n in range(num_att_max_naive, num_att_max):
    output_file.write('{} {}\n'.format(n, execution_time1[n - num_att_min]))


output_file.write("\n\nnumber of patterns checked\n")
for n in range(num_att_min, num_att_max_naive):
    output_file.write('{} {} {}\n'.format(n, num_calculation1[n-num_att_min], num_calculation2[n-num_att_min]))
for n in range(num_att_max_naive, num_att_max):
    output_file.write('{} {}\n'.format(n, num_calculation1[n-num_att_min]))



output_file.write("\n\nnumber of patterns found lowebound\n")
for n in range(num_att_min, num_att_max):
    output_file.write('{} {} \n {}\n'.format(n, num_patterns_found_lowerbound[n-num_att_min],
                                             patterns_found_lowerbound[n-num_att_min]))



# when number of attributes = 8, naive algorithm running time > 10min
# so we only use x[:6]
x_new = list(range(num_att_min, num_att_max))
x_naive = list(range(num_att_min, num_att_max_naive))




fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(x_new, execution_time1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.plot(x_naive, execution_time2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
             markersize=marker_size)
plt.xlabel('Number of attributes')
plt.ylabel('Execution time (s)')
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/Ranking_definition1_1/GermanData/num_att_time_urb_german.png",
            bbox_inches='tight')
plt.show()
plt.close()





fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(x_new, num_calculation1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.plot(x_naive, num_calculation2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
             markersize=marker_size)
plt.xlabel('Number of attributes')
plt.ylabel('Number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/Ranking_definition1_1/GermanData/num_att_calculations_urb_german.png",
            bbox_inches='tight')
plt.show()
plt.close()


plt.clf()

