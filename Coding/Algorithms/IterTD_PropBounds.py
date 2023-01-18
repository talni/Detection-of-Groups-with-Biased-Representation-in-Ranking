"""
full copy from NewAlgRanking_definition2_15.py
examples at the end
"""

import time
import math
import pandas as pd
from Coding.Algorithms import pattern_count
from sortedcontainers import SortedDict
import logging


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


def P1DominatedByP2ForStr(str1, str2, num_att):
    if str1 == str2:
        return True
    num_separator = num_att - 1
    start_pos1 = 0
    start_pos2 = 0
    for i in range(num_separator):
        p1 = str1.find("|", start_pos1)
        p2 = str2.find("|", start_pos2)
        s1 = str1[start_pos1:p1]
        s2 = str2[start_pos2:p2]
        if s1 != s2 and s2 != '':
            return False
        start_pos1 = p1 + 1
        start_pos2 = p2 + 1
    s1 = str1[start_pos1:]
    s2 = str2[start_pos2:]
    if s1 != s2 and s2 != '':
        return False
    return True


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


def PatternEqual(m, P):
    length = len(P)
    if len(m) != length:
        return False
    for i in range(length):
        if m[i] != P[i]:
            return False
    return True


# for a tuple, there is one parent generating it
# but it has more than one parent, and those who doesn't generate this tuple, their size also increase
def GenerateChildrenRelatedToTuple(P, new_tuple):
    children = []
    length = len(P)
    i = 0
    for i in range(length - 1, -1, -1):
        if P[i] != -1:
            break
    if P[i] == -1:
        i -= 1
    for j in range(i + 1, length, 1):
        s = P.copy()
        s[j] = new_tuple[j]
        children.append(s)
    return children


def GenerateUnrelatedChildren(P, whole_data_frame, ranked_data, attributes, new_tuple):
    children = []
    length = len(P)
    i = 0
    for i in range(length - 1, -1, -1):
        if P[i] != -1:
            break
    if P[i] == -1:
        i -= 1
    for j in range(i + 1, length, 1):
        if P[j] == -1:
            all_values = ranked_data[attributes[j]].unique()
            for a in all_values:
            # for a in range(int(whole_data_frame[attributes[j]]['min']),
            #                int(whole_data_frame[attributes[j]]['max']) + 1):
                if a == new_tuple[j]:
                    continue
                s = P.copy()
                s[j] = a
                children.append(s)
    return children

#
# def GenerateDominatedGroup(P, whole_data_frame, ranked_data, attributes, smallest_valid_k, k, smallest_valid_k_ancestor, K_values):
#     children = []
#     length = len(P)
#     i = 0
#     for i in range(length - 1, -1, -1):
#         if P[i] != -1:
#             break
#     if P[i] == -1:
#         i -= 1
#     for j in range(i + 1, length, 1):
#         all_values = ranked_data[attributes[j]].unique()
#         for a in all_values:
#         # for a in range(int(whole_data_frame[attributes[j]]['min']), int(whole_data_frame[attributes[j]]['max']) + 1):
#             s = P.copy()
#             s[j] = a
#             children.append(s)
#     if smallest_valid_k_ancestor > smallest_valid_k:
#         K_values = K_values + [smallest_valid_k] * len(children)
#     else:
#         K_values = K_values + [smallest_valid_k_ancestor] * len(children)
#
#     for j in range(0, i):
#         if P[i] == -1:
#             all_values = ranked_data[attributes[j]].unique()
#             for a in all_values:
#             # for a in range(int(whole_data_frame[attributes[j]]['min']),
#             #                int(whole_data_frame[attributes[j]]['max']) + 1):
#                 s = P.copy()
#                 s[j] = a
#                 children.append(s)
#                 K_values.append(k + 1)
#     return children, K_values


def GenerateChildren(P, whole_data_frame, ranked_data, attributes):
    children = []
    length = len(P)
    i = 0
    for i in range(length - 1, -1, -1):
        if P[i] != -1:
            break
    if P[i] == -1:
        i -= 1
    for j in range(i + 1, length, 1):
        all_values = ranked_data[attributes[j]].unique()
        for a in all_values:
            s = P.copy()
            s[j] = a
            children.append(s)
        # for a in range(int(whole_data_frame[attributes[j]]['min']), int(whole_data_frame[attributes[j]]['max']) + 1):
        #     s = P.copy()
        #     s[j] = a
        #     children.append(s)
    return children


# def GenerateChildrenAndChildrenRelatedToNewTuple(P, whole_data_frame, ranked_data, attributes, new_tuple):
#     children = []
#     children_related_to_new_tuple = []
#     length = len(P)
#     i = 0
#     for i in range(length - 1, -1, -1):
#         if P[i] != -1:
#             break
#     if P[i] == -1:
#         i -= 1
#     for j in range(i + 1, length, 1):
#         all_values = ranked_data[attributes[j]].unique()
#         for a in all_values:
#             s = P.copy()
#             s[j] = a
#             children.append(s)
#             if s[j] == new_tuple[j]:
#                 children_related_to_new_tuple.append(s)
#         # for a in range(int(whole_data_frame[attributes[j]]['min']), int(whole_data_frame[attributes[j]]['max']) + 1):
#         #     s = P.copy()
#         #     s[j] = a
#         #     children.append(s)
#         #     if s[j] == new_tuple[j]:
#         #         children_related_to_new_tuple.append(s)
#     return children, children_related_to_new_tuple


def num2string(pattern):
    st = ''
    for i in pattern:
        if i != -1:
            st += str(i)
        st += '|'
    st = st[:-1]
    return st


def string2num(st):
    p = list()
    idx = 0
    item = ''
    i = ''
    for i in st:
        if i == '|':
            if item == '':
                p.append(-1)
            else:
                p.append(int(item))
                item = ''
            idx += 1
        else:
            item += i
    if i != '|':
        p.append(int(item))
    else:
        p.append(-1)
    return p


# whether a pattern P is dominated by MUP M
# except from P itself
def PDominatedByM(P, M):
    for m in M:
        if PatternEqual(m, P):
            continue
        if P1DominatedByP2(P, m):
            return True, m
    return False, None


"""
whole_data: the original data file 
mis_class_data: file containing mis-classified tuples
Tha: delta fairness value 
Thc: size threshold
"""


def findParent(child, length):
    parent = child.copy()
    for i in range(length - 1, -1, -1):
        if parent[i] != -1:
            parent[i] = -1
            break
    return parent


# find parent when string doesn't have ' '
def findParentForStr(child):
    end = 0
    length = len(child)
    i = length - 1
    while i > -1:
        if child[i] != '|':
            end = i + 1
            i -= 1
            break
        i -= 1
    while i > -1:
        if child[i] == '|':
            start = i
            parent = child[:start + 1] + child[end:]
            return parent
        i -= 1
    parent = child[end:]
    return parent


# closest ancestor must be the ancestor with smallest k value
# smallest_ancestor != "" if and only if there is an ancestor having the same k value
class Node:
    # init method or constructor
    def __init__(self, pattern, st, smallest_valid_k):
        self.pattern = pattern
        self.st = st
        self.smallest_valid_k = smallest_valid_k
        # string of ancestor with smallest k, "" means itself
        # in case of same k, smallest_ancestor points to the ancestor rather than the node itself
        # since when we reach that k, all these nodes need updating
        # self.smallest_ancestor = smallest_ancestor
        # whether this node has the smallest k in the path from the root
        # self.self_smallest_k = self_smallest_k  # must be true. It may have children with smaller k but it doesn't know


# find the closest ancestor of pattern p in nodes_dict
# by checking each of p's ancestor in nodes_dict
def Find_closest_ancestor(string_set, st, num_att):
    if st in string_set:
        return True, st
    original_st = st
    length = len(st)
    j = length - 1
    i = length - 1
    find = False
    while True:
        if i < 0:
            if find:
                parent_str = st[j + 1:]
                if parent_str in string_set:
                    return True, parent_str
                else:
                    return False, original_st
            else:
                return False, original_st
        if find is False and st[i] == "|":
            i -= 1
            j -= 1
            continue
        elif find is False and st[i] != "|":
            j = i
            find = True
            i -= 1
            continue
        elif find and st[i] != "|":
            i -= 1
            continue
        else:
            parent_str = st[:i + 1] + st[j + 1:]
            if parent_str in string_set:
                return True, parent_str
            else:
                st = parent_str
                j = i - 1
                i -= 1
                continue
    return False, original_st


# assumption: p is not in nodes_dict, and we don't know its ancestor
# in this function, we find the ancestor with the smallest k for pattern p
# and only add p if p has a smaller k
# if the k is same, don't add p
# this function is executed during a top-down search, so p's descendants are not in nodes_dict
def Add_node_to_set(nodes_dict, k_dict, smallest_valid_k, p, st, num_att):
    att = 0
    end = 0
    length = len(st)
    i = length - 1
    original_st = st
    while i > -1:
        if st[i] != '|':
            end = i + 1
            i -= 1
            break
        if st[i] == '|':
            att += 1
        i -= 1
    while att < num_att:
        while i > -1:
            if st[i] == '|':
                start = i
                parent = st[:start + 1] + st[end:]
                if parent in nodes_dict.keys():
                    if nodes_dict[parent].smallest_valid_k > smallest_valid_k:
                        nodes_dict[original_st] = Node(p, original_st, smallest_valid_k)
                        k_dict[smallest_valid_k].add(original_st)
                        return
                    else:  # smallest_valid_k is larger than the k value of an ancestor
                        return
                st = parent
                i -= 1
                break
            i -= 1
        att += 1
    # no ancestors in nodes_dict
    nodes_dict[original_st] = Node(p, original_st, smallest_valid_k)
    k_dict[smallest_valid_k].add(original_st)


def Check_k_related_patterns_in_dominated_by_results(p, st, whole_data_frame, attributes, nodes_dict, k_dict,
                                                     patterns_size_whole, pc_whole_data, patterns_top_k, data_size,
                                                     new_tuple, dominated_by_result, num_att, k, Thc,
                                                     alpha):
    S = GenerateChildrenRelatedToTuple(p, new_tuple)
    while len(S) > 0:
        p = S.pop(0)
        st = num2string(p)
        if st in patterns_size_whole:
            whole_cardinality = patterns_size_whole[st]
        else:
            whole_cardinality = pc_whole_data.pattern_count(st)
        if whole_cardinality < Thc:
            continue
        if st in dominated_by_result:
            # num_top_k = patterns_top_k.pattern_count(st)
            # lowerbound = alpha * whole_cardinality * k / data_size
            # if num_top_k >= lowerbound:
            dominated_by_result.remove(st)
            Check_and_remove_a_larger_k(nodes_dict, k_dict, p, st)
        else:
            Check_and_remove_a_larger_k(nodes_dict, k_dict, p, st)
            if p[num_att - 1] == -1:
                children = GenerateChildrenRelatedToTuple(p, new_tuple)
                S = S + children


"""
check the k for children unrelated with new tuple, add to nodes_dict is they are smaller than ancestor k
stop when a node is in nodes_dict
"""


def Check_k_with_non_related_patterns(nodes_dict, k_dict, smallest_valid_k_ancestor, p, st, whole_data_frame,
                                      attributes,
                                      patterns_size_whole, pc_whole_data, patterns_top_k, data_size, new_tuple,
                                      result_set, dominated_by_result, num_att, k, Thc, alpha, k_max, k_min, ranked_data):
    S = GenerateUnrelatedChildren(p, whole_data_frame, ranked_data, attributes, new_tuple)
    K_values = [smallest_valid_k_ancestor] * len(S)
    while len(S) > 0:
        p = S.pop(0)
        st = num2string(p)
        smallest_valid_k_ancestor = K_values.pop(0)
        if st in patterns_size_whole:
            whole_cardinality = patterns_size_whole[st]
        else:
            whole_cardinality = pc_whole_data.pattern_count(st)
        if whole_cardinality < Thc:
            continue
        num_top_k = patterns_top_k.pattern_count(st)
        lowerbound = alpha * whole_cardinality * k / data_size
        if num_top_k < lowerbound:
            CheckDominationAndAddForLowerbound(st, result_set, dominated_by_result, num_att)
            smallest_valid_k = math.floor(num_top_k * data_size / (alpha * whole_cardinality))
            if smallest_valid_k > k_max:
                smallest_valid_k = k_max + 1
            elif smallest_valid_k < k_min:
                smallest_valid_k = k_min - 1
            if smallest_valid_k < smallest_valid_k_ancestor:
                Update_or_add_node_w_smaller_k(nodes_dict, k_dict, smallest_valid_k, p, st)
                continue
        smallest_valid_k = math.floor(num_top_k * data_size / (alpha * whole_cardinality))
        if smallest_valid_k > k_max:
            smallest_valid_k = k_max + 1
        elif smallest_valid_k < k_min:
            smallest_valid_k = k_min - 1
        if smallest_valid_k < smallest_valid_k_ancestor:
            Update_or_add_node_w_smaller_k(nodes_dict, k_dict, smallest_valid_k, p, st)
            children = GenerateChildren(p, whole_data_frame, ranked_data, attributes)
            S += children
            K_values += [smallest_valid_k] * len(children)
        else:
            if st in nodes_dict.keys():
                old_k = nodes_dict[st].smallest_valid_k
                nodes_dict.pop(st)
                k_dict[old_k].remove(st)
                if st in dominated_by_result:
                    dominated_by_result.remove(st)
            children = GenerateChildren(p, whole_data_frame, ranked_data, attributes)
            S += children
            K_values += [smallest_valid_k_ancestor] * len(children)
    return




def Update_or_add_node_w_smaller_k(nodes_dict, k_dict, smallest_valid_k, p, st):
    if st in nodes_dict.keys():
        k_dict[nodes_dict[st].smallest_valid_k].remove(st)
        k_dict[smallest_valid_k].add(st)
        nodes_dict[st].smallest_valid_k = smallest_valid_k
    else:
        k_dict[smallest_valid_k].add(st)
        nodes_dict[st] = Node(p, st, smallest_valid_k)


def Check_and_remove_a_larger_k(nodes_dict, k_dict, p, st):
    if st in nodes_dict.keys():
        old_k = nodes_dict[st].smallest_valid_k
        nodes_dict.pop(st)
        k_dict[old_k].remove(st)


# whether a is an ancestor of b, a and b are string
def A_is_ancestor_of_B(a, b):
    if len(a) >= len(b):
        return False
    length = len(a)  # len(b) should >= len(a)
    i = 0
    for i in range(length):
        if a[i] != b[i]:
            if a[i] != "|":
                return False
            else:
                break
    for j in range(i, length):
        if a[j] != "|":
            return False
    return True


# def PatternInSet(p, set):
#     if isinstance(p, str):
#         p = string2num(p)
#     for q in set:
#         if PatternEqual(p, q):
#             return True
#     return False


def AddDominatedToLowerbound(pattern, pattern_treated_unfairly, dominated_by_result):
    to_remove = []
    for p in pattern_treated_unfairly:
        # if PatternEqual(p, pattern):
        #     return
        if P1DominatedByP2(pattern, p):
            return False
        elif P1DominatedByP2(p, pattern):
            to_remove.append(p)
    for p in to_remove:
        pattern_treated_unfairly.remove(p)
    pattern_treated_unfairly.append(pattern)
    return True


#
# # return whether it is added or not, patterns are stored as list
# def CheckDominationAndAddForLowerbound(pattern, pattern_treated_unfairly, dominated_by_result):
#     to_remove = []
#     for p in pattern_treated_unfairly:
#         # if PatternEqual(p, pattern):
#         #     return
#         if P1DominatedByP2(pattern, p):
#             if pattern not in dominated_by_result:
#                 dominated_by_result.append(pattern)
#             return False
#         elif P1DominatedByP2(p, pattern):
#             to_remove.append(p)
#             if p not in dominated_by_result:
#                 dominated_by_result.append(p)
#     for p in to_remove:
#         pattern_treated_unfairly.remove(p)
#     if pattern in dominated_by_result:
#         dominated_by_result.remove(pattern)
#     pattern_treated_unfairly.append(pattern)
#     return True
#

# p is added to result set or dominated set
# we need to remove its children from dominated set
def Remove_children_from_dominated(st, p, result_set, dominated_by_result, num_att, whole_data_frame, attributes,
                                   patterns_size_whole, pc_whole_data, patterns_top_k, alpha, k, data_size, Thc, ranked_data):
    S = GenerateChildren(p, whole_data_frame, ranked_data, attributes)
    while len(S) > 0:
        p = S.pop(0)
        st = num2string(p)
        if st in patterns_size_whole:
            whole_cardinality = patterns_size_whole[st]
        else:
            whole_cardinality = pc_whole_data.pattern_count(st)
        if whole_cardinality < Thc:
            continue
        if st in dominated_by_result:
            dominated_by_result.remove(st)
        else:
            children = GenerateChildren(p, whole_data_frame, ranked_data, attributes)
            S += children


# return whether it is added or not, strings are stored rather than patterns
def CheckDominationAndAddForLowerbound(pattern_st, pattern_treated_unfairly, dominated_by_result, num_att):
    to_remove = []
    for st in pattern_treated_unfairly:
        # if PatternEqual(p, pattern):
        #     return
        if P1DominatedByP2ForStr(pattern_st, st, num_att):
            if pattern_st not in dominated_by_result:
                dominated_by_result.add(pattern_st)
            return False
        elif P1DominatedByP2ForStr(st, pattern_st, num_att):
            to_remove.append(st)
            if st not in dominated_by_result:
                dominated_by_result.add(st)
    for st in to_remove:
        pattern_treated_unfairly.remove(st)
    if pattern_st in dominated_by_result:
        dominated_by_result.remove(pattern_st)
    pattern_treated_unfairly.add(pattern_st)
    return True


def Remove_descendants_str(c_str, patterns_to_search_lowest_level):
    to_remove = set()
    for st in patterns_to_search_lowest_level:
        if A_is_ancestor_of_B(c_str, st):
            to_remove.add(st)
    for r in to_remove:
        patterns_to_search_lowest_level.remove(r)


# Stop set: doesn't allow ancestor and children, but allow dominance between others.
# We can only put a node itself into the stop set, whether it is in result set or not.
# Nodes in stop set:
# 1. size is too small
# 2. already in result set
# 3. others.
def GraphTraverse(ranked_data, attributes, Thc, alpha, k_min, k_max, time_limit):
    time0 = time.time()
    data_size = len(ranked_data)
    pc_whole_data = pattern_count.PatternCounter(ranked_data, encoded=False)
    pc_whole_data.parse_data()
    whole_data_frame = ranked_data.describe(include='all')
    num_patterns_visited = 0
    num_att = len(attributes)
    root = [-1] * num_att
    S = GenerateChildren(root, whole_data_frame, ranked_data, attributes)
    root_str = '|' * (num_att - 1)
    store_children = {root_str: S}
    pattern_treated_unfairly = []  # looking for the most general patterns
    patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k_min], encoded=False)
    patterns_top_kmin.parse_data()
    patterns_size_whole = dict()
    k_dict = dict()
    dominated_by_result = set()

    """
    to get pattern in top k for the purpose of demo:
    """
    patterns_size_topk = dict()
    patterns_size_topk[k_min] = patterns_top_kmin

    # this dict stores all patterns, indexed by num2string(p)
    nodes_dict = SortedDict()
    time_setup1 = 0
    time_Add_node_to_set = 0
    # DFS
    # this part is the main time consumption

    result_set = set()
    for k in range(0, k_max + 2):
        k_dict[k] = set()
    k = k_min
    while len(S) > 0:
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        time1 = time.time()
        P = S.pop(0)
        st = num2string(P)
        num_patterns_visited += 1
        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        time2 = time.time()
        time_setup1 += time2 - time1
        if whole_cardinality < Thc:
            continue
        num_top_k = patterns_top_kmin.pattern_count(st)
        smallest_valid_k = math.floor(num_top_k * data_size / (alpha * whole_cardinality))
        if smallest_valid_k > k_max:
            smallest_valid_k = k_max + 1
        elif smallest_valid_k < k_min:
            smallest_valid_k = k_min - 1
        # lowerbound = (whole_cardinality / data_size - alpha) * k
        lowerbound = alpha * whole_cardinality * k / data_size
        # print("pattern {}, lb = {}, smallest_valid_k = {}".format(P, lowerbound, smallest_valid_k))
        if num_top_k < lowerbound:
            CheckDominationAndAddForLowerbound(st, result_set, dominated_by_result, num_att)
            # Add_node_to_set(nodes_dict, k_dict, smallest_valid_k, P, st, num_att)
        else:
            if P[num_att - 1] == -1:
                if st in store_children:
                    children = store_children[st]
                else:
                    children = GenerateChildren(P, whole_data_frame, ranked_data, attributes)
                    store_children[st] = children
                S = children + S
            # maintain sets for k values only for a node not in result set.
            # so now we add this node to nodes_dict
            # smallest k before which lower bound is ok
            if st not in nodes_dict.keys():
                time11 = time.time()
                Add_node_to_set(nodes_dict, k_dict, smallest_valid_k, P, st, num_att)
                time12 = time.time()
                time_Add_node_to_set += time12 - time11
            else:
                raise Exception("st is impossible to be in nodes_dict.keys()")
    time1 = time.time()
    pattern_treated_unfairly.append(result_set)

    for k in range(k_min + 1, k_max):
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        time1 = time.time()
        patterns_top_k = pattern_count.PatternCounter(ranked_data[:k], encoded=False)
        patterns_top_k.parse_data()
        patterns_size_topk[k] = patterns_top_k
        new_tuple = ranked_data.iloc[[k - 1]].values.flatten().tolist()
        # print("k={}, new tuple = {}".format(k, new_tuple))
        # print("dominated_by_result: ", dominated_by_result)
        # top down for related patterns, using similar methods as k_min, add to result set if needed
        # ancestors are patterns checked in Add_new_tuple() function, to avoid checking them again
        result_set = set(pattern_treated_unfairly[k - 1 - k_min])
        ancestors, num_patterns_visited = Add_new_tuple(new_tuple, Thc,
                                                        whole_data_frame, patterns_top_k, k, k_min, k_max,
                                                        pc_whole_data,
                                                        num_patterns_visited,
                                                        patterns_size_whole, alpha, num_att,
                                                        data_size, nodes_dict, k_dict, attributes,
                                                        result_set, dominated_by_result, store_children, ranked_data)
        to_added_to_dominated_by_result = set()
        to_remove_from_dominated_by_result = set()
        for d in dominated_by_result:
            to_remove_from_result_set = set()
            d_dominated_by_result_set = False
            for st in result_set:
                if P1DominatedByP2ForStr(d, st, num_att):
                    d_dominated_by_result_set = True
                    break
                elif P1DominatedByP2ForStr(st, d, num_att):
                    to_remove_from_result_set.add(st)
                    to_added_to_dominated_by_result.add(st)
            if not d_dominated_by_result_set:
                for p in to_remove_from_result_set:
                    result_set.remove(p)
                result_set.add(d)
                to_remove_from_dominated_by_result.add(d)
        for d in to_remove_from_dominated_by_result:
            if d in to_added_to_dominated_by_result:
                to_added_to_dominated_by_result.remove(d)
            else:
                dominated_by_result.remove(d)
        for d in to_added_to_dominated_by_result:
            dominated_by_result.add(d)
        for st in k_dict[k - 1]:
            if st in ancestors:
                continue
            if st in result_set:
                continue
            CheckDominationAndAddForLowerbound(st, result_set, dominated_by_result, num_att)
            if st in dominated_by_result:
                Remove_children_from_dominated(st, string2list(st), result_set, dominated_by_result, num_att,
                                               whole_data_frame, attributes,
                                               patterns_size_whole, pc_whole_data, patterns_top_k, alpha, k, data_size,
                                               Thc, ranked_data)
        pattern_treated_unfairly.append(result_set)
    time1 = time.time()
    """
    to get pattern in top k for the purpose of demo:
    when the string format of a pattern is st, then size of st in top k is 
    size = patterns_size_topk[k].pattern_count(st)
    """
    return pattern_treated_unfairly, num_patterns_visited, time1 - time0, pc_whole_data, patterns_size_topk


# search top-down to go over all patterns related to new_tuple
# using similar checking methods as k_min
# add to result set if they are outliers
# need to update k values for these patterns

# for lower bound, when k and s_k increase by 1 at the same time, Sk/k is still larger than Sd(p)/Sd*alpha
# So if a pattern is above lower bound, after this, it is still above lower bound
# No patterns will be added to result set for lower bound here. Some will be added to upper bound result set
# but the smallest k values may maintain unchanged, may also increase
# thus, for lower bound, we only need to check for k values;
# for upper bound, we only need to check whether it is above upper bound
def Add_new_tuple(new_tuple, Thc, whole_data_frame, patterns_top_k, k, k_min, k_max, pc_whole_data,
                  num_patterns_visited, patterns_size_whole, alpha, num_att, data_size, nodes_dict, k_dict, attributes,
                  result_set, dominated_by_result, store_children, ranked_data):
    ancestors = []
    root = [-1] * num_att
    S = GenerateChildrenRelatedToTuple(root, new_tuple)  # pattern with one deterministic attribute
    # if the k values increases, go to function () without generating children
    # otherwise, generating children and add children to queue
    K_values = [k_max] * len(S)
    while len(S) > 0:
        P = S.pop(0)
        st = num2string(P)
        smallest_valid_k_ancestor = K_values.pop(0)
        ancestors.append(P)
        num_patterns_visited += 1
        if st in patterns_size_whole:
            whole_cardinality = patterns_size_whole[st]
        else:
            whole_cardinality = pc_whole_data.pattern_count(st)
        if whole_cardinality < Thc:
            continue
        # special case: this pattern itself is in the result set
        num_top_k = patterns_top_k.pattern_count(st)
        lowerbound = alpha * whole_cardinality * k / data_size
        # lowerbound = (whole_cardinality / data_size - alpha) * k
        smallest_valid_k = math.floor(num_top_k * data_size / (alpha * whole_cardinality))
        if st in nodes_dict:
            old_k = nodes_dict[st].smallest_valid_k
        else:
            old_k = 0
        if smallest_valid_k > k_max:
            smallest_valid_k = k_max + 1
        elif smallest_valid_k < k_min:
            smallest_valid_k = k_min - 1
        if num_top_k < lowerbound:
            if st in nodes_dict.keys():
                k_dict[nodes_dict[st].smallest_valid_k].remove(st)
                nodes_dict.pop(st)
            if st in result_set:
                Check_k_related_patterns_in_dominated_by_results(P, st, whole_data_frame,
                                                                 attributes, nodes_dict, k_dict,
                                                                 patterns_size_whole, pc_whole_data, patterns_top_k,
                                                                 data_size,
                                                                 new_tuple, dominated_by_result,
                                                                 num_att, k, Thc, alpha)
                continue
            CheckDominationAndAddForLowerbound(st, result_set, dominated_by_result, num_att)
            Check_k_related_patterns_in_dominated_by_results(P, st, whole_data_frame,
                                                             attributes, nodes_dict, k_dict,
                                                             patterns_size_whole, pc_whole_data, patterns_top_k,
                                                             data_size,
                                                             new_tuple, dominated_by_result,
                                                             num_att, k, Thc, alpha)
            continue
        smaller_k = smallest_valid_k
        if old_k < smallest_valid_k:  # k increases
            if smallest_valid_k_ancestor > smallest_valid_k:
                Update_or_add_node_w_smaller_k(nodes_dict, k_dict, smallest_valid_k, P, st)
            else:
                smaller_k = smallest_valid_k_ancestor
                Check_and_remove_a_larger_k(nodes_dict, k_dict, P, st)
            if st in result_set:
                result_set.remove(st)
            if st in dominated_by_result:
                dominated_by_result.remove(st)
            Check_k_with_non_related_patterns(nodes_dict, k_dict, smaller_k, P, st, whole_data_frame, attributes,
                                              patterns_size_whole, pc_whole_data, patterns_top_k, data_size,
                                              new_tuple, result_set, dominated_by_result, num_att, k, Thc, alpha,
                                              k_max, k_min, ranked_data)
        else:  # k doesn't change
            if smallest_valid_k_ancestor > smallest_valid_k:
                Update_or_add_node_w_smaller_k(nodes_dict, k_dict, smallest_valid_k, P, st)
            else:
                smaller_k = smallest_valid_k_ancestor
                Check_and_remove_a_larger_k(nodes_dict, k_dict, P, st)
            if st in result_set:
                result_set.remove(st)
            if st in dominated_by_result:
                dominated_by_result.remove(st)
        if P[num_att - 1] == -1:
            children = GenerateChildrenRelatedToTuple(P, new_tuple)
            S = S + children
            K_values += [smaller_k] * len(children)
    return ancestors, num_patterns_visited




############################################ Example ##################################################


#
# all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
#                       'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
#                       'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
#                       'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
#                       'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']
#
# all_attributes_original = ['student\'s school', 'student\'s sex', 'student\'s age',
#                   'student\'s home address type', 'family size', 'parent\'s cohabitation status',
#                   'mother\'s education', 'father\'s education', 'mother\'s job',
#                   'father\'s job', 'reason to choose this school', 'student\'s guardian',
#                   'home to school travel time', 'weekly study time', 'number of past class failures',
#                   'extra educational support', 'family educational support', 'extra paid classes within the course subject',
#                   'extra-curricular activities', 'attended nursery school', 'wants to take higher education',
#                   'Internet access at home', 'with a romantic relationship', 'quality of family relationships',
#                   'free time after school', 'going out with friends', 'workday alcohol consumption',
#                   'weekend alcohol consumption', 'current health status', 'number of school absences',
#                   'first period grade', 'second period grade', 'final grade']
#
#
# original_data_file = r"../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"
#
# ranked_data = pd.read_csv(original_data_file, index_col=False)
#
# selected_attributes = all_attributes[:6]
#
# alpha = 0.8
# k = 49
# k_min = k
# k_max = k
# thc = 50
# time_limit = 5*60
# result_prop_bounds, num_patterns_visited1_, t1_ \
#             = GraphTraverse(
#             ranked_data[selected_attributes], selected_attributes, thc,
#             alpha,
#             k_min, k_max, time_limit)
# print(result_prop_bounds)
# groups_prop_bounds = result_prop_bounds[0]
# print(groups_prop_bounds)
# print([string2list(s) for s in groups_prop_bounds])
#
