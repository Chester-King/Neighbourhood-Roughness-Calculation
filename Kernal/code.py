import pandas as pd
import numpy as np
import sys
from collections import defaultdict
import os

Data = pd.read_csv("D:\\Work\\Neibhourhood roughness set\\voilin\\Kernal\\Database.csv", header=None, prefix="V")
k = len(list(Data['V6'].unique()))  # counting the number of classes in the last column

Data = Data.drop(Data.columns[[6]], axis=1)  # removing last column
Data.head(10)
neighborhood_class_dict = {}
threshold = 5
for index, row in Data.iterrows():
    neighbor_list = []
    for index1, row1 in Data.iterrows():
        if ((row['V0'] > row1['V0'])):
            if (row['V0'] - row1['V0'] <= threshold):
                neighbor_list.append(index1)
        elif (row['V0'] < row1['V0']):
            if (row1['V0'] - row['V0'] <= threshold):
                neighbor_list.append(index1)
        else:
            neighbor_list.append(index1)
    neighborhood_class_dict[index] = neighbor_list
# print(neighborhood_class_dict)
for index in neighborhood_class_dict:
    print("key=", index)
    print(neighborhood_class_dict[index])
cluster2_numerical_global = []  # global variable


def MMeR(U, k):
    C = {}  # the clusters made through MMeR alogrithm
    C_distance = {}
    Rough_Data = U.copy()
    index = 0

    while (2 > len(C.keys())):

        split_attr, min_roughness_attr_value = mmer_roughness(Rough_Data)
        print(split_attr)
        # min_roughness_attr_value = min(roughness_list_dict[split_attr]) #find the min. value of min_roughness_attr for which alpha is minimum
        # print(min_roughness_attr_value)
        if (split_attr == 'V0'):
            cluster1 = cluster1_numerical_global
            # print("split-cluster1",cluster1 )
            cluster2 = cluster2_numerical_global
            # print("split-cluster2",cluster2 )
        else:
            cluster1 = Rough_Data[Rough_Data[split_attr] == min_roughness_attr_value].index.tolist()
            cluster2 = Rough_Data.loc[Rough_Data[split_attr] != min_roughness_attr_value].index.tolist()

        cluster1_distance = cluster_distance(cluster1, Data)  # calculating the cluster distance of cluster1
        cluster2_distance = cluster_distance(cluster2, Data)  # calculating the cluster distance of cluster2
        C[index] = cluster1
        C_distance[index] = cluster1_distance
        index += 1
        C[index] = cluster2
        C_distance[index] = cluster2_distance
        index += 1
        max_index = max(C_distance, key=C_distance.get)  # find the index with the max cluster distance

        Rough_Data = U.ix[C[max_index]]  # send data of cluster with max distance
        for key in C:
            print("Cluster distance", C_distance[key])
            print("Cluster", C[key])

        if (len(C.keys()) != (k - 1)):
            del C[max_index]
            del C_distance[max_index]

        print("C length:", len(C.keys()))


def roughness(col_name1, col_name2, val, Data):
    equiv_class_dict = {}  # dictionary storing equiv classes for all attribute
    for col in Data:
        equiv_class_dict[col] = list(Data.groupby([col]))

    arr = equiv_class_dict[col_name2]  # group equiv classes according to column2
    target_set = Data[Data[col_name1] == val].index.tolist()  # calculating target set for each unique value in column1
    lower_approx = []
    upper_approx = []
    for name, group in arr:
        selected_list = list(group.index)
        if (set(selected_list).issubset(target_set)):  # if a subset, then append to lower_approx
            lower_approx.append(selected_list)
        if (set(selected_list) & set(target_set)):  # if intersection is not null, then append to upper_approx
            upper_approx.append(selected_list)
    lower_approx_count = sum(map(len, lower_approx))  # count the no. of elements in lower_approx
    upper_approx_count = sum(map(len, upper_approx))
    if (lower_approx_count == 0):
        roughness = 1
    else:
        roughness = 1 - (lower_approx_count / upper_approx_count)  # return roughness for a(i) = alpha
    return roughness


def roughness_categorical_to_numerical(neighborhood_class_dict, col_name1, col_name2, val, Data):
    target_set = Data[Data[col_name1] == val].index.tolist()  # calculating target set for each unique value in column1
    lower_approx1 = []
    upper_approx1 = []
    for k in neighborhood_class_dict:
        selected_list = neighborhood_class_dict[k]
        if (set(selected_list).issubset(target_set)):  # if a subset, then append to lower_approx
            lower_approx1.append(k)
        if (set(selected_list) & set(target_set)):  # if intersection is not null, then append to upper_approx
            upper_approx1.append(k)
    lower_approx_count = len(lower_approx1)  # count the no. of elements in lower_approx
    upper_approx_count = len(upper_approx1)
    if (lower_approx_count == 0):
        roughness = 1
    else:
        roughness = 1 - (lower_approx_count / upper_approx_count)  # return roughness for a(i) = alpha
    return roughness


def roughness_numerical_to_categorical(neighborhood, col2_name2, Data):
    target_set = neighborhood  # target set will be neighbourhood of col1
    equiv_class_dict = {}  # dictionary storing equiv classes for all attribute
    for col in Data:
        equiv_class_dict[col] = list(Data.groupby([col]))

    arr = equiv_class_dict[col2_name2]  # group equiv classes according to column2
    lower_approx = []
    upper_approx = []
    for name, group in arr:
        selected_list = list(group.index)
        if (set(selected_list).issubset(target_set)):  # if a subset, then append to lower_approx
            lower_approx.append(selected_list)
        if (set(selected_list) & set(target_set)):  # if intersection is not null, then append to upper_approx
            upper_approx.append(selected_list)
    lower_approx_count = sum(map(len, lower_approx))  # count the no. of elements in lower_approx
    upper_approx_count = sum(map(len, upper_approx))
    if (lower_approx_count == 0):
        roughness = 1
    else:
        roughness = 1 - (lower_approx_count / upper_approx_count)  # return roughness for a(i) = alpha
    # print("roughness of numerical", roughness)
    return roughness


def cluster_distance(cluster, Data):
    distance = 0
    print("Cluster", cluster)
    Categorical_Data = Data.ix[cluster]
    Numerical_Data = Categorical_Data[['V0']]
    Categorical_Data = Categorical_Data.drop(Categorical_Data.columns[[4]], axis=1)

    for i in range(0, len(cluster) - 1):  # categorical
        a = np.array(Categorical_Data.ix[cluster[i]])
        for k in range(i + 1, len(cluster)):
            b = np.array(Categorical_Data.ix[cluster[k]])
            distance += sum(a != b)

    neighborhood_class_dict = {}  # calculating neighborhood of numerical attr
    threshold = 10
    for index, row in Data.iterrows():
        neighbor_list = []
        for index1, row1 in Data.iterrows():
            if ((row['V0'] > row1['V0'])):
                if (row['V0'] - row1['V0'] <= threshold):
                    neighbor_list.append(index1)
            elif (row['V0'] < row1['V0']):
                if (row1['V0'] - row['V0'] <= threshold):
                    neighbor_list.append(index1)
            else:
                neighbor_list.append(index1)
        neighborhood_class_dict[index] = neighbor_list

    for i in range(0, len(cluster) - 1):  # for numerical
        a = neighborhood_class_dict[i]
        for k in range(i + 1, len(cluster)):
            b = neighborhood_class_dict[k]
            if (a != b):
                distance += 1

    print(len(cluster))
    final_distance = (2 * distance) / (len(cluster) * (len(cluster) - 1))

    return final_distance


def cluster_numerical(cluster1, cluster2):
    global cluster1_numerical_global
    cluster1_numerical_global = cluster1
    global cluster2_numerical_global
    cluster2_numerical_global = cluster2
    # print("cluster1_numerical_global", cluster1_numerical_global)
    # print("cluster2_numerical_global", cluster2_numerical_global)


def mmer_roughness(Data):
    # check if all columns have more than 1 unique values, if not drop that column/columns
    # print("Dropped columns:")
    # print(Data)
    for col1 in Data:
        values_in_col_name1 = list(Data[col1].unique())  # extracting each unique value in column1
        # print(col1, len(values_in_col_name1))
        if (len(values_in_col_name1) == 1):
            Data = Data.drop(col1, axis=1)

    min_mean_roughness = {}  # array to store the minimum of average of roughness of each attr
    roughness_list_dict = defaultdict(dict)  # needed to use 2D dictionary

    for col1 in Data:

        neighborhood_class_dict = {}  # calculating neighborhood of numerical attr
        threshold = 10
        for index, row in Data.iterrows():
            neighbor_list = []
            for index1, row1 in Data.iterrows():
                if ((row['V0'] > row1['V0'])):
                    if (row['V0'] - row1['V0'] <= threshold):
                        neighbor_list.append(index1)
                elif (row['V0'] < row1['V0']):
                    if (row1['V0'] - row['V0'] <= threshold):
                        neighbor_list.append(index1)
                else:
                    neighbor_list.append(index1)
            neighborhood_class_dict[index] = neighbor_list

        if (col1 == 'V0'):

            mean_roughness_numerical = []
            roughness_list_dict_numerical = {}
            for key in neighborhood_class_dict:
                roughness_list_numerical = []
                for col2 in Data:
                    if (col1 != col2):
                        roughness_list_numerical.append(
                            roughness_numerical_to_categorical(neighborhood_class_dict[key], col2, Data))
                mean_roughness_numerical.append(sum(roughness_list_numerical) / float(len(roughness_list_numerical)))
                roughness_list_dict_numerical[key] = sum(roughness_list_numerical) / float(
                    len(roughness_list_numerical))

            min_mean_roughness['V0'] = (min(mean_roughness_numerical))
            min_roughness_obj = min(roughness_list_dict_numerical, key=roughness_list_dict_numerical.get)
            l = Data.index.tolist()
            cluster1 = neighborhood_class_dict[min_roughness_obj]
            cluster2 = list(set(l) - set(neighborhood_class_dict[min_roughness_obj]))

        values_in_col_name1 = list(Data[col1].unique())  # extracting each unique value in column1
        mean_roughness = []  # list for mean roughness for a(i) = alpha
        for val in values_in_col_name1:
            roughness_list = []  # array of a(i) = alpha w.r.t. other columns
            for col2 in Data:
                if (col1 != col2):  # not to be compared with itself
                    if (col2 == 'V0'):
                        roughness_list.append(
                            roughness_categorical_to_numerical(neighborhood_class_dict, col1, col2, val, Data))

                    else:
                        roughness_list.append(roughness(col1, col2, val, Data))
            mean_roughness.append(sum(roughness_list) / float(len(roughness_list)))
            roughness_list_dict[col1][val] = sum(roughness_list) / float(len(roughness_list))
        min_mean_roughness[col1] = (min(mean_roughness))
    for key in min_mean_roughness:
        print(key, min_mean_roughness[key])

    min_roughness_attr = min(min_mean_roughness, key=min_mean_roughness.get)  # find the attr with minimum roughness
    print(min_roughness_attr)
    if (min_roughness_attr == 'V0'):
        cluster_numerical(cluster1, cluster2)
        return 'V0', 1

    return min_roughness_attr, min(roughness_list_dict[min_roughness_attr])


MMeR(Data, k)
