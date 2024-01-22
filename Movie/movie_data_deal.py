import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

def read_movies():
    with open("../data/movies.txt", "r") as file:
        data = file.readlines()
    all_genres = ["Action","Adventure","Animation","Children\'s","Comedy","Crime","Documentary","Drama","Fantasy",
                  "Film-Noir","Horror","Musical", "Mystery","Romance","Sci-Fi","Thriller","War","Western"]
    all_genres_to_encode = {}
    for genre,i in zip(all_genres,range(len(all_genres))):
        all_genres_to_encode[genre] = i

    movies_id = []
    movie_data = {}
    for item in data:
        parts = item.strip().split("::")
        movie_id = int(parts[0]) - 1
        movies_id.append(movie_id)
        title = parts[1]
        movie_genres = parts[2].split("|")
        feature_vector = [0] * 18
        for genre in movie_genres:
            feature_vector[all_genres_to_encode[genre]] = 1
        movie_data[movie_id] = feature_vector

    mapping_movies_id = {}
    movies_id.sort()
    for index, value in enumerate(movies_id):
        mapping_movies_id[value] = index
    return movie_data,movies_id,mapping_movies_id


def read_rating(mapping_movies_id):
    with open("../data/ratings.txt", "r") as file:
        data = file.readlines()
    rating_movies_id = []
    user_movie_matrix = {}

    for item in data:
        parts = item.strip().split("::")
        user_id = int(parts[0]) - 1
        if user_id not in user_movie_matrix:
            user_movie_matrix[user_id] = [0]*len(mapping_movies_id)
        movie_id = int(parts[1]) - 1
        if movie_id not in rating_movies_id:
            rating_movies_id.append(movie_id)
        rating = int(parts[2])
        map_id = mapping_movies_id[movie_id]
        user_movie_matrix[user_id][map_id] = rating

    with open('../data/user_movie_rating_matrix.txt', 'w') as file:
        for user_id in user_movie_matrix.keys():
            for rating in user_movie_matrix[user_id]:
                file.write(str(rating) + '\t')
            file.write("\n")

    matrix_decompose(user_movie_matrix)
    return rating_movies_id


def movies_k_means(movie_data,mapping_movies_id):
    num_columns = len(movie_data[1])
    movie_array = np.zeros((len(movie_data), num_columns))
    for i, key in enumerate(sorted(movie_data.keys())):
        movie_array[i] = movie_data[key]

    kmeans = KMeans(n_clusters=10, random_state=0,n_init='auto').fit(movie_array)
    cluster_labels = kmeans.labels_

    with open('../data/movies_group.txt', 'w') as file:
        for movie_index, cluster_label in zip(movie_data.keys(),cluster_labels):
            line = f'{mapping_movies_id[movie_index]}\t{cluster_label}\n'
            file.write(line)

def read_movie_group_and_matrix():
    column_profiles = ['movie_id','group']
    data = pd.read_csv('../data/movies_group.txt', sep='\t', names=column_profiles,
                       index_col=False, header=None, encoding='utf-8')
    foldername = "../data/"
    with open(foldername + "matrix-factorized.p", 'rb') as handle:
        matrix_factorized = pickle.load(handle)

    user_movie_matrix = {}
    with open('../data/user_movie_rating_matrix.txt', 'r') as file:
        for line in file:
            ratings = line.strip().split('\t')
            ratings = [int(rating) for rating in ratings]
            user_movie_matrix[len(user_movie_matrix)] = ratings


    return data,matrix_factorized,user_movie_matrix


def matrix_decompose(user_movie_matrix):
    num_columns = len(user_movie_matrix[1])
    matrix_array = np.zeros((len(user_movie_matrix), num_columns))
    for key in user_movie_matrix.keys():
        matrix_array[key] = user_movie_matrix[key]

    scaler = MinMaxScaler()
    matrix_array = scaler.fit_transform(matrix_array)

    n_components = 80
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=1000)

    U = model.fit_transform(matrix_array)
    V = model.components_.T

    print(U.shape,V.shape)
    VxV = np.dot(V, V.T)
    VxU = np.dot(V, U.T)
    print(" VxV,VxU ",VxV.shape,VxU.shape)
    matrix_factorized = {"VxV": VxV, "VxU": VxU}

    foldername = "../data/"
    with open(foldername + "matrix-factorized.p", 'wb') as handle:
        pickle.dump(matrix_factorized, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_matrix(file_name):
    with open(file_name, 'r') as file_W:
        lines = file_W.readlines()
    W = None
    matrix_dict = {}
    for i, line in enumerate(lines):
        elements = line.strip().split('\t')
        row = [float(element) for element in elements]
        if W is None:
            W = np.array([row])
        else:
            W = np.vstack([W, row])
        matrix_dict[i+1] = row
    return  matrix_dict,W

def decide_k_PR(data,K,l):
    group_counts = data['group'].value_counts().sort_index()
    initial_group_allocations = (group_counts / group_counts.sum() * K).round().astype(int)
    initial_total_allocation = initial_group_allocations.sum()
    difference = K - initial_total_allocation
    if difference > 0:
        while difference > 0:
            max_group = initial_group_allocations.idxmax()
            initial_group_allocations[max_group] += 1
            difference -= 1
    else:
        while difference < 0:
            non_zero_groups = initial_group_allocations[initial_group_allocations > 0]
            min_group = non_zero_groups.idxmin()
            initial_group_allocations[min_group] -= 1
            difference += 1
    k_list = initial_group_allocations.tolist()
    print('k_list',k_list)
    return k_list

def decide_k_ER(data,K,l):
    resources_per_group = K // l
    remaining_resources = K % l
    group_allocations = [resources_per_group] * l
    for i in range(remaining_resources):
        group_allocations[i] += 1
    k_list = group_allocations
    print('k_list',k_list)
    return k_list

if __name__ == "__main__":
    movie_data,movies_id,mapping_movies_id = read_movies()
    movies_k_means(movie_data,mapping_movies_id)
    rating_movies_id = read_rating(mapping_movies_id)


