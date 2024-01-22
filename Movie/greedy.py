import copy
import time
import movie_data_deal
import numpy as np

class Greedy():
    def __init__(self,data,matrix_factorized,user_movie_matrix,K,uid):
        self.data = data
        self.l = 10
        self.K = K
        self.k_list=[]
        self.ni = -float('inf')
        self.group_count = [0] * self.l
        self. maximum_value = [self.ni] * self.l
        self. maximum_value_id = [self.ni] * self.l
        self. maximum_value_group = [self.ni] * self.l
        self.S = []
        self.func_value = 0
        self.data_id = None
        self.data_group = None
        self.counter = 0
        self.tracking_delta = {}
        self.matrix_factorized = matrix_factorized
        self.VxV = matrix_factorized["VxV"]
        self.VxU = matrix_factorized["VxU"][:,uid]
        self.lambda_ = 0.75
        self.user_movie_matrix = user_movie_matrix[uid]
        self.time = None
        self.to_filter = []
        self.movie_id_mapping_to_group = {}


    def data_deal(self):
        self.data_id = self.data['movie_id'].tolist()
        self.data_group = self.data['group'].tolist()
        self.k_list = movie_data_deal.decide_k_PR(self.data,self.K,self.l)
        for movie_id in range(len(self.user_movie_matrix)):
            if self.user_movie_matrix[movie_id] != 0:
                self.to_filter.append(movie_id)
        print(self.VxU.shape,self.VxV.shape)
        for movie_id, group in zip(self.data_id,self.data_group):
            self.movie_id_mapping_to_group[movie_id] = group

    def Greedy_algorithm(self):
        start_time = time.time()
        for p in range(self.K):
            for movie_id, group in zip(self.data_id,self.data_group):
                if self.group_count[group] == self.k_list[group]:
                    continue
                if self.user_movie_matrix[movie_id] != 0 or movie_id in self.S:
                    continue
                tmp_value = self.delta_recsys(movie_id)
                if tmp_value > self.maximum_value[group]:
                    self.maximum_value[group] = tmp_value
                    self.maximum_value_id[group] = movie_id
                    self.maximum_value_group[group] = group
            tuples = list(zip(self.maximum_value, self.maximum_value_id, self.maximum_value_group))
            sorted_data = sorted(tuples, key=lambda x: x[0], reverse=True)
            for value, movie_id, group in sorted_data:
                if self.group_count[group] == self.k_list[group]:
                    continue
                self.group_count[group] += 1
                self.S.append(movie_id)
                self.func_value = self.get_f_recsys(self.S)
                for g in range(self.l):
                    self.maximum_value[g] = -1
                    self.maximum_value_id[g] = -1
                    self.maximum_value_group[g] = -1
                break
        self.func_value = self.get_f_recsys(self.S)
        print("ops ",self.func_value,len(self.S),self.group_count)
        end_time = time.time()
        self.time = end_time - start_time


    def delta_recsys(self, movie_id):
        S = copy.copy(self.S)
        S.append(movie_id)
        out  = self.get_f_recsys(S)
        delta_ = out - self.func_value
        return delta_

    def get_f_recsys(self, S):
        S_idxs = sorted(S)
        sub_matrix1 = self.VxV[:, S_idxs]
        # sub_matrix1 = sub_matrix1[~np.isin(np.arange(sub_matrix1.shape[0]), self.to_filter)]
        max_vector = np.max(sub_matrix1, axis=1)
        out = max_vector.sum() * self.lambda_
        tot_relevance = (self.VxU[S_idxs]).sum()
        out = out + (1-self.lambda_) * tot_relevance
        return out

    def main(self):
        self.data_deal()
        self.Greedy_algorithm()
        file_name = "Greedy_" + "K=" + str(self.K) + "_movie_PR_output.txt"
        with open(file_name, "w") as file:
            file.write("value: " + str(self.func_value)+"\n")
            file.write("time: " + str(self.time)+"\n")
            file.write(" ".join(map(str, self.S)))

def func():
    data,matrix_factorized,user_movie_matrix = movie_data_deal.read_movie_group_and_matrix()
    uid = 0
    k_list = [10,20,30,40,50,60,70,80,90,100]
    for k in k_list:
        greedy = Greedy(data,matrix_factorized,user_movie_matrix,k,uid)
        greedy.main()

if __name__ == "__main__":
    data,matrix_factorized,user_movie_matrix = movie_data_deal.read_movie_group_and_matrix()
    uid = 0
    k_list = [10,20,30,40,50,60,70,80,90,100]
    for k in k_list:
        greedy = Greedy(data,matrix_factorized,user_movie_matrix,k,uid)
        greedy.main()

