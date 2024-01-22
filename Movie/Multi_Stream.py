import random
import copy
import time
import numpy as np
import movie_data_deal

class Multi_Stream():
    def __init__(self,data,matrix_factorized,user_movie_matrix,K,uid,beta):
        self.data = data
        self.l = 10
        self.beta = beta
        self.alpha = 0.1
        self.K = K
        self.k_list=[]
        self.friend_ships = {}
        self.group_count = [0] * self.l
        self.Ri = []
        self.Ri_group = []
        self.S = []
        self.S_group = []
        self.func_value = 0
        self.data_id = []
        self.data_group = []
        self.counter = 1
        self.tracking_delta = {}
        self.time = None
        self.matrix_factorized = matrix_factorized
        self.VxV = matrix_factorized["VxV"]
        self.VxU = matrix_factorized["VxU"][:,uid]
        self.lambda_ = 0.75
        self.user_movie_matrix = user_movie_matrix[uid]

    def data_deal(self):
        self.data_id = self.data['movie_id'].tolist()
        self.data_group = self.data['group'].tolist()
        self.k_list = movie_data_deal.decide_k_PR(self.data,self.K,self.l)
        print(self.VxU.shape,self.VxV.shape)


    def MS_algorithm(self):
        start_time = time.time()
        self.sampling_Ri()
        boundary = self.alpha / self.K
        while (1-self.alpha)**((1+self.alpha)**self.counter) > boundary  and len(self.S) != self.K:
            max_value = self.ops_stream()
            self.select_neighbor(max_value)
            self.counter += 1
        self.Greedy_algorithm(self.Ri,self.Ri_group)
        self.func_value = self.get_f_recsys(self.S)
        print("ops ",self.func_value,len(self.S),self.group_count)
        end_time = time.time()
        self.time = end_time - start_time

    def sampling_Ri(self):
        for group in range(self.l):
            group_ids = [id for id, g in zip(self.data_id, self.data_group) if g == group]
            sample_id = random.sample(group_ids, int(len(group_ids)/2))
            sample_group = [group] * len(sample_id)
            self.Ri += sample_id
            self.Ri_group += sample_group


    def ops_stream(self):
        max_value = -1
        max_id = -1
        max_group = -1
        for movie_id, group in zip(self.data_id,self.data_group):
            if self.group_count[group] == self.k_list[group]:
                continue
            if self.user_movie_matrix[movie_id] != 0 or movie_id in self.S:
                continue
            if movie_id not in self.tracking_delta:
                tmp_value = self.delta_recsys(movie_id)
                self.tracking_delta[movie_id] = tmp_value
            else:
                if self.tracking_delta[movie_id] <= max_value:
                    continue
                tmp_value = self.delta_recsys(movie_id)
                self.tracking_delta[movie_id] = tmp_value
            if tmp_value > max_value:
                max_id = movie_id
                max_value = tmp_value
                max_group = group
        self.S.append(max_id)
        self.S_group.append(max_group)
        self.func_value = self.get_f_recsys(self.S)
        self.group_count[max_group] += 1
        return max_value

    def select_neighbor(self,max_value):
        for movie_id, group in zip(self.data_id,self.data_group):
            if self.group_count[group] == self.k_list[group]:
                continue
            if self.user_movie_matrix[movie_id] != 0 or movie_id in self.S:
                continue
            if self.tracking_delta[movie_id] <= max_value * (1-self.alpha):
                continue
            tmp_value = self.delta_recsys(movie_id)
            self.tracking_delta[movie_id] = tmp_value
            if tmp_value > max_value * (1 - self.alpha):
                self.S.append(movie_id)
                self.S_group.append(group)
                self.func_value = self.get_f_recsys(self.S)
                self.group_count[group] += 1


    def Greedy_algorithm(self, extra_id, extra_group):
        greedy_count = self.K - len(self.S)
        print("greedy",greedy_count)
        maximum_value = [-1] * self.l
        maximum_value_id = [-1] * self.l
        maximum_value_group = [-1] * self.l
        for p in range(greedy_count):
            for movie_id, group in zip(extra_id, extra_group):
                if self.group_count[group] == self.k_list[group]:
                    continue
                if self.user_movie_matrix[movie_id] != 0 or movie_id in self.S:
                    continue
                if movie_id not in self.tracking_delta:
                    tmp_value = self.delta_recsys(movie_id)
                    self.tracking_delta[movie_id] = tmp_value
                    if tmp_value > maximum_value[group]:
                        maximum_value[group] = tmp_value
                        maximum_value_id[group] = movie_id
                        maximum_value_group[group] = group
                else:
                    if self.tracking_delta[movie_id] > maximum_value[group]:
                        tmp_value = self.delta_recsys(movie_id)
                        self.tracking_delta[movie_id] = tmp_value
                        if tmp_value > maximum_value[group]:
                            maximum_value[group] = tmp_value
                            maximum_value_id[group] = movie_id
                            maximum_value_group[group] = group

            tuples = list(zip(maximum_value, maximum_value_id, maximum_value_group))
            sorted_data = sorted(tuples, key=lambda x: x[0], reverse=True)
            for value, movie_id, group in sorted_data:
                if self.group_count[group] == self.k_list[group]:
                    continue
                self.group_count[group] += 1
                self.S.append(movie_id)
                self.func_value = self.get_f_recsys(self.S)
                for g in range(self.l):
                    maximum_value[g] = -1
                    maximum_value_id[g] = -1
                    maximum_value_group[g] = -1
                break


    def delta_recsys(self, movie_id):
        S = copy.copy(self.S)
        S.append(movie_id)
        out = self.get_f_recsys(S)
        delta_ = out - self.func_value
        return delta_

    def get_f_recsys(self, S):
        S_idxs = sorted(S)
        sub_matrix1 = self.VxV[:, S_idxs]
        max_vector = np.max(sub_matrix1, axis=1)

        out = max_vector.sum() * self.lambda_
        tot_relevance = (self.VxU[S_idxs]).sum()
        out = out + (1-self.lambda_) * tot_relevance
        return out

    def main(self):
        self.data_deal()
        self.MS_algorithm()
        file_name = "MS_" + "K=" + str(self.K) + "_movie_PR_output.txt"
        with open(file_name, "w") as file:
            file.write("value: " + str(self.func_value)+"\n")
            file.write("time: " + str(self.time)+"\n")
            file.write(" ".join(map(str, self.S)))


def func():
    data,matrix_factorized,user_movie_matrix = movie_data_deal.read_movie_group_and_matrix()
    uid = 0
    k_list = [10,20,30,40,50,60,70,80,90,100]
    for k in k_list:
        ms = Multi_Stream(data,matrix_factorized,user_movie_matrix,k,uid,40)
        ms.main()

if __name__ == "__main__":
    data,matrix_factorized,user_movie_matrix = movie_data_deal.read_movie_group_and_matrix()
    uid = 0
    k_list = [10,20,30,40,50,60,70,80,90,100]
    for k in k_list:
        ms = Multi_Stream(data,matrix_factorized,user_movie_matrix,k,uid,40)
        ms.main()
