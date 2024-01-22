import copy
import random
import time
import numpy as np
import movie_data_deal
import math

class OnePassStream():
    def __init__(self,data,matrix_factorized,user_movie_matrix,K,uid,beta):
        self.data = data
        self.l = 10
        self.beta = beta
        self.K = K
        self.k_list=[]
        self.friend_ships = {}

        self.group_count = [0] * self.l
        self.minimum_value = [-1] * self.l
        self.minimum_value_id = [-1] * self.l
        self.minimum_value_group = [-1] * self.l

        self.S = []
        self.S_value = []
        self.S_group = []
        # Buffer
        self.B = []
        self.B_group = []
        self.B_limit = int(2 * self.K * math.log10(self.K) + 3 * self.K)
        self.func_value = 0
        self.data_id = []
        self.data_group = []
        self.counter = 0
        self.is_S_update = False
        self.matrix_factorized = matrix_factorized
        self.VxV = matrix_factorized["VxV"]
        self.VxU = matrix_factorized["VxU"][:,uid]
        self.lambda_ = 0.75
        self.user_movie_matrix = user_movie_matrix[uid]
        self.movie_id_mapping_to_group = {}
        self.time = None

    def data_deal(self):
        self.data_id = self.data['movie_id'].tolist()
        self.data_group = self.data['group'].tolist()
        self.k_list = movie_data_deal.decide_k_PR(self.data,self.K,self.l)
        print(self.VxU.shape,self.VxV.shape)
        for movie_id, group in zip(self.data_id,self.data_group):
            self.movie_id_mapping_to_group[movie_id] = group


    def OPS_algorithm(self):
        for movie_id, group in zip(self.data_id,self.data_group):
            if self.counter % 1000 == 0:
                print(self.counter," ",len(self.S)," ",self.func_value, len(self.B))
            self.counter += 1
            if self.user_movie_matrix[movie_id] != 0:
                continue

            if self.group_count[group] < self.k_list[group]:
                self.S.append(movie_id)
                tmp_value = self.delta_recsys(movie_id)
                self.S_value.append(tmp_value)
                self.S_group.append(group)
                self.group_count[group] += 1
                self.func_value += tmp_value

                if self.minimum_value[group] == -1 or self.minimum_value[group] >= tmp_value:
                    self.minimum_value[group] = tmp_value
                    self.minimum_value_id[group] = movie_id
                    self.minimum_value_group[group] = group
            else:
                min_value = self.minimum_value[group]
                min_id = self.minimum_value_id[group]

                if min_id == -1:
                    continue
                if self.is_element_add(min_id,movie_id,self.beta * min_value):
                    self.update_S(min_id,movie_id,group)
                    self.is_S_update = True
                else:
                    g_value = self.cal_value_by_group(group) / self.k_list[group]
                    tmp_value = self.delta_recsys(movie_id)
                    if min_value >= g_value/(1 + self.beta) and tmp_value >= g_value / 2:
                        self.B.append(movie_id)
                        self.B_group.append(group)
                        if len(self.B) >= self.B_limit and self.is_S_update:
                            self.use_Buffer()
                            self.delete_Buffer()
                        if len(self.B) >= self.B_limit:
                            index_to_remove = random.randint(0, len(self.B) - 1)
                            self.B.pop(index_to_remove)
                            self.B_group.pop(index_to_remove)
        self.use_Buffer()
        self.delete_Buffer()
        print("ops ",self.func_value)


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

    def is_element_add(self,min_id,movie_id,threshold):
        S = copy.copy(self.S)
        S.remove(min_id)
        S.append(movie_id)
        tmp_func_value = self.get_f_recsys(S)
        return tmp_func_value - self.func_value >= threshold

    def update_S(self,min_id,movie_id,group):
        min_index = self.S.index(min_id)
        self.S.pop(min_index)
        self.S_group.pop(min_index)
        self.S.append(movie_id)
        self.S_group.append(group)
        self.update_minimum_element()

    def update_minimum_element(self):
        S = copy.copy(self.S)
        self. minimum_value = [-1] * self.l
        self. minimum_value_id = [-1] * self.l
        self. minimum_value_group = [-1] * self.l
        self.S_value = []
        self.S = []
        self.func_value = 0
        for movie_id, group in zip(S,self.S_group):
            tmp_value = self.delta_recsys(movie_id)
            self.S.append(movie_id)
            self.func_value += tmp_value
            self.S_value.append(tmp_value)
            if self.minimum_value[group] == -1 or self.minimum_value[group] >= tmp_value:
                self.minimum_value[group] = tmp_value
                self.minimum_value_id[group] = movie_id
                self.minimum_value_group[group] = group
        self.func_value = self.get_f_recsys(self.S)

    def cal_value_by_group(self,target_group):
        tmp_value = 0
        for value, group in zip(self.S_value,self.S_group):
            if group == target_group:
                tmp_value += value
        return tmp_value

    def use_Buffer(self):
        for movie_id, group in zip(self.B, self.B_group):
            min_value = self.minimum_value[group]
            min_id = self.minimum_value_id[group]

            if min_id == -1:
                continue
            if self.is_element_add(min_id,movie_id,self.beta * min_value):
                self.update_S(min_id,movie_id,group)
                id_index = self.B.index(movie_id)
                self.B.pop(id_index)
                self.B_group.pop(id_index)

    def delete_Buffer(self):
        for movie_id, group in zip(self.B, self.B_group):
            b_value = self.delta_recsys(movie_id)
            g_value = self.cal_value_by_group(group) / self.k_list[group]
            if b_value < g_value / 2:
                id_index = self.B.index(movie_id)
                self.B.pop(id_index)
                self.B_group.pop(id_index)

    def local_search(self):
        print("local_search_begin ")
        iterator = 0
        while True:
            iterator += 1
            flag = False
            for i in range(len(self.B)):
                b_id = self.B[i]
                b_group = self.B_group[i]
                min_id = self.minimum_value_id[b_group]
                if self.is_element_add(min_id,b_id,1):
                    flag = True
                    self.update_S(min_id,b_id,b_group)
                    self.B[i] = min_id
                    self.B_group[i] = b_group
            if not flag:
                break
        print("local_search ops ",self.func_value)
        print("local_search_end ")


    def main(self):
        self.data_deal()
        start_time = time.time()
        self.OPS_algorithm()
        self.local_search()
        end_time = time.time()
        self.time = end_time - start_time
        file_name = "OPS_limit_" + "K=" + str(self.K) + "_movie_PR_output.txt"
        with open(file_name, "w") as file:
            file.write("value: " + str(self.func_value)+"\n")
            file.write("time: " + str(self.time)+"\n")
            file.write(" ".join(map(str, self.S)))


def func():
    data,matrix_factorized,user_movie_matrix = movie_data_deal.read_movie_group_and_matrix()
    uid = 0
    k_list = [10,20,30,40,50,60,70,80,90,100]
    for k in k_list:
        ops = OnePassStream(data,matrix_factorized,user_movie_matrix,k,uid,40)
        ops.main()


if __name__ == "__main__":
    data,matrix_factorized,user_movie_matrix = movie_data_deal.read_movie_group_and_matrix()
    uid = 0
    k_list = [10,20,30]
    for k in k_list:
        ops = OnePassStream(data,matrix_factorized,user_movie_matrix,k,uid,40)
        ops.main()
