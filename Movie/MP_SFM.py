import copy
import random
import time
import numpy as np
import movie_data_deal


class MP_FSM():
    def __init__(self,data,matrix_factorized,user_movie_matrix,K,uid):
        self.data = data
        self.l = 10
        self.K = K
        self.k_list=[]
        self.data_id = None
        self.data_group = None
        self.group_count = [0] * self.l
        self.S = []
        self.func_value = 0
        self.Ri = []
        self.epsilon = 0.1
        self.counter = 0
        self.tracking_delta = {}
        self.movie_id_mapping_to_group = {}
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
        for movie_id,group in zip(self.data_id,self.data_group):
            self.movie_id_mapping_to_group[movie_id] = group

    def MP_FSM_algorithm(self):
        start_time = time.time()
        self.sampling_Ri()
        delta_max = 0
        v_max = None
        v_max_group = None
        for movie_id, group in zip(self.data_id,self.data_group):
            self.movie_id_mapping_to_group[movie_id] = group
            if self.user_movie_matrix[movie_id] != 0:
                continue
            self.tracking_delta[movie_id] = self.cal_fv(movie_id)
            if delta_max < self.tracking_delta[movie_id]:
                delta_max = self.tracking_delta[movie_id]
                v_max = movie_id
                v_max_group = group
        self.S.append(v_max)
        self.func_value += delta_max
        self.group_count[v_max_group] += 1
        tau = (1 - self.epsilon) * delta_max
        threshold = delta_max * (self.epsilon / self.K)

        while tau > threshold:
            self.counter += 1
            for movie_id, group in zip(self.data_id,self.data_group):
                if self.group_count[group] == self.k_list[group]:
                    continue
                if self.user_movie_matrix[movie_id] != 0 or movie_id in self.S:
                    continue
                if self.tracking_delta[movie_id] < tau:
                    continue
                tmp_value = self.delta_recsys(movie_id)
                if tmp_value >= tau:
                    # print(movie_id,tmp_value,tau)
                    self.S.append(movie_id)
                    self.func_value += tmp_value
                    self.group_count[group] += 1
                    self.tracking_delta[movie_id] = movie_id

            if len(self.S) == self.K:
                break
            tau *= (1 - self.epsilon)


        for movie_id in self.Ri:
            group = self.movie_id_mapping_to_group[movie_id]
            if self.group_count[group] == self.k_list[group]:
                continue
            if self.user_movie_matrix[movie_id] != 0 or movie_id in self.S:
                continue
            self.S.append(movie_id)

        self.func_value = self.get_f_recsys(self.S)
        end_time = time.time()
        self.time = end_time - start_time




    def sampling_Ri(self):
        for group in range(self.l):
            group_ids = [id for id, g in zip(self.data_id, self.data_group) if g == group]
            sample_id = self.reservoir_sampling(group_ids,self.k_list[group])
            self.Ri += sample_id

    def reservoir_sampling(self,array, k):
        array = list(array)
        sample = array[:k].copy()
        size_left = len(array) - k
        left_array = array[k:]
        idxs = np.random.choice(array, size=size_left, replace=True)
        for position, idx in enumerate(idxs):
            if idx < k:
                sample[idx] = left_array[position]
        return sample


    def delta_recsys(self, movie_id):
        S = copy.copy(self.S)
        S.append(movie_id)
        out = self.get_f_recsys(S)
        delta_ = out - self.func_value
        return delta_

    def cal_fv(self,movie_id):
        S = [movie_id]
        out = self.get_f_recsys(S)
        return out

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
        self.MP_FSM_algorithm()
        file_name = "MP_FSM_" + "K=" + str(self.K) + "_movie_PR_output.txt"
        with open(file_name, "w") as file:
            file.write("value: " + str(self.func_value)+"\n")
            file.write("time: " + str(self.time)+"\n")
            file.write(" ".join(map(str, self.S)))

def func():
    data,matrix_factorized,user_movie_matrix = movie_data_deal.read_movie_group_and_matrix()
    uid = 0
    k_list = [10,20,30,40,50,60,70,80,90,100]
    for k in k_list:
        mp_fsm = MP_FSM(data,matrix_factorized,user_movie_matrix,k,uid)
        mp_fsm.main()


if __name__ == "__main__":
    data,matrix_factorized,user_movie_matrix = movie_data_deal.read_movie_group_and_matrix()
    uid = 0
    k_list = [10,20,30,40,50,60,70,80,90,100]
    for k in k_list:
        mp_fsm = MP_FSM(data,matrix_factorized,user_movie_matrix,k,uid)
        mp_fsm.main()

