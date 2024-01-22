import copy
import math
import random
import time
import numpy as np
import movie_data_deal


class SP_FSM():
    def __init__(self,data,matrix_factorized,user_movie_matrix,K,uid):
        self.data = data

        self.l = 10
        self.K = K
        self.k_list=[]
        self.friend_ships = {}
        self.data_id = None
        self.data_group = None
        self.S_tau = {}
        self.S_tau_to_value = {}
        self.S_tau_to_element_sets = {}
        self.count_ki_tau = {}
        self.delta_max = 0
        self.Ri = []
        self.B = set()
        self.LB = 0
        self.alpha = 0.5
        self.log_alpha = np.log(1 + self.alpha)
        self.beta = 0.5
        self.T = []
        self.max_tau = None
        self.counter = 0
        self.movie_id_mapping_to_group = {}
        self.history_best_value = {}
        self.matrix_factorized = matrix_factorized
        self.VxV = matrix_factorized["VxV"]
        self.VxU = matrix_factorized["VxU"][:,uid]
        self.lambda_ = 0.75
        self.user_movie_matrix = user_movie_matrix[uid]
        self.B_limit = int(2 * self.K * math.log10(self.K))
        self.time = None
        self.B_group= {}

    def data_deal(self):
        self.data_id = self.data['movie_id'].tolist()
        self.data_group = self.data['group'].tolist()
        self.k_list = movie_data_deal.decide_k_PR(self.data,self.K,self.l)
        print(self.VxU.shape,self.VxV.shape)
        for movie_id,group in zip(self.data_id,self.data_group):
            self.movie_id_mapping_to_group[movie_id] = group


    def SP_FSM_algorithm(self):
        self.sampling_Ri()
        for movie_id, group in zip(self.data_id,self.data_group):
            if self.user_movie_matrix[movie_id] != 0:
                continue
            fv = self.cal_fv(movie_id)
            self.delta_max = max(self.delta_max,fv)
            self.history_best_value[movie_id] = fv
            self.cal_and_update_S_tau()
            self.counter += 1
            for tau in self.T:
                if self.count_ki_tau[tau][group] == self.k_list[group]:
                    continue
                tmp_value = self.delta_recsys(movie_id,tau)
                if tmp_value >= tau:
                    self.S_tau[tau].append(movie_id)
                    self.S_tau_to_value[tau] += tmp_value
                    self.count_ki_tau[tau][group] += 1
                elif tmp_value >= self.beta * self.LB / self.K:
                    self.B.add(movie_id)
            self.LB = max(self.S_tau_to_value.values())

        upper_tau = self.choose_tau()
        # for tau in self.T:
        #     print("tau ",tau, len(self.S_tau[tau]), len(self.S_tau_to_element_sets[tau]))

        self.choose_Bi()
        self.B.update(self.Ri)
        for tau in self.T:
            if tau > upper_tau:
                break
            self.Greedy_algorithm(tau,self.B)
            # print("tau ",tau, len(self.S_tau[tau]),self.S_tau_to_value[tau])

        print("---------------------------------------------------------------")
        self.max_tau = max(self.S_tau_to_value, key=lambda k: self.S_tau_to_value[k])
        print("maxtau ",self.max_tau, len(self.S_tau[self.max_tau]),self.S_tau_to_value[self.max_tau])



    def choose_tau(self):
        check_group_limit = {}
        for group in range(self.l):
            check_group_limit[group] = []
        upper_tau = None
        for tau in self.T:
            tau_flag = False
            for group in range(self.l):
                if self.count_ki_tau[tau][group] == self.k_list[group]:
                    check_group_limit[group].append(tau)
                    tau_flag = True
            if not tau_flag:
                upper_tau = tau
                break

        if upper_tau == None:
            for group in range(self.l):
                if len(check_group_limit[group]) != len(self.T):
                    continue
                upper_tau = max(check_group_limit[group])
        return  upper_tau

    def Greedy_algorithm(self, tau, extra_id):
        print("greedy tau  ",tau)
        greedy_count = self.K - len(self.S_tau[tau])
        maximum_value = [-1] * self.l
        maximum_value_id = [-1] * self.l
        maximum_value_group = [-1] * self.l
        for p in range(greedy_count):
            for movie_id in extra_id:
                group = self.movie_id_mapping_to_group[movie_id]
                if self.count_ki_tau[tau][group] == self.k_list[group]:
                    continue
                if self.user_movie_matrix[movie_id] != 0 or movie_id in self.S_tau[tau]:
                    continue
                tmp_value = self.history_best_value[movie_id]
                # tmp_value = self.delta_recsys(movie_id,tau)
                if tmp_value > maximum_value[group]:
                    maximum_value[group] = tmp_value
                    maximum_value_id[group] = movie_id
                    maximum_value_group[group] = group

            tuples = list(zip(maximum_value, maximum_value_id, maximum_value_group))
            sorted_data = sorted(tuples, key=lambda x: x[0], reverse=True)

            for value, movie_id, group in sorted_data:
                if self.count_ki_tau[tau][group] == self.k_list[group]:
                    continue
                self.count_ki_tau[tau][group] += 1
                self.S_tau[tau].append(movie_id)
                self.S_tau_to_value[tau] = self.get_f_recsys(self.S_tau[tau])
                for g in range(self.l):
                    maximum_value[g] = -1
                    maximum_value_id[g] = -1
                    maximum_value_group[g] = -1
                break


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

    def choose_Bi(self):
        B_limit = []
        for group in range(self.l):
            group_ids = [id for id in self.B if self.movie_id_mapping_to_group[id] == group]
            B_value = [self.history_best_value[id] for id in self.B if self.movie_id_mapping_to_group[id] == group]
            sorted_ids = sorted(group_ids, key=lambda id: B_value[group_ids.index(id)], reverse=True)
            if self.k_list[group] > 10:
                k_group_limit = int(2 * self.k_list[group] * math.log10(self.k_list[group]))
            else:
                k_group_limit = 2 * self.k_list[group]
            B_limit += sorted_ids[:k_group_limit]
            self.B_group[group] = k_group_limit
        self.B = set(B_limit)

    def cal_and_update_S_tau(self):
        j_min = int(np.ceil(np.log(max(self.delta_max, self.LB) / (2 * self.K)) / self.log_alpha))
        j_max = int(np.floor(np.log(self.delta_max) / self.log_alpha))
        self.T = [(1+self.alpha)**j for j in range(j_min, j_max + 1)]
        for tau in self.T:
            if tau not in self.S_tau_to_value:
                self.S_tau[tau] = []
                self.S_tau_to_value[tau] = 0
                self.count_ki_tau[tau] = {i : 0 for i in range(self.l)}
        self.S_tau = {tau: self.S_tau[tau] for tau in self.T}
        self.S_tau_to_value = {tau: self.S_tau_to_value[tau] for tau in self.T}
        self.count_ki_tau = {tau : self.count_ki_tau[tau] for tau in self.T}


    def delta_recsys(self, movie_id,tau):
        S = copy.copy(self.S_tau[tau])
        S.append(movie_id)
        out = self.get_f_recsys(S)
        delta_ = out - self.S_tau_to_value[tau]
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
        start_time = time.time()
        self.SP_FSM_algorithm()
        end_time = time.time()
        self.time = end_time - start_time
        file_name = "SP_FSM_limit_" + "K=" + str(self.K) + "_movie_PR_output.txt"
        with open(file_name, "w") as file:
            file.write("value: " + str(self.S_tau_to_value[self.max_tau])+"\n")
            file.write("time: " + str(self.time)+"\n")
            file.write(" ".join(map(str, self.S_tau[self.max_tau])))

def func():
    data,matrix_factorized,user_movie_matrix = movie_data_deal.read_movie_group_and_matrix()
    uid = 0
    k_list = [10,20,30,40,50,60,70,80,90,100]
    for k in k_list:
        sp_fsm = SP_FSM(data,matrix_factorized,user_movie_matrix,k,uid)
        sp_fsm.main()

if __name__ == "__main__":
    data,matrix_factorized,user_movie_matrix = movie_data_deal.read_movie_group_and_matrix()
    uid = 0
    k_list = [10,20,30,40,50,60,70,80,90,100]
    for k in k_list:
        sp_fsm = SP_FSM(data,matrix_factorized,user_movie_matrix,k,uid)
        sp_fsm.main()

