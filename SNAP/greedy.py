import copy
import time
from intbitset import intbitset
import pandas as pd
import data_deal


class Greedy():
    def __init__(self,data,relationships,l,K):
        self.data = data
        self.relationships = relationships
        self.max_num = max(self.data.shape[0],self.relationships['user_id2'].max()) + 1

        self.l = l
        self.K = K
        self.k_list=[]
        self.friend_ships = {}

        self.group_count = [0] * self.l
        self. maximum_value = [-1] * self.l
        self. maximum_value_id = [-1] * self.l
        self. maximum_value_group = [-1] * self.l

        self.S = []
        self.func_value = 0
        self.element_set = intbitset()
        self.data_id = None
        self.data_group = None
        self.counter = 0
        self.tracking_delta = {}
        self.time = None

    def assign_group(self,categorie):
        if self.l == 2:
            if categorie == 0:
                return 0
            else:
                return 1
        elif self.l == 7:
            if categorie < 72:
                return categorie // 12
            else:
                return 6

    def decide_k_PR(self):
        group_counts = self.data['group'].value_counts().sort_index()
        initial_group_allocations = (group_counts / group_counts.sum() * self.K).round().astype(int)
        initial_total_allocation = initial_group_allocations.sum()
        difference = self.K - initial_total_allocation
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
        self.k_list = initial_group_allocations.tolist()
        print('k_list',self.k_list)

    def decide_k_ER(self):
        resources_per_group = self.K // self.l
        remaining_resources = self.K % self.l
        group_allocations = [resources_per_group] * self.l
        for i in range(remaining_resources):
            group_allocations[i] += 1
        self.k_list = group_allocations
        print('k_list',self.k_list)

    def write_deal_relationships(self):
        relationships_dict = {}
        for _, row in self.relationships.iterrows():
            user1 = row['user_id1']
            user2 = row['user_id2']
            if user1 not in relationships_dict:
                relationships_dict[user1] = []
            relationships_dict[user1].append(user2)

        file_name = None
        if self.l == 2:
            file_name = '../data/gender_relationships_deals.txt'
        elif self.l == 7:
            file_name = '../data/AGE_relationships_deals.txt'
        with open(file_name, 'w') as f:
            for user_id, friend_list in relationships_dict.items():
                f.write(f"{user_id} {' '.join(map(str, friend_list))}\n")


    def read_deal_relationships(self):
        file_name = None
        if self.l == 2:
            file_name = '../data/gender_relationships_deals.txt'
        elif self.l == 7:
            file_name = '../data/AGE_relationships_deals.txt'
        with open(file_name, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                user_id = int(parts[0])
                friends = list(map(int, parts[1:]))
                self.friend_ships[user_id] = friends
        for id in self.data['user_id']:
            if id not in self.friend_ships:
                self.friend_ships[id] = []


    def data_deal(self):
        self.data['group'] = self.data['categories'].apply(self.assign_group)
        self.data_id = self.data['user_id'].tolist()
        self.data_group = self.data['group'].tolist()
        self.decide_k_PR()
        self.read_deal_relationships()

    def Greedy_algorithm(self):
        for p in range(self.K):
            for id, group in zip(self.data_id,self.data_group):
                if self.group_count[group] == self.k_list[group]:
                    continue
                if id not in self.tracking_delta:
                    tmp_value = self.get_element_value(id)
                    self.tracking_delta[id] = tmp_value
                    if tmp_value > self.maximum_value[group]:
                        self.maximum_value[group] = tmp_value
                        self.maximum_value_id[group] = id
                        self.maximum_value_group[group] = group
                else:
                    if self.tracking_delta[id] > self.maximum_value[group]:
                        tmp_value = self.get_element_value(id)
                        self.tracking_delta[id] = tmp_value
                        if tmp_value > self.maximum_value[group]:
                            self.maximum_value[group] = tmp_value
                            self.maximum_value_id[group] = id
                            self.maximum_value_group[group] = group
            tuples = list(zip(self.maximum_value, self.maximum_value_id, self.maximum_value_group))
            sorted_data = sorted(tuples, key=lambda x: x[0], reverse=True)
            for value, id, group in sorted_data:
                if self.group_count[group] == self.k_list[group]:
                    continue
                self.group_count[group] += 1
                self.S.append(id)
                self.element_set.update(self.friend_ships[id])
                for g in range(self.l):
                    self.maximum_value[g] = -1
                    self.maximum_value_id[g] = -1
                    self.maximum_value_group[g] = -1
                break
            self.func_value = len(self.element_set)
        print("ops ",self.func_value)



    def get_element_value(self, id):
        element_set = intbitset(self.friend_ships[id])
        f_S1 = len(self.element_set | element_set)
        return f_S1 - self.func_value


    def main(self):
        self.data_deal()
        start_time = time.time()
        self.Greedy_algorithm()
        end_time = time.time()
        self.time = end_time - start_time

        file_name = "Greedy_" + "K=" + str(self.K) + "_age_PR_output.txt"
        with open(file_name, "w") as file:
            file.write("value: " + str(self.func_value)+"\n")
            file.write("time: " + str(self.time)+"\n")
            file.write(" ".join(map(str, self.S)))


if __name__ == "__main__":
    data,relationships = data_deal.read_AGE(-1)
    k_list = [100]
    for k in k_list:
        greedy = Greedy(data,relationships,7, k)
        greedy.main()

