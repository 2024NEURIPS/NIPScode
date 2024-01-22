import copy
import time
import data_deal
import random
from intbitset import intbitset


class Multi_Stream():
    def __init__(self,data,relationships,l,K,beta):
        self.data = data
        self.relationships = relationships
        self.max_num = max(self.data.shape[0],self.relationships['user_id2'].max()) + 1
        self.l = l
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
        self.element_set = intbitset()
        self.data_id = []
        self.data_group = []
        self.counter = 1
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




    def MS_algorithm(self):
        start_time = time.time()
        self.sampling_Ri()
        boundary = self.alpha / self.K
        while (1-self.alpha)**((1+self.alpha)**self.counter) > boundary  and len(self.S) != self.K:
            # print((1-self.alpha)**((1+self.alpha)**self.counter))
            # print(self.counter," ",len(self.S)," ",self.func_value ," ",len(self.element_set))
            max_value = self.ops_stream()
            self.select_neighbor(max_value)
            self.counter += 1
        self.Greedy_algorithm(self.Ri,self.Ri_group)
        # print("ops ",self.func_value)
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
        for id, group in zip(self.data_id,self.data_group):
            if self.group_count[group] == self.k_list[group]:
                continue
            if id not in self.tracking_delta:
                tmp_value = self.get_element_value(id)
                self.tracking_delta[id] = tmp_value
            else:
                if self.tracking_delta[id] <= max_value:
                    continue
                tmp_value = self.get_element_value(id)
                self.tracking_delta[id] = tmp_value
            if tmp_value > max_value:
                max_id = id
                max_value = tmp_value
                max_group = group
        self.S.append(max_id)
        self.S_group.append(max_group)
        self.element_set.update(self.friend_ships[max_id])
        self.func_value = len(self.element_set)
        self.group_count[max_group] += 1
        return max_value

    def select_neighbor(self,max_value):
        for id, group in zip(self.data_id,self.data_group):
            if self.group_count[group] == self.k_list[group]:
                continue
            if self.tracking_delta[id] <= max_value * (1-self.alpha):
                continue
            tmp_value = self.get_element_value(id)
            self.tracking_delta[id] = tmp_value
            if tmp_value > max_value * (1 - self.alpha):
                self.S.append(id)
                self.S_group.append(group)
                self.element_set.update(self.friend_ships[id])
                self.func_value = len(self.element_set)
                self.group_count[group] += 1


    def Greedy_algorithm(self, extra_id, extra_group):
        greedy_count = self.K - len(self.S)
        print("greedy",greedy_count)
        maximum_value = [-1] * self.l
        maximum_value_id = [-1] * self.l
        maximum_value_group = [-1] * self.l
        for p in range(greedy_count):
            for id, group in zip(extra_id, extra_group):
                if id not in self.tracking_delta:
                    tmp_value = self.get_element_value(id)
                    self.tracking_delta[id] = tmp_value
                    if tmp_value > maximum_value[group]:
                        maximum_value[group] = tmp_value
                        maximum_value_id[group] = id
                        maximum_value_group[group] = group
                else:
                    if self.tracking_delta[id] > maximum_value[group]:
                        tmp_value = self.get_element_value(id)
                        self.tracking_delta[id] = tmp_value
                        if tmp_value > maximum_value[group]:
                            maximum_value[group] = tmp_value
                            maximum_value_id[group] = id
                            maximum_value_group[group] = group
            tuples = list(zip(maximum_value, maximum_value_id, maximum_value_group))
            sorted_data = sorted(tuples, key=lambda x: x[0], reverse=True)
            for value, id, group in sorted_data:
                if self.group_count[group] == self.k_list[group]:
                    continue
                self.group_count[group] += 1
                self.S.append(id)
                self.element_set.update(self.friend_ships[id])
                for g in range(self.l):
                    maximum_value[g] = -1
                    maximum_value_id[g] = -1
                    maximum_value_group[g] = -1
                break
            self.func_value = len(self.element_set)

    def get_element_value(self, id):
        element_set = intbitset(self.friend_ships[id])
        f_S1 = len(self.element_set | element_set)
        return f_S1 - len(self.element_set)


    def main(self):
        self.data_deal()
        self.MS_algorithm()
        file_name = "MS_" + "K=" + str(self.K) + "_age_PR_output.txt"
        with open(file_name, "w") as file:
            file.write("value: " + str(self.func_value)+"\n")
            file.write("time: " + str(self.time)+"\n")
            file.write(" ".join(map(str, self.S)))


if __name__ == "__main__":
    data,relationships = data_deal.read_AGE(-1)
    k_list = [100,200,300,400,500,600,700,800,900,1000]
    for k in k_list:
        ms = Multi_Stream(data,relationships,7,k,40)
        ms.main()
