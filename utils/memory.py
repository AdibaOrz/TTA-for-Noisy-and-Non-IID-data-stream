import random

class PBRS():
    def __init__(self, capacity, num_class):
        self.data = [[[],[],[]] for _ in range(num_class)]
        self.counter = [0] * num_class
        self.capacity = capacity
        self.num_class = num_class

    def get_memory(self):
        tmp = [[], [], []]
        for data in self.data:
            feats, cls, do = data
            tmp[0].extend(feats)
            tmp[1].extend(cls)
            tmp[2].extend(do)
        return tmp

    def get_occupancy(self):
        occupancy = 0
        for data in self.data:
            occupancy += len(data[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy = [0] * self.num_class
        for i, data in enumerate(self.data):
            occupancy[i] = len(data[0])
        return occupancy

    def get_largest_classes(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_occupancy = max(occupancy_per_class)
        largest = []
        for i, occupancy in enumerate(occupancy_per_class):
            if occupancy == max_occupancy:
                largest.append(i)
        return largest

    def add_instance(self, instance):
        feat, cls, do = instance
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            self.data[cls][0].append(feat)
            self.data[cls][1].append(cls)
            self.data[cls][2].append(do)

    def remove_instance(self, cls):
        largest = self.get_largest_classes()
        if cls not in largest:
            largest = random.choice(largest)
            vict = random.randrange(0, len(self.data[largest][0]))

            self.data[largest][0].pop(vict)
            self.data[largest][1].pop(vict)
            self.data[largest][2].pop(vict)
        else:
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                vict = random.randrange(0, len(self.data[cls][0]))

                self.data[cls][0].pop(vict)
                self.data[cls][1].pop(vict)
                self.data[cls][2].pop(vict)
            else:
                return False
        return True
    
    def reset_memory(self):
        self.data = [[[],[],[]] for _ in range(self.num_class)]
        self.counter = [0] * self.num_class

class FIFO():
    def __init__(self, capacity):
        self.data = [[],[],[]]
        self.capacity = capacity

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i in range(len(instance)):
            self.data[i].append(instance[i])

    def remove_instance(self):
        for i in range(len(self.data)):
            self.data[i].pop(0)
    
    def reset_memory(self):
        self.data = [[[],[],[]] for _ in range(self.num_class)]