import numpy as np
import json
#from create_options import check_and_write
import time
import tables
import os.path

def check_and_write(array, filename, NUM_ENTRIES):
    if not os.path.isfile(filename):
        f = tables.open_file(filename, mode='w')
        atom = tables.Float64Atom()
        array_c = f.create_earray(f.root, 'data', atom, (0, 8))
        print (array)
        array_c.append(array)
        f.close()
        print ("new file")
        return True
    f = tables.open_file(filename, mode='r')
    i = 0
    check = False
    for item in f.root.data[0]:
        c = np.array((f.root.data[i:i+NUM_ENTRIES,0:]))
        idx = np.where(abs((c[:,np.newaxis,:] - array)).sum(axis=2) == 0)
        i = i + NUM_ENTRIES
        if len(idx[0]) == NUM_ENTRIES:
            check = True
            break
    f.close()
    if check:
        print ("Duplicate Solution")
        return False
    else:
        print ("Unique Solution")
        f = tables.open_file(filename, mode='a')
        f.root.data.append(array)
        f.close()
        return True

def load_options(json_name):
    json1_file = open(json_name)
    json1_str = json1_file.read()
    json1_data = json.loads(json1_str)
    return json1_data

def create_dictionary(possible_options):
    for i in range(len(possible_options)):
        for j in range(len(possible_options[i])):
            if possible_options[i][j] == 0:
                possible_options[i][j] = 0.+0.j
            elif possible_options[i][j] == 1:
                possible_options[i][j] = 1.+0.j
            elif possible_options[i][j] == -1:
                possible_options[i][j] = -1.+0.j
            elif possible_options[i][j] % 2 == 0:
                possible_options[i][j] = 0. -1.j
            elif possible_options[i][j] % 3 == 0:
                possible_options[i][j] = 0. + 1.j
    top_dictionary = {}
    for option in possible_options:
        key1 = ""
        for entry in option:
            key1 += str(entry)
        top_dictionary[key1] = {}
        for option2 in possible_options:
            key2 = ""
            for entry2 in option2:
                key2 += str(entry2)
            top_dictionary[key1][key2] = []
            for i in range(len(option)):
                for j in range(len(option2)):
                    top_dictionary[key1][key2].append(option[i]*option2[j])
    return top_dictionary

def find_options(matrix_size, mirror):
    options_size = matrix_size ** 2
    options = []
    if mirror:
        for i in range(int((5 ** options_size))):
            rows = []
            for j in range(0, options_size):
                rows.append(1 - int(i % 5 ** (options_size - j) / (5 ** (options_size - (j + 1)))))
            # number = np.count_nonzero(rows)
            # if 0 < number < 5:
            options.append(rows)
        return options
    else:
        print(int((3 ** options_size) / 2))
        print(int((5 ** options_size) / 2))
        for i in range(int((5 ** options_size) / 2)):
            
            rows = []
            for j in range(0, options_size):
                rows.append(1 - int(i % 5 ** (options_size - j) / (5 ** (options_size - (j + 1)))))
            # number = np.count_nonzero(rows)
            # if 0 < number < 5:
            #     options.append(rows)
            options.append(rows)
        return options

def create_solution():
    c1 = np.array([[1.+0j], [0.+0.j], [0.+0.j], [0.+0.j],
                   [0.+0.j], [1.+0.j], [0.+0.j], [0.+0.j],
                   [0.+0.j], [0.+0.j], [0.+0.j], [0.+0.j],
                   [0.+0.j], [0.+0.j], [0.+0.j], [0.+0.j]])

    c2 = np.array([[0.+0.j], [0.+0.j], [1.+0.j], [0.+0.j],
                   [0.+0.j], [0.+0.j], [0.+0.j], [1.+0.j],
                   [0.+0.j], [0.+0.j], [0.+0.j], [0.+0.j],
                   [0.+0.j], [0.+0.j], [0.+0.j], [0.+0.j]])

    c3 = np.array([[0.+0.j], [0.+0.j], [0.+0.j], [0.+0.j],
                   [0.+0.j], [0.+0.j], [0.+0.j], [0.+0.j],
                   [1.+0.j], [0.+0.j], [0.+0.j], [0.+0.j],
                   [0.+0.j], [1.+0.j], [0.+0.j], [0.+0.j]])

    c4 = np.array([[0.+0.j], [0.+0.j], [0.+0.j], [0.+0.j],
                   [0.+0.j], [0.+0.j], [0.+0.j], [0.+0.j],
                   [0.+0.j], [0.+0.j], [1.+0.j], [0.+0.j],
                   [0.+0.j], [0.+0.j], [0.+0.j], [1.+0.j]])

    final_sol = np.concatenate((c1, c2, c3, c4), axis=1)
    return final_sol

class StrassenSearch:
    def __init__(self, number, dimensions, multiplications, mutation_rate, options_file):
        self.negative_imaginary = 0. - 1.j
        self.imaginary = 0.+1.j
        self.one = 1. + 0.j
        self.zero = 0. + 0.j
        self.neg_one = -1. + 0.j
        self.b_negative_imaginary = 0b10000
        self.b_imaginary = 0b01000
        self.b_one = 0b00100
        self.b_zero = 0b00010
        self.b_neg_one = 0b00001
        # Look to reduce this list
        self.improvement = [1]*number
        self.temp_cost_finder = [0]*number
        self.success = 0
        self.purge_rate = int(20)
        self.filename = '3by3imaginary.h5'
        self.prev_best_i1 = 1000
        self.prev_best_cost1 = 1000
        self.prev_best_i2 = 1000
        self.prev_best_cost2 = 1000
        self.num_of_pop = number
        self.best_cost = 0
        self.best_x = []
        self.x = []
        self.best_value = []
        self.final_best_value = []
        self.best_i = 0
        self.best_in_population = []
        self.count = 0
        self.running = 1
        # ###############################
        self.dimension = dimensions
        self.multiplication = multiplications
        self.mutation = mutation_rate
        self.value = []
        self.final_value = []
        self.population = []
        self.cost = []
        self.solution = create_solution()
        self.options = create_dictionary(find_options(self.dimension,True))
        print("Dictionary Initialized")
        # self.search_options = find_options(self.dimension, False)
        for i in range(self.num_of_pop):
            chromosome = self.create_chromosome()
            val = self.decode(chromosome)
            final_val = self.expand(val)
            fitness, temp_x = self.determine_fitness(final_val)
            self.value.append(val)
            self.final_value.append(final_val)
            self.population.append(chromosome)
            self.cost.append(fitness)
            self.x.append(temp_x)

    def expand(self, value):
        rows = self.dimension ** 3
        cols = self.multiplication
        final_value = []
        for i in range(cols):
            key1 = ""
            key2 = ""
            for j in range(int(rows/2)):
                key1 += str(value[j][i])
            for j in range(int(rows/2), rows):
                key2 += str(value[j][i])
            if key1 == '0j0j0j0j' or key2 == '0j0j0j0j':
                final_value.append([0]*16)
            else:
                final_value.append(self.options[key1][key2])
        return np.array(final_value).T
    def local_search(self, value, final_value, x, fitness):
        # TODO: randomize starting position of local search
        
        best_cost = fitness
        best_val = value
        best_final_value = final_value
        best_x = x
        for i in range(0, len(value), 1):
            for j in range(0, len(value[0]), 1):
                val1 = np.copy(value)
                val2 = np.copy(value)
                val3 = np.copy(value)
                val4 = np.copy(value)
                if value[i][j] == self.one:
                    val1[i][j] = self.zero
                    val2[i][j] = self.neg_one
                    val3[i][j] = self.negative_imaginary
                    val4[i][j] = self.imaginary
                    final_val1 = self.expand(val1)
                    final_val2 = self.expand(val2)
                    final_val3 = self.expand(val3)
                    final_val4 = self.expand(val4)
                    cost1, x1 = self.determine_fitness(final_val1)
                    cost2, x2 = self.determine_fitness(final_val2)
                    cost3, x3 = self.determine_fitness(final_val3)
                    cost4, x4 = self.determine_fitness(final_val4)
                elif value[i][j] == self.zero:
                    val1[i][j] = self.one
                    val2[i][j] = self.neg_one
                    val3[i][j] = self.negative_imaginary
                    val4[i][j] = self.imaginary
                    final_val1 = self.expand(val1)
                    final_val2 = self.expand(val2)
                    final_val3 = self.expand(val3)
                    final_val4 = self.expand(val4)
                    cost1, x1 = self.determine_fitness(final_val1)
                    cost2, x2 = self.determine_fitness(final_val2)
                    cost3, x3 = self.determine_fitness(final_val3)
                    cost4, x4 = self.determine_fitness(final_val4)
                elif value[i][j] == self.imaginary:
                    val1[i][j] = self.one
                    val2[i][j] = self.neg_one
                    val3[i][j] = self.negative_imaginary
                    val4[i][j] = self.zero
                    final_val1 = self.expand(val1)
                    final_val2 = self.expand(val2)
                    final_val3 = self.expand(val3)
                    final_val4 = self.expand(val4)
                    cost1, x1 = self.determine_fitness(final_val1)
                    cost2, x2 = self.determine_fitness(final_val2)
                    cost3, x3 = self.determine_fitness(final_val3)
                    cost4, x4 = self.determine_fitness(final_val4)
                elif value[i][j] == self.negative_imaginary:
                    val1[i][j] = self.one
                    val2[i][j] = self.neg_one
                    val3[i][j] = self.zero
                    val4[i][j] = self.imaginary
                    final_val1 = self.expand(val1)
                    final_val2 = self.expand(val2)
                    final_val3 = self.expand(val3)
                    final_val4 = self.expand(val4)
                    cost1, x1 = self.determine_fitness(final_val1)
                    cost2, x2 = self.determine_fitness(final_val2)
                    cost3, x3 = self.determine_fitness(final_val3)
                    cost4, x4 = self.determine_fitness(final_val4)
                else:
                    val1[i][j] = self.one
                    val2[i][j] = self.neg_one
                    val3[i][j] = self.zero
                    val4[i][j] = self.imaginary
                    final_val1 = self.expand(val1)
                    final_val2 = self.expand(val2)
                    final_val3 = self.expand(val3)
                    final_val4 = self.expand(val4)
                    cost1, x1 = self.determine_fitness(final_val1)
                    cost2, x2 = self.determine_fitness(final_val2)
                    cost3, x3 = self.determine_fitness(final_val3)
                    cost4, x4 = self.determine_fitness(final_val4)
                if cost1 > cost2:
                    winner_cost1 = cost1
                    winner_val1 = val1
                    winner_final_value1 = final_val1
                    winner_x1 = x1
                else:
                    winner_cost1 = cost2
                    winner_val1 = val2
                    winner_final_value1 = final_val2
                    winner_x1 = x2
                if cost3 > cost4:
                    winner_cost2 = cost3
                    winner_val2 = val3
                    winner_final_value2 = final_val3
                    winner_x2 = x3
                else:
                    winner_cost2 = cost4
                    winner_val2 = val4
                    winner_final_value2 = final_val4
                    winner_x2 = x4
                if winner_cost1 > winner_cost2:
                    win_cost = winner_cost1
                    win_val = winner_val1
                    win_final_val = winner_final_value1
                    win_x = winner_x1
                else:
                    win_cost = winner_cost2
                    win_val = winner_val2
                    win_final_val = winner_final_value2
                    win_x = winner_x2

                if win_cost > best_cost:
                    best_cost = win_cost
                    best_val = win_val
                    best_final_value = win_final_val
                    best_x = win_x
                    return best_val, best_final_value, best_x, best_cost
        return best_val, best_final_value, best_x, best_cost


    def create_chromosome(self):
        # is not rows now- is in the form [a1 a2 a3 a4] [ b1 b2 b3 b4 ]
        rows = self.dimension ** 3
        cols = self.multiplication
        chromosome = 0b0
        for i in range(rows * cols):
            choice = np.random.randint(0, 34)
            chromosome = chromosome << 5
            # starting with more zeros seems to work faster
            if choice < 5:
                chromosome = chromosome | self.b_one
            elif choice < 10:
                chromosome = chromosome | self.b_imaginary
            elif choice < 25:
                chromosome = chromosome | self.b_zero
            elif choice < 30:
                chromosome = chromosome | self.b_negative_imaginary
            else:
                chromosome = chromosome | self.b_neg_one
        return chromosome
    # def lookup_final_value(self, value):
    #

    def crossover(self, bin_a, bin_b):
        rows = self.dimension ** 3
        cols = self.multiplication
        point = np.random.randint(0, rows * cols)
        mask_a = 0b0
        mask_b = 0b0
        for i in range(0, rows * cols):
            if i < point:
                mask_a = mask_a << 5
                mask_b = mask_b << 5
                mask_a = mask_a | 0b11111

            else:
                mask_b = mask_b << 5
                mask_a = mask_a << 5
                mask_b = mask_b | 0b11111
        child1 = (bin_a & mask_a) | (bin_b & mask_b)
        child2 = (bin_b & mask_a) | (bin_a & mask_b)
        return child1, child2

    def mutate(self, binary, rate):
        rows = self.dimension ** 3
        cols = self.multiplication
        # rows = self.dimension ** 3
        # cols = self.multiplication
        mask_a = 0b0
        mask_b = 0b0
        mask_negative_imaginary = self.b_negative_imaginary << 5 * (self.multiplication * (self.dimension ** 3) - 1)
        mask_imaginary = self.b_imaginary << 5 * (self.multiplication * (self.dimension ** 3) - 1)
        mask_one = self.b_one << 5 * (self.multiplication * (self.dimension ** 3) - 1)
        mask_zero = self.b_zero << 5 * (self.multiplication * (self.dimension ** 3) - 1)
        mask_neg_one = self.b_neg_one << 5 * (self.multiplication * (self.dimension ** 3) - 1)
        for i in range(rows * cols):
            choice_a = np.random.randint(0, 100)
            mask_a = mask_a << 5
            mask_b = mask_b << 5
            if choice_a < rate:
                is_set = False
                while not is_set: 
                    choice_b = np.random.randint(0, 66)
                    if (choice_b < 22 or choice_b > 55) and not (binary & mask_one):
                        mask_b = mask_b | self.b_one
                        is_set = True
                    elif (21 < choice_b < 56) and not (binary & mask_zero):
                        mask_b = mask_b | self.b_zero
                        is_set = True
                    elif (21 < choice_b < 56) and not (binary & mask_negative_imaginary):
                        mask_b = mask_b | self.b_negative_imaginary
                        is_set = True
                    elif (21 < choice_b < 56) and not (binary & mask_imaginary):
                        mask_b = mask_b | self.b_imaginary
                        is_set = True
                    elif (21 < choice_b < 56) and not (binary & mask_neg_one):
                        mask_b = mask_b | self.b_neg_one
                        is_set = True
            else:
                mask_a = mask_a | 0b11111
            mask_negative_imaginary >> 5
            mask_imaginary >> 5 
            mask_one = mask_one >> 5
            mask_zero = mask_zero >> 5
        binary = binary & mask_a
        binary = binary | mask_b
        return binary

    def decode(self, binary):
        rows = self.dimension ** 3
        cols = self.multiplication
        value = []
        for i in range(rows):
            temp = []
            for j in range(cols):
                if binary & self.b_negative_imaginary:
                    temp.append(self.negative_imaginary)
                elif binary & self.b_imaginary:
                    temp.append(self.imaginary)
                elif  binary & self.b_one:
                    temp.append(self.one)
                elif binary & self.b_zero:
                    temp.append(self.zero)
                else:
                    temp.append(self.neg_one)
                binary = binary >> 5
            value.append(temp)
        return np.array(value)

    def encode(self, value):
        val = value
        
        rows = self.dimension ** 3
        cols = self.multiplication
        bins = 0b0
        count =0
        for i in range(rows):
            for j in range(cols):
                temp_bins = 0b0
                if val[i][j] == self.negative_imaginary:
                    temp_bins = self.b_negative_imaginary
                    temp_bins = temp_bins << 5*(count)
                elif val[i][j] == self.imaginary:
                    temp_bins = self.b_imaginary
                    temp_bins = temp_bins << 5*(count)
                elif val[i][j] == self.one:
                    temp_bins = self.b_one
                    temp_bins = temp_bins << 5*(count)
                elif val[i][j] == self.zero:
                    temp_bins = self.b_zero
                    temp_bins = temp_bins << 5*(count)
                else:
                    temp_bins = self.b_neg_one
                    temp_bins = temp_bins << 5*(count)
                count += 1
                bins = bins | temp_bins
        return bins

    def determine_fitness(self, value):
        # solution = create_sols2()
        a = np.dot(value, value.T)
        b = np.linalg.pinv(a)
        c = np.dot(value.T, b)
        d = np.dot(value.T, self.solution)
        e = np.dot(c.T, d)
        f = np.subtract(e, self.solution)
        g = np.dot(f, f.T)
        h = np.trace(g)
        absolute = np.absolute(h)
        return 1 / (1 + absolute*100000000000 ),d

    def check_for_improvement(self):
        if self.count % self.num_of_pop*self.mutation == 0:
            if (self.best_i == self.prev_best_i2) and (self.best_cost == self.prev_best_cost2):
                self.purge(self.purge_rate)
                print("repeat")
                print(self.count)
            elif (self.best_i == self.prev_best_i1) and (self.best_cost == self.prev_best_cost1):
                # self.running = 0
                #     self.success = 1
                self.prev_best_i1 = 1000
                self.prev_best_cost1 = 1000
                self.prev_best_i2 = self.best_i
                self.prev_best_cost2 = self.best_cost
                # self.temp_val = self.best_value
                temp_cost = self.best_cost
                # self.best_value, self.final_best_value, self.best_x, self.best_cost = self.final_search(
                #                                                                                   self.best_value,
                #                                                                                   self.final_best_value,
                #                                                                                   self.best_x,
                #                                                                                   self.best_cost)
                if temp_cost != self.best_cost:
                    self.best_in_population = self.encode(self.best_value)
                    self.population[self.best_i] = self.best_in_population
                    self.cost[self.best_i] = self.best_cost
                    self.x[self.best_i] = self.best_x
                    self.value[self.best_i] = self.best_value
                    self.final_value[self.best_i] = self.final_best_value
                if self.best_cost == 1:
                    print (self.best_value)
                    check_and_write(self.best_value.T, self.filename, self.multiplication)
                    self.running = 0
                    self.success = 1
                #     #self.purge(1)
            else:
                self.prev_best_i1 = self.best_i
                self.prev_best_cost1 = self.best_cost
                # self.temp_val = self.best_value
                # self.temp_cost = self.best_cost
                self.best_value, self.final_best_value, self.best_x, self.best_cost = self.local_search(
                                                                                                  self.best_value,
                                                                                                  self.final_best_value,
                                                                                                  self.best_x,
                                                                                                  self.best_cost)
                self.best_in_population = self.encode(self.best_value)
                self.population[self.best_i] = self.best_in_population
                self.cost[self.best_i] = self.best_cost
                self.x[self.best_i] = self.best_x
                self.value[self.best_i] = self.best_value
                self.final_value[self.best_i] = self.final_best_value
                if self.best_cost == 1:
                    print (self.best_value)
                    check_and_write(self.best_value.T, self.filename, self.multiplication)
                    self.running = 0
                    self.success = 1
                    # self.purge(1)
                print (self.best_cost)

    def purge(self, purge_rate):
        self.population[self.best_i] = self.create_chromosome()
        self.best_in_population = self.population[self.best_i]
        self.value[self.best_i] = self.decode(self.best_in_population)
        self.final_value[self.best_i] = self.expand(self.value[self.best_i])
        self.cost[self.best_i], self.x[self.best_i] = self.determine_fitness(self.final_value[self.best_i])
        self.best_cost = self.cost[self.best_i]
        self.best_x = self.x[self.best_i]
        self.best_value = self.value[self.best_i]
        self.final_best_value = self.final_value[self.best_i]
        if purge_rate > 1:
            items_for_purge = np.random.choice(self.num_of_pop, purge_rate - 1)
            for i in range(len(items_for_purge)-1):
                if items_for_purge[i] != self.best_i:
                    chromosome = self.create_chromosome()
                    val = self.decode(chromosome)
                    final_val = self.expand(val)
                    fitness, temp_x = self.determine_fitness(final_val)
                    self.value[items_for_purge[i]] = val
                    self.final_value[items_for_purge[i]] = final_val
                    self.population[items_for_purge[i]] = chromosome
                    self.cost[items_for_purge[i]] = fitness
                    self.x[items_for_purge[i]] = temp_x

    def simple_search(self):
        while self.count < 700 and self.running:
            pop2 = np.copy(self.population)
            for i in range(self.num_of_pop):
                # trial_a = 0b0
                # trial_b = 0b0
                while True:
                    a = int(np.random.random() * self.num_of_pop)
                    if a != i:
                        break
                trial_a, trial_b = self.crossover(pop2[i], pop2[a])
                trial_a = self.mutate(trial_a, self.mutation)  # int(best_cost*100)
                trial_b = self.mutate(trial_b, self.mutation)  # int(best_cost*100)
                val_a = self.decode(trial_a)
                final_val_a = self.expand(val_a)
                val_b = self.decode(trial_b)
                final_val_b = self.expand(val_b)
                cost_a, temp_x_a = self.determine_fitness(final_val_a)
                cost_b, temp_x_b = self.determine_fitness(final_val_b)
                if cost_a > cost_b:
                    winner = trial_a
                    better_cost = cost_a
                    better_x = temp_x_a
                    better_val = val_a
                    better_final_val = final_val_a
                else:
                    winner = trial_b
                    better_cost = cost_b
                    better_x = temp_x_b
                    better_val = val_b
                    better_final_val = final_val_b
                if better_cost > self.cost[i]:

                    pop2[i] = winner
                    self.cost[i] = better_cost
                    self.x[i] = better_x
                    self.value[i] = better_val
                    self.final_value[i] = better_final_val
                    # update best cost
                if self.cost[i] > self.best_cost:
                    self.best_i = i
                    self.best_cost = self.cost[i]
                    self.best_x = self.x[i]
                    self.best_in_population = self.population[i]
                    self.best_value = self.value[i]
                    self.final_best_value = self.final_value[i]

            for i in range(self.num_of_pop):
                if (not self.improvement[i]) and self.temp_cost_finder[i] == self.cost[i]:
                    pass
                else:

                    temp_cost = self.cost[i]

                    self.value[i], self.final_value[i], self.x[i], self.cost[i] = self.local_search(self.value[i],
                                                                                                    self.final_value[i],
                                                                                                    self.x[i],
                                                                                                    self.cost[i])
                    if temp_cost == self.cost[i]:
                        self.improvement[i] = 0
                        self.temp_cost_finder[i] = self.cost[i]
                    else:
                        self.improvement[i] = 1

                    if self.cost[i] > self.best_cost:
                        self.best_i = i
                        self.best_cost = self.cost[i]
                        self.best_x = self.x[i]
                        self.best_in_population = self.population[i]
                        self.best_value = self.value[i]
                        self.final_best_value = self.final_value[i]

            self.population = np.copy(pop2)
            self.count += 1
            self.check_for_improvement()


if __name__ == "__main__":
    file_name = '2by2data.json'
    # start = time.time()

    # copy here
    start = time.time()
    while True:
        pop = 40  # np.random.randint(10, 21)
        m = 14  # np.random.randint(35, 45)

        first = StrassenSearch(pop, 2, 7, m, file_name)
        first.simple_search()

        if first.success:
            end = time.time()
            total_time = end - start
            results = "{},{},{} \n".format(total_time, pop, m)
            print("the final running time is {}".format(end - start))
            print("the parameters are {} and {}".format(pop, m))
            fd = open('parameters.csv', 'a')
            fd.write(results)
            fd.close()
            start = time.time()
        else:
            print ("fail")
