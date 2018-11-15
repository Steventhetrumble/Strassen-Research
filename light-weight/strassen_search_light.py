import numpy as np
import json
from create_options import check_and_write
import time


def load_options(json_name):
    json1_file = open(json_name)
    json1_str = json1_file.read()
    json1_data = json.loads(json1_str)
    return json1_data


def create_solution():
    c1 = np.array([[1], [0], [0], [0],
                   [0], [1], [0], [0],
                   [0], [0], [0], [0],
                   [0], [0], [0], [0]])

    c2 = np.array([[0], [0], [1], [0],
                   [0], [0], [0], [1],
                   [0], [0], [0], [0],
                   [0], [0], [0], [0]])

    c3 = np.array([[0], [0], [0], [0],
                   [0], [0], [0], [0],
                   [1], [0], [0], [0],
                   [0], [1], [0], [0]])

    c4 = np.array([[0], [0], [0], [0],
                   [0], [0], [0], [0],
                   [0], [0], [1], [0],
                   [0], [0], [0], [1]])

    final_sol = np.concatenate((c1, c2, c3, c4), axis=1)
    return final_sol


class StrassenSearch:
    def __init__(self, number, dimensions, multiplications, mutation, purge_rate, solution, options):
        # Look to reduce this list
        self.improvement = [1]*number
        self.temp_cost_finder = [0]*number
        self.success = 0
        self.purge_rate = int(number * purge_rate)
        self.filename = '2by2.h5'
        self.prev_best_i1 = 1000
        self.prev_best_cost1 = 1000
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
        self.mutation = mutation
        self.value = []
        self.final_value = []
        self.population = []
        self.cost = []
        self.solution = solution
        self.options = options
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
            if key1 == '0000' or key2 == '0000':
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
                if value[i][j] == 1:
                    val1[i][j] = 0
                    val2[i][j] = -1
                    final_val1 = self.expand(val1)
                    final_val2 = self.expand(val2)
                    cost1, x1 = self.determine_fitness(final_val1)
                    cost2, x2 = self.determine_fitness(final_val2)
                elif value[i][j] == 0:
                    val1[i][j] = 1
                    val2[i][j] = -1
                    final_val1 = self.expand(val1)
                    final_val2 = self.expand(val2)
                    cost1, x1 = self.determine_fitness(final_val1)
                    cost2, x2 = self.determine_fitness(final_val2)
                else:
                    val1[i][j] = 0
                    val2[i][j] = 1
                    final_val1 = self.expand(val1)
                    final_val2 = self.expand(val2)
                    cost1, x1 = self.determine_fitness(final_val1)
                    cost2, x2 = self.determine_fitness(final_val2)
                if cost1 > cost2:
                    winner_cost = cost1
                    winner_val = val1
                    winner_final_value = final_val1
                    winner_x = x1
                else:
                    winner_cost = cost2
                    winner_val = val2
                    winner_final_value = final_val2
                    winner_x = x2
                if winner_cost > best_cost:
                    best_cost = winner_cost
                    best_val = winner_val
                    best_final_value = winner_final_value
                    best_x = winner_x
                    return best_val, best_final_value, best_x, best_cost
        return best_val, best_final_value, best_x, best_cost

    def create_chromosome(self):
        one = 0b100
        zero = 0b010
        neg_one = 0b001
        # is not rows now- is in the form [a1 a2 a3 a4] [ b1 b2 b3 b4 ]
        rows = self.dimension ** 3
        cols = self.multiplication
        chromosome = 0b0
        for i in range(rows * cols):
            choice = np.random.randint(0, 34)
            chromosome = chromosome << 3
            # starting with more zeros seems to work faster
            if choice < 6:
                chromosome = chromosome | one
            elif choice < 29:
                chromosome = chromosome | zero
            else:
                chromosome = chromosome | neg_one
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
                mask_a = mask_a << 3
                mask_b = mask_b << 3
                mask_a = mask_a | 0b111

            else:
                mask_b = mask_b << 3
                mask_a = mask_a << 3
                mask_b = mask_b | 0b111
        child1 = (bin_a & mask_a) | (bin_b & mask_b)
        child2 = (bin_b & mask_a) | (bin_a & mask_b)
        return child1, child2

    def mutate(self, binary, rate):
        rows = self.dimension ** 3
        cols = self.multiplication
        mask_a = 0b0
        mask_b = 0b0
        one = 0b100
        zero = 0b010
        neg_one = 0b001
        mask_one = one << 3 * (self.multiplication * (self.dimension ** 3) - 1)
        mask_zero = zero << 3 * (self.multiplication * (self.dimension ** 3) - 1)
        for i in range(rows * cols):
            choice_a = np.random.randint(0, 100)
            mask_a = mask_a << 3
            mask_b = mask_b << 3
            if choice_a < rate:
                choice_b = np.random.randint(0, 66)
                if (choice_b < 22 or choice_b > 55) and not (binary & mask_one):
                    mask_b = mask_b | one
                elif (21 < choice_b < 56) and not (binary & mask_zero):
                    mask_b = mask_b | zero
                else:
                    mask_b = mask_b | neg_one
            else:
                mask_a = mask_a | 0b111
            mask_one = mask_one >> 3
            mask_zero = mask_zero >> 3
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
                if binary & 0b100:
                    temp.append(1)
                elif binary & 0b010:
                    temp.append(0)
                else:
                    temp.append(-1)
                binary = binary >> 3
            value.append(temp)
        return np.array(value)

    def encode(self, value):
        val = value
        rows = self.dimension ** 3
        cols = self.multiplication
        bins = 0b0
        for i in range(rows):
            for j in range(cols):
                bins = bins << 3
                if val[i][j] == 1:
                    bins = bins | 0b100
                elif val[i][j] == 0:
                    bins = bins | 0b010
                else:
                    bins = bins | 0b001
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
        return 1 / (1 + h), d

    def check_for_improvement(self):
        # if self.count % self.num_of_pop*self.mutation == 0:
        if (self.best_i == self.prev_best_i1) and (self.best_cost == self.prev_best_cost1):
            self.purge(self.purge_rate)
            # print "purge - " + str(self.purge_rate)
            # print self.count
        else:
            self.prev_best_i1 = self.best_i
            self.prev_best_cost1 = self.best_cost
            if self.best_cost == 1:
                print (self.best_value)
                check_and_write(self.best_value.T, self.filename, self.multiplication)
                self.running = 0
                self.success = 1
                # self.purge(1)
            # print self.best_cost

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

    def simple_search(self, number_of_runs):
        while self.count < number_of_runs and self.running:
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
    sol = create_solution()
    option = load_options(file_name)
    pop = 40  # np.random.randint(10, 21)
    mutation_rate = 14  # np.random.randint(35, 45)
    runs = 240
    purge = .25
    # copy here
    start = time.time()
    while True:
        # purge rate is the amount of chromosomes (randomly selected) reinitialized if no improvement is made
        # if the number of runs is met then the entire population is reset

        first = StrassenSearch(pop, 2, 7, mutation_rate, purge, sol, option)
        first.simple_search(runs)

        if first.success:
            end = time.time()
            total_time = end - start
            results = "{},{},{},{},{} \n".format(total_time, pop, mutation_rate, purge, runs)
            print("the final running time is {}".format(end - start))
            print("the parameters are pop:{} mutation:{} purge: {} runs: {}".format(pop, mutation_rate, purge, runs))
            fd = open('parameters.csv', 'a')
            fd.write(results)
            fd.close()
            start = time.time()
        else:
            print ("-restart-")


