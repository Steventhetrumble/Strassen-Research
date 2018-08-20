import tables
import numpy as np
import os.path
import json


def check_and_write(array, filename, NUM_ENTRIES):
    if not os.path.isfile(filename):
        f = tables.open_file(filename, mode='w')
        atom = tables.Float64Atom()
        array_c = f.create_earray(f.root, 'data', atom, (0, 8))
        print(array)
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


def create_dictionary(options):
    top_dictionary = {}
    for option in options:
        key1 = ""
        for entry in option:
            key1 += str(entry)
        top_dictionary[key1] = {}
        for option2 in options:
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

def create_chromosome(dimension , multiplication):
        negative_imaginary = 0b10000
        imaginary = 0b01000
        one = 0b00100
        zero = 0b00010
        neg_one = 0b00001

        # is not rows now- is in the form [a1 a2 a3 a4] [ b1 b2 b3 b4 ]
        # rows = self.dimension ** 3
        # cols = self.multiplication
        rows = dimension ** 3
        cols = multiplication
        chromosome = 0b0
        for i in range(rows * cols):
            choice = np.random.randint(0, 34)
            chromosome = chromosome << 5
            # starting with more zeros seems to work faster
            if choice < 5:
                chromosome = chromosome | one
            elif choice < 10:
                chromosome = chromosome | imaginary
            elif choice < 25:
                chromosome = chromosome | zero
            elif choice < 30:
                chromosome = chromosome | negative_imaginary
            else:
                chromosome = chromosome | neg_one
        return chromosome
    # def lookup_final_value(self, value):
    #

    
def decode(binary, dimension, multiplication):
     
    negative_imaginary = 0. - 1.j
    imaginary = 0. + 1.j
    one = 1. + 0.j
    zero = 0. + 0.j
    neg_one = -1. + 0.j
    b_negative_imaginary = 0b10000
    b_imaginary = 0b01000
    b_one = 0b00100
    b_zero = 0b00010
    b_neg_one = 0b00001
    # rows = self.dimension ** 3
    # cols = self.multiplication
    rows = dimension ** 3
    cols = multiplication
    value = []
    for i in range(rows):
        temp = []
        for j in range(cols):
            if binary & b_negative_imaginary:
                temp.append(negative_imaginary)
            elif binary & b_imaginary:
                temp.append(imaginary)
            elif  binary & b_one:
                temp.append(one)
            elif binary & b_zero:
                temp.append(zero)
            else:
                temp.append(neg_one)
            binary = binary >> 5
        value.append(temp)
    return np.array(value)


def expand(value, dimension, multiplication, options):
    
    # rows = self.dimension ** 3
    # cols = self.multiplication
    rows = dimension ** 3
    cols = multiplication
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
            final_value.append(options[key1][key2])
    return np.array(final_value).T

def determine_fitness(value, solution):
    # solution = create_sols2()
    a = np.dot(value, value.T)
    b = np.linalg.pinv(a)
    c = np.dot(value.T, b)
    d = np.dot(value.T, solution)
    e = np.dot(c.T, d)
    f = np.subtract(e, solution)
    g = np.dot(f, f.T)
    h = np.trace(g)
    print(h)
    abssolute = np.absolute(h)
    return 1 / (1 + abssolute*10000000 ),d

def encode(value, dimension, multiplication):
    val = value
    negative_imaginary = 0. - 1.j
    imaginary = 0.+1.j
    one = 1. + 0.j
    zero = 0. + 0.j
    neg_one = -1. + 0.j
    b_negative_imaginary = 0b10000
    b_imaginary = 0b01000
    b_one = 0b00100
    b_zero = 0b00010
    b_neg_one = 0b00001
    # rows = self.dimension ** 3
    # cols = self.multiplication
    rows = dimension ** 3
    cols = multiplication
    bins = 0b0
    count =0
    for i in range(rows):
        for j in range(cols):
            
            print(val[i][j])
            temp_bins = 0b0
            if val[i][j] == negative_imaginary:
                temp_bins = b_negative_imaginary
                temp_bins = temp_bins << 5*(count)
            elif val[i][j] == imaginary:
                temp_bins = b_imaginary
                temp_bins = temp_bins << 5*(count)
            elif val[i][j] == one:
                temp_bins = b_one
                temp_bins = temp_bins << 5*(count)
            elif val[i][j] == zero:
                temp_bins = b_zero
                temp_bins = temp_bins << 5*(count)
            else:
                temp_bins = b_neg_one
                temp_bins = temp_bins << 5*(count)
            count += 1
            bins = bins | temp_bins
    return bins


def crossover(bin_a, bin_b,dimension, multiplication):
    rows = dimension ** 3
    cols = multiplication
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

def mutate(binary, rate, dimension, multiplication):
    rows = dimension ** 3
    cols = multiplication
    # rows = self.dimension ** 3
    # cols = self.multiplication
    mask_a = 0b0
    mask_b = 0b0
    b_negative_imaginary = 0b10000
    b_imaginary = 0b01000
    b_one = 0b00100
    b_zero = 0b00010
    b_neg_one = 0b00001
    mask_negative_imaginary = b_negative_imaginary << 5 * (multiplication * (dimension ** 3) - 1)
    mask_imaginary = b_imaginary << 5 * (multiplication * (dimension ** 3) - 1)
    mask_one = b_one << 5 * (multiplication * (dimension ** 3) - 1)
    mask_zero = b_zero << 5 * (multiplication * (dimension ** 3) - 1)
    mask_neg_one = b_neg_one << 5 * (multiplication * (dimension ** 3) - 1)
    for i in range(rows * cols):
        choice_a = np.random.randint(0, 100)
        mask_a = mask_a << 5
        mask_b = mask_b << 5
        if choice_a < rate:
            is_set = False
            while not is_set: 
                choice_b = np.random.randint(0, 66)
                if (choice_b < 22 or choice_b > 55) and not (binary & mask_one):
                    mask_b = mask_b | b_one
                    is_set = True
                elif (21 < choice_b < 56) and not (binary & mask_zero):
                    mask_b = mask_b | b_zero
                    is_set = True
                elif (21 < choice_b < 56) and not (binary & mask_negative_imaginary):
                    mask_b = mask_b | b_negative_imaginary
                    is_set = True
                elif (21 < choice_b < 56) and not (binary & mask_imaginary):
                    mask_b = mask_b | b_imaginary
                    is_set = True
                elif (21 < choice_b < 56) and not (binary & mask_neg_one):
                    mask_b = mask_b | b_neg_one
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

def local_search(value, final_value, x, fitness,dimension, multiplication, options, solution ):
    # TODO: randomize starting position of local search
    negative_imaginary = 0. - 1.j
    imaginary = 0.+1.j
    one = 1. + 0.j
    zero = 0. + 0.j
    neg_one = -1. + 0.j
    # b_negative_imaginary = 0b10000
    # b_imaginary = 0b01000
    # b_one = 0b00100
    # b_zero = 0b00010
    # b_neg_one = 0b00001
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
            if value[i][j] == one:
                val1[i][j] = zero
                val2[i][j] = neg_one
                val3[i][j] = negative_imaginary
                val4[i][j] = imaginary
                final_val1 = expand(val1, dimension, multiplication, options)
                final_val2 = expand(val2, dimension, multiplication, options)
                final_val3 = expand(val3, dimension, multiplication, options)
                final_val4 = expand(val4, dimension, multiplication, options)
                cost1, x1 = determine_fitness(final_val1, solution)
                cost2, x2 = determine_fitness(final_val2, solution)
                cost3, x3 = determine_fitness(final_val3, solution)
                cost4, x4 = determine_fitness(final_val4, solution)
            elif value[i][j] == zero:
                val1[i][j] = one
                val2[i][j] = neg_one
                val3[i][j] = negative_imaginary
                val4[i][j] = imaginary
                final_val1 = expand(val1, dimension, multiplication, options)
                final_val2 = expand(val2, dimension, multiplication, options)
                final_val3 = expand(val3, dimension, multiplication, options)
                final_val4 = expand(val4, dimension, multiplication, options)
                cost1, x1 = determine_fitness(final_val1, solution)
                cost2, x2 = determine_fitness(final_val2, solution)
                cost3, x3 = determine_fitness(final_val3, solution)
                cost4, x4 = determine_fitness(final_val4, solution)
            elif value[i][j] == imaginary:
                val1[i][j] = one
                val2[i][j] = neg_one
                val3[i][j] = negative_imaginary
                val4[i][j] = zero
                final_val1 = expand(val1, dimension, multiplication, options)
                final_val2 = expand(val2, dimension, multiplication, options)
                final_val3 = expand(val3, dimension, multiplication, options)
                final_val4 = expand(val4, dimension, multiplication, options)
                cost1, x1 = determine_fitness(final_val1, solution)
                cost2, x2 = determine_fitness(final_val2, solution)
                cost3, x3 = determine_fitness(final_val3, solution)
                cost4, x4 = determine_fitness(final_val4, solution)
            elif value[i][j] == negative_imaginary:
                val1[i][j] = one
                val2[i][j] = neg_one
                val3[i][j] = zero
                val4[i][j] = imaginary
                final_val1 = expand(val1, dimension, multiplication, options)
                final_val2 = expand(val2, dimension, multiplication, options)
                final_val3 = expand(val3, dimension, multiplication, options)
                final_val4 = expand(val4, dimension, multiplication, options)
                cost1, x1 = determine_fitness(final_val1, solution)
                cost2, x2 = determine_fitness(final_val2, solution)
                cost3, x3 = determine_fitness(final_val3, solution)
                cost4, x4 = determine_fitness(final_val4, solution)
            else:
                val1[i][j] = one
                val2[i][j] = neg_one
                val3[i][j] = zero
                val4[i][j] = imaginary
                final_val1 = expand(val1, dimension, multiplication, options)
                final_val2 = expand(val2, dimension, multiplication, options)
                final_val3 = expand(val3, dimension, multiplication, options)
                final_val4 = expand(val4, dimension, multiplication, options)
                cost1, x1 = determine_fitness(final_val1, solution)
                cost2, x2 = determine_fitness(final_val2, solution)
                cost3, x3 = determine_fitness(final_val3, solution)
                cost4, x4 = determine_fitness(final_val4, solution)
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


if __name__ == '__main__':

    possible_options = find_options(2, True)
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


    dictionary = create_dictionary(possible_options)
    # print(dictionary)
    # for key in dictionary:
    #     print(key)
    chrome1 = create_chromosome(2,7)
    chrome2 = create_chromosome(2,7)
    print("{0:b}".format(chrome1))
    print("{0:b}".format(chrome2))
    value1 = decode(chrome1, 2, 7)
    value2 = decode(chrome2, 2, 7)
    expanded = expand(value1, 2, 7, dictionary)
    solution = create_solution()
    fitness, temp_x = determine_fitness(expanded, solution)
    print(fitness)
    # while fitness < 1:

    #     value1 , expanded, temp_x, fitness = local_search(value1, expanded, temp_x, fitness, 2, 7, dictionary, solution)
    #     print("********************")
    #     print(fitness)


    test_value = [[ 0.+0.j , 1.+0.j , 0.+0.j , 0.+0.j , 0.+0.j , 0.+0.j ,-1.+0.j]
                ,[ 0.+0.j , 0.+0.j , 0.-1.j , 0.-1.j , 0.+1.j , 0.-1.j , 0.+0.j]
                ,[-1.+0.j,  0.+1.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.-1.j]
                ,[ 0.-1.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j]
                ,[ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j]
                ,[ 0.-1.j,  0.+0.j,  1.+0.j,  1.+0.j, -1.+0.j,  0.+0.j,  0.+0.j]
                ,[ 0.+0.j,  1.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j]
                ,[ 0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.-1.j,  0.+1.j,  0.+0.j]]

    print(test_value)
    expand2 = expand(test_value,2,7,dictionary)
    print(expand2)
    fitness2, temp_x2 = determine_fitness(expand2, solution)
    print(temp_x)
    print(fitness2)

    test_sol = np.matmul(expand2,temp_x)
    print(test_sol)
    # data = json.dumps(dictionary)
    # with open('2by2imaginaryData.json', 'w') as outfile:
    #     outfile.write(data)
    # 
    # print(type(data))