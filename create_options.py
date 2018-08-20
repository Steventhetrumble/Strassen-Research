import tables
import numpy as np
import os.path
import json


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
        for i in range(int((3 ** options_size))):
            rows = []
            for j in range(0, options_size):
                rows.append(1 - int(i % 3 ** (options_size - j) / (3 ** (options_size - (j + 1)))))
            number = np.count_nonzero(rows)
            if 0 < number < 5:
                options.append(rows)
        return options
    else:
        for i in range(int((3 ** options_size) / 2)):
            rows = []
            for j in range(0, options_size):
                rows.append(1 - int(i % 3 ** (options_size - j) / (3 ** (options_size - (j + 1)))))
            number = np.count_nonzero(rows)
            if 0 < number < 5:
                options.append(rows)
        return options


if __name__ == '__main__':

    possible_options = find_options(2, True)
    dictionary = create_dictionary(possible_options)
    data = json.dumps(dictionary)
    with open('2by2data.json', 'w') as outfile:
        outfile.write(data)
    print (dictionary["10-10"]["10-10"])
    print (type(data))