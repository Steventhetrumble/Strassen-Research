import json


if __name__ == '__main__':
    json1_file = open('2by2data.json')
    json1_str = json1_file.read()
    json1_data = json.loads(json1_str)

    print json1_data["1010"]["1010"]