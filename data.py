from imports import *

with open('TVs-all-merged.json', 'r') as f:
  data = json.load(f)


def load_data(path):
    """
    Load and translate the json data into list of dictionaries
    """
    with open(path, 'r') as f:
        data = json.load(f)
    data_json = []

    for key, value in data.items():
        for dict in value:
            data_json.append(dict)
    return data_json

def dictionary_generator(unpacked_data):
    """
    Returns dict with key=original index, value = title, featuremap and title+featuremap
    """
    title_dict = {}
    featureMap_dict = {}
    featureMapKV_dict = {}
    titleFeatureMap_dict = {}
    modelID_dict = {}
    shop_dict = {}

    index = 0

    for item in unpacked_data:
        title = item["title"]
        featureMap = item["featuresMap"]
        modelID = item["modelID"]
        shop = item["shop"]

        featureMapRed = ' '.join(map(str, list(featureMap.values())))

        featureMapKV_dict.update({index: featureMap})
        title_dict.update({index: title})
        featureMap_dict.update({index: featureMapRed})
        titleFeatureMap_dict.update({index: title + " " + featureMapRed})
        modelID_dict.update({index: modelID})
        shop_dict.update({index: shop})

        index = index + 1
    return title_dict, featureMap_dict, titleFeatureMap_dict, modelID_dict, shop_dict, featureMapKV_dict

unpacked_data = load_data('TVs-all-merged.json')

def bootstrap_sample(data, percentage):
    """
    Returns a 63% trainingset of the data and the testset
    """

    # Set the splitting index at 63% of data
    no_items = len(data)
    splitting_index =int(no_items * percentage)

    # shuffle the data in random position
    shuffle(data)

    # Extract training and test set
    training_set = data[:splitting_index]
    test_set = data[splitting_index:]
    return training_set, test_set

def bootstrap_sample_resample(data, N):
    """
    Returns a bootstrap of N draws with replacement
    """
    draw_list = []

    for i in range(N):
        draw = randint(0, N - 1)
        if draw not in draw_list:
            draw_list.append(draw)

    bootstrap = []

    for draw in draw_list:
        bootstrap.append(data[draw])

    return bootstrap



# print(len(bootstrap2))


def string_cleaning(string):
    cleaned_string = string.lower()

    #clean inches
    cleaned_string = cleaned_string.replace('"', "inch")
    cleaned_string = cleaned_string.replace(' "', "inch")
    cleaned_string = cleaned_string.replace('inches', 'inch')
    cleaned_string = cleaned_string.replace(' inch', 'inch')
    cleaned_string = cleaned_string.replace('-inch', 'inch')
    cleaned_string = cleaned_string.replace('\'', 'inch')

    # clean Hertz
    cleaned_string = cleaned_string.replace('hertz', 'hz')
    cleaned_string = cleaned_string.replace('-hz', 'hz')
    cleaned_string = cleaned_string.replace(' hz', 'hz')

    # clean lbs NEW
    cleaned_string = cleaned_string.replace(' lbs', 'lbs')
    cleaned_string = cleaned_string.replace('Ibs', 'lbs')
    cleaned_string = cleaned_string.replace('lbs.', 'lbs')
    cleaned_string = cleaned_string.replace(' Ibs', 'lbs')
    cleaned_string = cleaned_string.replace(' pounds', 'lbs')

    # remove some punctuation
    cleaned_string = cleaned_string.replace("(", "")
    cleaned_string = cleaned_string.replace(")", "")
    cleaned_string = cleaned_string.replace("[", "")
    cleaned_string = cleaned_string.replace("]", "")

    # remove store names (never same store)
    cleaned_string = cleaned_string.replace('newegg.com', '')
    cleaned_string = cleaned_string.replace('best buy', '')
    cleaned_string = cleaned_string.replace('thenerds.net', '')

    # combine dimensions NEW
    # cleaned_string = cleaned_string.replace(' x ', 'x')

    return cleaned_string

def data_cleaning(dictionary):
    """
    Cleans a string of redundant characteristics, i.e.:
    - Normalizes values of inch, hz, lbs (sometimes Ibs or pounds NEW) REMEMBER TO CHANGE INCH FIRST!!
    - Remove interpunction??
    - Remove all forms of shop names: newegg, amazon, best buy, nerds ??
    - NEW: Remove 'Refurbished:' (every renames a refurbished article differently
    - lowercase all data !
    """
    cleaned_dictionary = {}

    for key, value in dictionary.items():
        # lower case
        cleaned_value = value.lower()
        # clean inches
        cleaned_value = cleaned_value.replace('"', "inch")
        cleaned_value = cleaned_value.replace(' "', "inch")
        cleaned_value = cleaned_value.replace('inches', 'inch')
        cleaned_value = cleaned_value.replace(' inch', 'inch')
        cleaned_value = cleaned_value.replace('-inch', 'inch')
        cleaned_value = cleaned_value.replace('\'', 'inch')
        cleaned_value = cleaned_value.replace(' \'', 'inch')

        # clean Hertz
        cleaned_value = cleaned_value.replace('hertz', 'hz')
        cleaned_value = cleaned_value.replace('-hz', 'hz')
        cleaned_value = cleaned_value.replace(' hz', 'hz')

        # clean lbs NEW
        cleaned_value = cleaned_value.replace(' lbs', 'lbs')
        cleaned_value = cleaned_value.replace('Ibs', 'lbs')
        cleaned_value = cleaned_value.replace('lbs.', 'lbs')
        cleaned_value = cleaned_value.replace(' Ibs', 'lbs')
        cleaned_value = cleaned_value.replace(' pounds', 'lbs')

        # remove some punctuation
        cleaned_value = cleaned_value.replace("(", "")
        cleaned_value = cleaned_value.replace(")", "")
        cleaned_value = cleaned_value.replace("[", "")
        cleaned_value = cleaned_value.replace("]", "")
        cleaned_value = cleaned_value.replace("-", "")

        # remove store names (never same store)
        cleaned_value = cleaned_value.replace('newegg.com', '')
        cleaned_value = cleaned_value.replace('best buy', '')
        cleaned_value = cleaned_value.replace('thenerds.net', '')

        # combine dimensions NEW
        # cleaned_value = cleaned_value.replace(' x ', 'x')

        # Removes punctuation !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~ NOTE: REVISE IF NEEDED
        # cleaned_value = value.translate(str.maketrans("", "", string.punctuation))


        cleaned_dictionary.update({key: cleaned_value})

    return cleaned_dictionary


def decimal_dict(feature_dictionary):
    """
    Cleans a string of redundant characteristics, i.e.:
    - Normalizes values of inch, hz, lbs (sometimes Ibs or pounds NEW) REMEMBER TO CHANGE INCH FIRST!!
    - Remove interpunction??
    - Remove all forms of shop names: newegg, amazon, best buy, nerds ??
    - NEW: Remove 'Refurbished:' (every renames a refurbished article differently
    - lowercase all data !
    """
    cleaned_dictionary = {}

    for key, value in feature_dictionary.items():
        # lower case
        cleaned_value = value.lower()
        # clean inches
        cleaned_value = cleaned_value.replace('"', "")
        cleaned_value = cleaned_value.replace(' "', "")
        cleaned_value = cleaned_value.replace('inches', '')
        cleaned_value = cleaned_value.replace(' inch', '')
        cleaned_value = cleaned_value.replace('-inch', '')

        # clean Hertz
        cleaned_value = cleaned_value.replace('hertz', '')
        cleaned_value = cleaned_value.replace('-hz', '')
        cleaned_value = cleaned_value.replace(' hz', '')

        # clean lbs NEW
        cleaned_value = cleaned_value.replace(' lbs', '')
        cleaned_value = cleaned_value.replace('Ibs', '')
        cleaned_value = cleaned_value.replace('lbs.', '')
        cleaned_value = cleaned_value.replace(' Ibs', '')
        cleaned_value = cleaned_value.replace(' pounds', '')

        # remove some punctuation
        cleaned_value = cleaned_value.replace("(", "")
        cleaned_value = cleaned_value.replace(")", "")
        cleaned_value = cleaned_value.replace("[", "")
        cleaned_value = cleaned_value.replace("]", "")

        mw_decimal = re.findall("([0-9]+\.[0-9]+)[a-zA-Z]*", value)

        cleaned_dictionary.update({key: mw_decimal})
    return cleaned_dictionary

def data_split(path):
    with open(path, 'r') as f:
        data = json.load(f)

    keys = data.keys()

    shops = []
    titles = []
    features = []

    for key in keys:
        shops.append(data[key][0]['shop'])
        titles.append(data[key][0]['title'])
        features.append(data[key][0]['featuresMap'])

    return shops, titles, features


# for key in data.keys(): # Gives all keys
#     print(key)

# for value in data.values(): # Gives all values
#     print(value)

# for item in data.items(): # Gives all key-value pairs
#     print(item)
#
# test = data["29PFL4508/F7"] #bestbuy
# test = data["SC-3211"] # newegg
test = data["UN46ES6580"] # amazon

# print(test)

unpacked_test = test[0]
# print(unpacked_test)

# print(unpacked_test['title'])

# for value in unpacked_test.values():
#     print(value)

