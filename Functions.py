from imports import *
from data import *
import time
start_time = time.time()

def flatten(lst: List[Any]) -> Iterable[Any]:
    """Flatten a list using generators comprehensions.
        Returns a flattened version of list lst.
    """

    for sublist in lst:
         if isinstance(sublist, list):
             for item in sublist:
                 yield item
         else:
             yield sublist

def modelWordExtraction(string):
    """
    modelWordExtraction extracts modelwords which are a combinations of numerical an alphabetic tokens and
    modelIDs which are a words that contain a sequence of alphabetic-numeric-alphabetic or numeric-alphabetic-numeric.
    If multiple modelIDs are found, the longest one is returned.

    :param string: any string
    :return: list of modelwords and modelID
    """

    modelID = []
    res = string.split()
    MW = []

    for word in res:
        if re.findall("((?:[a-zA-Z]+[0-9]|[0-9]+[a-zA-Z])[a-zA-Z0-9]*)", word):
            MW.append(word)
        if re.findall('[a-zA-Z0-9][0-9]+[a-zA-Z]+[0-9]+[a-zA-Z0-9]|[a-zA-Z0-9]*[a-zA-Z]+[0-9]+[a-zA-Z]+[a-zA-Z0-9]+', word):
            modelID.append(word)

    if len(modelID) > 1:
        longest_modelID = modelID[1]
        for ID in modelID:
            if len(ID) > len(longest_modelID):
                longest_modelID = ID
        modelID = [longest_modelID]
    return MW, modelID

def modelWordSet(modelWordDict):
    """
    modelWordSet creates a set of unique model words

    :param dictionary: dictionary of model words per product
    :return: set of unique model words
    """
    modelWordSet = set()

    for key, value in modelWordDict.items():
        for MW in value:
            modelWordSet.add(MW)
    return modelWordSet

def binaryVectors(modelWordList, modelWordSet):
    """
    binaryVectors converts model word lists to a binary representation

    :param modelWordList: list of model words in product
    :param modelWordSet: set of all model words in the product set
    :return: dictionary of binary vector representations of the model words
    """

    # make empty list of binary vectors to be filled
    binary_vectors = {}
    vec_length = len(modelWordSet)

    for key, value in modelWordList.items():

        vec = np.zeros(vec_length)
        index = 0 # required as sets are not subscriptable
        for MW in modelWordSet:
            if MW in value:
                vec[index] = 1
            index = index + 1

        binary_vectors.update({key: vec})
    return binary_vectors

# Different minhashing methods
def minhash(binary_vec, n):
    """
    Converts the binary vectors into a signature matrix with minimal loss of information. The hash function is of the
    form (a + b * x) % k, where both a and b are randomly generated integers and k is the first prime number larger
    than the number of rows in the new vector (n).

    :param binary_vec: a binary vector representation of the model words
    :param n: the number of hash functions, this will be the number of rows in the signature matrix
    :return: signature matrix (n x number of products)
    """

    # random.seed(1)

    rows_binaryVec = len(binary_vec[0]) # length of a binary vector, equal to length of modelWordSet
    no_items = len(binary_vec) # number of items
    binary_vec = np.array(binary_vec).T
    signature_matrix = np.full((n, no_items), math.inf)
    hashFunctionInput = np.empty((n, 2))
    k = nextprime(n)

    for i in range(n):
        a = randint(1, k - 1)
        hashFunctionInput[i, 0] = a

        b = randint(1, k - 1)
        hashFunctionInput[i, 1] = b

    for row in range(1, rows_binaryVec + 1):
        index = row - 1
        e, r = np.ones(n), np.full(n, row)
        hashFunctions = np.sum(hashFunctionInput * np.stack((e, r), axis=1), axis = 1) % k # Matrix multiplication to obtain n of proposed hash function

        for i in range(n):
            # vector where all entries that where 0 are set to inf and the others to their computed hash value
            check_vec = np.where(binary_vec[index] == 0, math.inf, hashFunctions[i])

            # if the hash value is smaller than the current value, replace. Otherwise keep old value
            signature_matrix[i] = np.where(check_vec < signature_matrix[i], hashFunctions[i], signature_matrix[i])
    return signature_matrix.astype(int)
def HD_minhash(binary_vec, n, modelIDMatches):
    """
    Converts the binary vectors into a signature matrix with minimal loss of information. The hash function is of the
    form (a + b * x) % k, where both a and b are randomly generated integers and k is the first prime number larger
    than the number of rows in the new vector (n). In addition, binary vectors of products that were matched on
    model ID are merged to increase information.

    :param binary_vec: a binary vector representation of the model words
    :param n: the number of hash functions, this will be the number of rows in the signature matrix
    :param modelIDMatches: list of previously matched pairs on model ID
    :return: signature matrix (n x number of products)
    """

    # random.seed(1)

    # Merge the matched modelID binary vectors
    vec_length = len(binary_vec[0])

    for pair in modelIDMatches:
        bin_vec1 = binary_vec[pair[0]]
        bin_vec2 = binary_vec[pair[1]]
        merged_vec = np.zeros(vec_length)
        for i in range(vec_length):
            if bin_vec1[i] == 1 or bin_vec2[i] == 1:
                merged_vec[i] = 1
        binary_vec[pair[0]] = merged_vec
        binary_vec[pair[1]] = merged_vec


    rows_binaryVec = len(binary_vec[0]) # length of a binary vector, equal to length of modelWordSet
    no_items = len(binary_vec) # number of items
    binary_vec = np.array(binary_vec).T
    signature_matrix = np.full((n, no_items), math.inf)
    hashFunctionInput = np.empty((n, 2))
    k = nextprime(n)

    for i in range(n):
        a = randint(1, k - 1)
        hashFunctionInput[i, 0] = a

        b = randint(1, k - 1)
        hashFunctionInput[i, 1] = b

    for row in range(1, rows_binaryVec + 1):
        index = row - 1
        e, r = np.ones(n), np.full(n, row)
        hashFunctions = np.sum(hashFunctionInput * np.stack((e, r), axis=1), axis = 1) % k # Matrix multiplication to obtain n of proposed hash function

        for i in range(n):
            # vector where all entries that where 0 are set to inf and the others to their computer hash value
            check_vec = np.where(binary_vec[index] == 0, math.inf, hashFunctions[i])

            # if the hash value is smaller than the current value, replace. Otherwise continue
            signature_matrix[i] = np.where(check_vec < signature_matrix[i], hashFunctions[i], signature_matrix[i])

    return signature_matrix.astype(int)
def HDplus_minhash(binary_vec, n, modelIDMatches):
    """
    Converts the binary vectors into a signature matrix with minimal loss of information. The hash function is of the
    form (a + b * x) % k, where both a and b are randomly generated integers and k is the first prime number larger
    than the number of rows in the new vector (n). In addition, binary vectors of products that were matched on
    model ID are merged to increase information. Hereafter, the pairs are merged into one column to reduce the
    number of columns.

    :param binary_vec: a binary vector representation of the model words
    :param n: the number of hash functions, this will be the number of rows in the signature matrix
    :param modelIDMatches: list of previously matched pairs on model ID
    :return: signature matrix (n x number of products), the keys of the corresponding signature matrix, reduction of products
    """

    # Merge the matched modelID binary vectors
    vec_length = len(binary_vec[0])

    # dictionary with pairs
    red_binVec_dict = {}

    for pair in modelIDMatches:
        bin_vec1 = binary_vec[pair[0]]
        bin_vec2 = binary_vec[pair[1]]
        merged_vec = np.zeros(vec_length)
        for i in range(vec_length):
            if bin_vec1[i] == 1 or bin_vec2[i] == 1:
                merged_vec[i] = 1
        red_binVec_dict.update({pair: merged_vec})

    for vector in enumerate(binary_vec):
        in_modelID = False
        for pair in modelIDMatches:
            if vector[0] in pair:
                in_modelID = True

        if in_modelID == False:
            red_binVec_dict.update({vector[0] : vector[1]})

    # Number of products/column reduced by merging
    product_reduction = len(binary_vec) - len(red_binVec_dict)
    red_keys = red_binVec_dict.keys()
    red_binVec_list = list(red_binVec_dict.values())

    rows_binaryVec = len(red_binVec_list[0]) # length of a binary vector, equal to length of modelWordSet
    no_items = len(red_binVec_list)  # number of items
    binary_vec = np.array(red_binVec_list).T
    signature_matrix = np.full((n, no_items), math.inf)
    hashFunctionInput = np.empty((n, 2))
    k = nextprime(n)

    for i in range(n):
        a = randint(1, k - 1)
        hashFunctionInput[i, 0] = a

        b = randint(1, k - 1)
        hashFunctionInput[i, 1] = b

    for row in range(1, rows_binaryVec + 1):
        index = row - 1
        e, r = np.ones(n), np.full(n, row)
        hashFunctions = np.sum(hashFunctionInput * np.stack((e, r), axis=1), axis = 1) % k # Matrix multiplication to obtain n of proposed hash function

        for i in range(n):
            # vector where all entries that where 0 are set to inf and the others to their computer hash value
            check_vec = np.where(binary_vec[index] == 0, math.inf, hashFunctions[i])

            # if the hash value is smaller than the current value, replace. Otherwise continue
            signature_matrix[i] = np.where(check_vec < signature_matrix[i], hashFunctions[i], signature_matrix[i])

    return signature_matrix.astype(int), red_keys, product_reduction

# Different LSH methods, HDplus requires some extra steps
def lsh(signatureMatrix, b, r, show: bool, modelIDDict):
    n_hash, n_products = signatureMatrix.shape  # number of hashkeys
    assert (b * r == n_hash) # check whether b * r = n holds
    bands = np.split(signatureMatrix, b, axis=0)
    potentialPairs = set()

    for band in bands:
        tempDict = {} # Temporary dictionary to store the hash function results
        for product, rows in enumerate(band.transpose()):
            hash_key = int(float(''.join(map(str, rows)))) # Hash key is joined values of the column (see paper)

            if hash_key in tempDict:
                tempDict[hash_key] = np.append(tempDict[hash_key], product)
            else:
                tempDict[hash_key] = np.array([product])

        for potentialPair in tempDict.values():
            if len(potentialPair) > 1:
                for i, item1 in enumerate(potentialPair):
                    for j in range(i + 1, len(potentialPair)):
                        potentialPairs.add(tuple(sorted((potentialPair[i], potentialPair[j]))))

    if show:
        correct, incorrect = modelIDCheck(potentialPairs, modelIDDict)
        true_duplicates, non_duplicates = true_duplicate_extraction(modelIDDict)

        PQ = len(correct) / (len(potentialPairs) + 0.001) # avoid dividing by zero
        PC = len(correct) / (len(true_duplicates) + 0.001)
        F1star = 2  * (PQ * PC) / (PQ + PC + 0.001)

        print("PQ:        %6.6f " % PQ)
        print("PC:        %6.6f " % PC)
        print("F1*:       %6.6f " % F1star)
    return potentialPairs

def HDplus_lsh(signatureMatrix, b, r, show: bool, modelIDDict, red_key_dict):
    n_hash, n_products = signatureMatrix.shape  # number of hashkeys
    assert (b * r == n_hash) # check whether b * r = n holds
    bands = np.split(signatureMatrix, b, axis=0)
    potentialKeyMatches = set()
    potentialPairs = set()

    for band in bands:
        tempDict = {} # Temporary dictionary to store the hash function results
        for product, rows in enumerate(band.transpose()):
            hash_key = int(float(''.join(map(str, rows)))) # Hash key is joined values of the column (see paper)

            if hash_key in tempDict:
                tempDict[hash_key] = np.append(tempDict[hash_key], product)
            else:
                tempDict[hash_key] = np.array([product])

        for potentialKeyMatch in tempDict.values():
            if len(potentialKeyMatch) > 1:
                for i, item1 in enumerate(potentialKeyMatch):
                    for j in range(i + 1, len(potentialKeyMatch)):
                        potentialKeyMatches.add(tuple(sorted((potentialKeyMatch[i], potentialKeyMatch[j]))))

    red_key_list = list(red_key_dict)

    for keyMatch in potentialKeyMatches:
        key1 = keyMatch[0]
        key2 = keyMatch[1]
        potentialPair = (red_key_list[key1], red_key_list[key2])
        if type(potentialPair[0]) is not int or type(potentialPair[1]) is not int:
            combinations = list(potentialPair[0])
            if type(potentialPair[1]) is int:
                combinations.append(potentialPair[1])
            else:
                combinations.extend(list(potentialPair[1]))
            res = [(a, b) for idx, a in enumerate(combinations) for b in combinations[idx + 1:]]
            for pair in res:
                if pair[0] != pair[1]:
                    potentialPairs.add(tuple(sorted(list(pair))))

        elif potentialPair[0] != potentialPair[1]:
            potentialPairs.add(tuple(sorted(list(potentialPair))))

    for pair in red_key_list:
        if type(pair) is not int:
            potentialPairs.add(pair)

    correct, incorrect = modelIDCheck(potentialPairs, modelIDDict)
    true_duplicates, non_duplicates = true_duplicate_extraction(modelIDDict)

    PQ = len(correct) / (len(potentialPairs) + 0.001)  # avoid dividing by zero
    PC = len(correct) / (len(true_duplicates) + 0.001)
    F1star = 2 * (PQ * PC) / (PQ + PC + 0.001)

    if show:
        print("PQ:        %6.6f " % PQ)
        print("PC:        %6.6f " % PC)
        print("F1*:       %6.6f " % F1star)
    return potentialPairs, [PQ, PC, F1star]

def modelWordDicts(dict):
    modelWord_dict = {}
    modelID_dict = {}

    for key, value in dict.items():
        modelWord_dict.update({key: modelWordExtraction(value)[0]})
        modelID_dict.update({key: list(set(modelWordExtraction(value)[1]))})
    return modelWord_dict, modelID_dict

def modelIDCandidate(shopDict, featureMapDict, modelIDDict, k):
    modelIDs = {}
    candidates_clusters = []
    candidates_pairs = set()
    # and not re.search("Refurbished", titleDict[key]):
    for key, value in modelIDDict.items():
        if len(value) > 0 and len(featureMapDict[key]) > k:
            # print(len(modelWordDict[key]))
            if value[0] not in modelIDs:
                modelIDs.update({value[0]: key})
            else:
                modelIDs[value[0]] = [modelIDs[value[0]], key]

    for key, values in modelIDs.items():
        if type(values) is not int:
            candidates_clusters.append(list(flatten(flatten(values))))

    for key, values in modelIDs.items():
        if type(values) is not int:
            flattened_list = list(flatten(flatten(values)))
            if len(flattened_list) > 2:
                res = [(a, b) for idx, a in enumerate(flattened_list) for b in flattened_list[idx + 1:]]
                for i in res:
                    candidates_pairs.add(i)
            else:
                candidates_pairs.add(tuple(values))

    return candidates_clusters, candidates_pairs

def modelIDCheck(modelIDCandidateList, modelID_dicti):
    correct = []
    incorrect = []
    for candidate_pair in modelIDCandidateList:
        modelIDcheck = []
        for item in candidate_pair:
            modelIDcheck.append(modelID_dicti[item])

        set_check = set(modelIDcheck)
        if len(set_check) > 1:
            incorrect.append(candidate_pair)
        else:
            correct.append(candidate_pair)
    return correct, incorrect

def true_duplicates_count(modelIDdictionary):
    modelIDList = list(modelIDdictionary.values())
    my_dict = {i: modelIDList.count(i) for i in modelIDList}

    count_pairs = 0
    count_three = 0
    count_four = 0
    for key, values in my_dict.items():
        if values > 1 and values < 3:
            count_pairs += 1
        if values > 2 and values < 4:
            count_three += 1
        if values > 3:
            count_four +=1
    return count_pairs, count_three, count_four

def true_duplicate_extraction(modelIDdictionary):
    true_duplicate_pairs = set()
    non_duplicate_pairs = set()

    for key1, value1 in modelIDdictionary.items():
        for key2, value2 in modelIDdictionary.items():
            if value1 == value2 and key1 != key2:
                true_duplicate = [key1, key2]
                true_duplicate_pairs.add(tuple(sorted(true_duplicate)))
            else:
                not_duplicate = [key1, key2]
                non_duplicate_pairs.add(tuple(sorted(not_duplicate)))
    return true_duplicate_pairs, non_duplicate_pairs

def count_duplicates(correctOnes):
    count_pairs = 0
    count_three = 0
    count_four = 0
    for pair in correctOnes:
        if len(pair) > 1 and len(pair) < 3:
            count_pairs += 1
        if len(pair) > 2 and len(pair) < 4:
            count_three += 1
        if len(pair) > 3:
            count_four +=1
    return count_pairs, count_three, count_four

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

def differentBrand(pair, KV_dict, brand_list, cl_titleDict):
    differentBrand = False # No proof that we have a different brand

    item1 = pair[0]
    item2 = pair[1]

    keys1 = KV_dict[item1].keys()
    keys2 = KV_dict[item2].keys()

    # Not the same brand in featureMap, hence we return True for different brand
    if "Brand" in keys1 and "Brand" in keys2:
        if KV_dict[item1]["Brand"].lower() != KV_dict[item2]["Brand"].lower():
            differentBrand = True

    brand1 = brand_list.intersection(cl_titleDict[item1].split())
    brand2 = brand_list.intersection(cl_titleDict[item2].split())

    # Brand list search, if we find two brands and they are not the same, proof that there are different brands!
    if len(brand1) > 0 and len(brand2) > 0 and brand1 != brand2:
        differentBrand = True

    return differentBrand

def differentModelID(pair, title_dict):
    differentModelID = False # No proof that we have a different modelID

    item1 = pair[0]
    item2 = pair[1]

    modelID1 = modelWordExtraction(title_dict[item1])[1]
    modelID2 = modelWordExtraction(title_dict[item2])[1]

    # Brand list search, if we find two brands and they are not the same, proof that there are different brands!
    if len(modelID1) > 0 and len(modelID2) > 0 and modelID1 != modelID2:
        differentModelID = True

    return differentModelID

def dissimilarity_matrix_NO(decimal_dict, candidatePairs, shopDict, KV_dict, cl_modelWordDict, cl_titleDict, brand_list):
    nItems = len(shopDict)
    simMatrix = np.matrix(-np.ones((nItems, nItems)) * math.inf) # initialize matrix with "-inf"

    joint_modelwords = {}

    for key, value in cl_modelWordDict.items():
        for key1, value1 in decimal_dict.items():
            joint_modelwords.update({key: value + value1})

    for pair in candidatePairs:
        product1 = pair[0]
        product2 = pair[1]

        if shopDict[product1] != shopDict[product2]:
            if not differentBrand(pair, KV_dict, brand_list, cl_titleDict):
                simMatrix[product1, product2] = jaccard_similarity(set(joint_modelwords[product1]),
                                                                   set(joint_modelwords[product2]))

    disSimMatrix = 1 - simMatrix
    dissimMatrix = np.clip(disSimMatrix, 0, 20000) # Clustering algorithm doesn't accept math.inf
    return dissimMatrix

def dissimilarity_matrix(modelIDMatches, decimal_dict, candidatePairs, shopDict, KV_dict, cl_modelWordDict, titleDict, cl_titleDict, brand_list):
    nItems = len(shopDict)
    simMatrix = np.matrix(-np.ones((nItems, nItems)) * math.inf) # initialize matrix with "-inf"

    joint_modelwords = {}

    for key, value in cl_modelWordDict.items():
        for key1, value1 in decimal_dict.items():
            joint_modelwords.update({key: value + value1})

    filteredPairs = candidatePairs - modelIDMatches

    for pair in modelIDMatches:
        product1 = pair[0]
        product2 = pair[1]
        simMatrix[product1, product2] = math.inf

    # Only look at the similarity for the pairs not found in initial filtering
    for pair in filteredPairs:
        product1 = pair[0]
        product2 = pair[1]

        if shopDict[product1] != shopDict[product2]:
            if not differentBrand(pair, KV_dict, brand_list, cl_titleDict):
                if not differentModelID(pair, titleDict):
                    # simMatrix[product1, product2] = jaccard_similarity(set(joint_modelwords[product1]),
                    #                                                    set(joint_modelwords[product2]))

                    MWs_tit1 = modelWordExtraction(cl_titleDict[product1])[0]
                    MWs_tit2 = modelWordExtraction(cl_titleDict[product2])[0]

                    wrongAlphNum = False

                    for MW1 in MWs_tit1:
                        for MW2 in MWs_tit2:
                            alphabetic1 = "".join(re.split("[^a-zA-Z]*", MW1))
                            alphabetic2 = "".join(re.split("[^a-zA-Z]*", MW2))

                            numeric1 = int(''.join(filter(str.isdigit, MW1)))
                            numeric2 = int(''.join(filter(str.isdigit, MW2)))

                            if levenshtein_distance(alphabetic1, alphabetic2) < 0.5:
                                if levenshtein_distance(str(numeric1), str(numeric2)) > 0.5:
                                    simMatrix[product1, product2] = 0
                                    wrongAlphNum = True #If we find equal alphabetics but mismatch in numerics assign 0

                    if wrongAlphNum == False:
                        simMatrix[product1, product2] = jaccard_similarity(set(joint_modelwords[product1]), set(joint_modelwords[product2]))

    disSimMatrix = 1 - simMatrix
    dissimMatrix = np.clip(disSimMatrix, 0, 20000) # Clustering algorithm doesn't accept math.inf
    return dissimMatrix

def clustering(disMatrix, t):
    clusterMethod = AgglomerativeClustering(affinity='precomputed', linkage='single', distance_threshold=t,
                                         n_clusters=None).fit_predict(disMatrix)

    cluster_dict = {}
    for i in range(len(clusterMethod)):
        if clusterMethod[i] not in cluster_dict.keys():
            cluster_dict[clusterMethod[i]] = [i]
        else:
            cluster_dict[clusterMethod[i]].append(i)

    cluster_dict_merged = {}
    for key, value in cluster_dict.items():
        if len(value) >= 2:
            cluster_dict_merged.update({key: value})

    candidatePairs = set()
    for key in cluster_dict_merged.keys():
        for pair in list(combinations(cluster_dict_merged[key], 2)):
            candidatePairs.add(tuple(sorted(list(pair))))

    return candidatePairs

def evaluateResults(candidatesLSH, candidatesClustering, modelIDDict, show: bool):
    true_duplicate_set = true_duplicate_extraction(modelIDDict)[0]

    TP_LSH = len(candidatesLSH.intersection(true_duplicate_set))
    TP = len(candidatesClustering.intersection(true_duplicate_set))
    FP = len(candidatesClustering) - TP
    FN = len(true_duplicate_set) - TP

    PC = TP_LSH / (len(true_duplicate_set) + 0.0001)
    PQ = TP_LSH / (len(candidatesLSH) + 0.0001)
    F1Star_LSH = 2 * (PQ * PC) / (PQ + PC + 0.0001)

    precision = TP / (TP + FP + 0.0001)
    recall = TP / (TP + FN + 0.0001)
    F1 = 2 * (precision * recall) / (precision + recall + 0.0001)

    N = len(modelIDDict)
    totalComparisons = (N * (N - 1)) / 2
    fraction = len(candidatesLSH) / totalComparisons

    if show:
        print("PQ:        %6.6f " % PQ)
        print("PC:        %6.6f " % PC)
        print("F1*:       %6.6f " % F1Star_LSH)
        print("precision: %6.6f " % precision)
        print("recall:    %6.6f " % recall)
        print("F1:        %6.6f " % F1)
        print("fraction:  %6.6f " % fraction)
    return [fraction, PQ, PC, F1Star_LSH, precision, recall, F1]

def evaluateResults_NO(candidatesLSH, candidatesClustering, modelIDDict, show: bool):
    true_duplicate_set = true_duplicate_extraction(modelIDDict)[0]

    TP_LSH = len(candidatesLSH.intersection(true_duplicate_set))
    TP = len(candidatesClustering.intersection(true_duplicate_set))
    FP = len(candidatesClustering) - TP
    FN = len(true_duplicate_set) - TP

    PC = TP_LSH / len(true_duplicate_set)
    PQ = TP_LSH / len(candidatesLSH)
    F1Star_LSH = 2 * (PQ * PC) / (PQ + PC + 0.0001)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall + 0.0001)

    N = len(modelIDDict)
    totalComparisons = (N * (N - 1)) / 2
    fraction = len(candidatesLSH) / totalComparisons

    if show:
        print("PQ:        %6.6f " % PQ)
        print("PC:        %6.6f " % PC)
        print("F1*:       %6.6f " % F1Star_LSH)
        print("precision: %6.6f " % precision)
        print("recall:    %6.6f " % recall)
        print("F1:        %6.6f " % F1)
        print("fraction:  %6.6f " % fraction)
    return [fraction, PQ, PC, F1Star_LSH, precision, recall, F1]

def is_prime(n):
  for i in range(2,n):
    if (n%i) == 0:
      return False
  return True

def bands_row_finder(n, t):
    optimal_b, optimal_r, ini_guess = 1,1,1

    # required to find logical bands
    if is_prime(n):
        n = n + 1

    valid_t = False

    while valid_t == False:
        for r in range(1, n + 1):
            for b in range(1, n + 1):
                if b * r == n:
                    approx = (1 / b) ** (1 /  r)
                    if abs(approx - t) < abs(ini_guess - t):
                        ini_guess = approx
                        optimal_b = b
                        optimal_r = r

                    if abs(approx - t) > 0.025:
                        n = n + 1

                    else:
                        valid_t = True
                        final_n = n

    return optimal_b, optimal_r, final_n, ini_guess

def bands_row_finder_old(n, t):
    optimal_b, optimal_r, ini_guess = 1,1,1

    # required to find logical bands
    if is_prime(n):
        n = n + 1


    for r in range(1, n + 1):
        for b in range(1, n + 1):
            if b * r == n:
                approx = (1 / b) ** (1 /  r)
                if abs(approx - t) < abs(ini_guess - t):
                    ini_guess = approx
                    optimal_b = b
                    optimal_r = r


    return optimal_b, optimal_r, ini_guess

def kgram_sim(string1, string2, k):
    dummy_string1 = " " * 2 + string1 + " " * 2
    dummy_string2 = " " * 2 + string2 + " " * 2

    shingles1 = ks.shingleset_list(dummy_string1, [3])
    shingles2 = ks.shingleset_list(dummy_string2, [3])

    n1 = len(shingles1)
    n2 = len(shingles2)

    diff_shingles = shingles1.union(shingles2) - shingles1.intersection(shingles2)

    similarity = (n1 + n2 - len(diff_shingles)) / (n1 + n2)
    return similarity

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    WORD = re.compile(r"\w+")
    words = WORD.findall(text)
    return Counter(words)

def get_avgLvSim(listWord1, listWord2):
    n = min(len(listWord1), len(listWord2))

    setWord1 = listWord1[0:n]
    setWord2 = listWord2[0:n]

    similarity = 0
    total_length = 0

    for x in setWord1:
        for y in setWord2:
            total_length += len(x) + len(y)


    for x in setWord1:
        for y in setWord2:
            lv = levenshtein_distance(x, y) / max(len(x), len(y))
            weight_sim = (len(x) + len(y)) / total_length
            similarity += (1 - lv) * weight_sim

    return similarity

def MSM(pair, gamma, mu, alpha, KV_featuremap, cl_titleDict):
    product1 = pair[0]
    product2 = pair[1]

    sim = 0
    avgSim = 0
    m = 0
    w = 0

    # Step 1: Key-matching
    # Find matching keys and return weighted k-gram similarity of these values
    keys1 = KV_featuremap[product1].keys()
    keys2 = KV_featuremap[product2].keys()

    keyMatch = set()

    for key1 in keys1:
        for key2 in keys2:
            keySim = kgram_sim(key1, key2, 3)
            if keySim > gamma:
                valueSim = kgram_sim(KV_featuremap[product1][key1], KV_featuremap[product2][key2], 3)
                weight = keySim
                sim = sim + weight * valueSim
                m += 1
                w = w + weight
                keyMatch.add(key1)
                keyMatch.add(key2)
    if w > 0:
        avgSim = sim / w

    # Step 2: Non-key matches
    MW_values1 = set()
    MW_values2 = set()

    for key1 in keys1:
        if key1 not in keyMatch:
            MWS1 = modelWordExtraction(string_cleaning(KV_featuremap[product1][key1]))[0]
            for MW in MWS1:
                if MW:
                    MW_values1.add(MW)


    for key2 in keys2:
        if key2 not in keyMatch:
            MWS2 = modelWordExtraction(string_cleaning(KV_featuremap[product2][key2]))[0]
            for MW in MWS2:
                if MW:
                    MW_values2.add(MW)

    mwPerc = len(MW_values1.intersection(MW_values2)) / len(MW_values1.union(MW_values2))


    # Step 3: TMWM
    vec1 = text_to_vector(cl_titleDict[product1])
    vec2 = text_to_vector(cl_titleDict[product2])

    nameCosineSim = get_cosine(vec1, vec2)

    if nameCosineSim > alpha:
        MWs_tit1 = modelWordExtraction(cl_titleDict[product1])[0]
        MWs_tit2 = modelWordExtraction(cl_titleDict[product2])[0]

        for MW1 in MWs_tit1:
            for MW2 in MWs_tit2:
                alphabetic1 = "".join(re.split("[^a-zA-Z]*", MW1))
                alphabetic2 = "".join(re.split("[^a-zA-Z]*", MW2))

                numeric1 = int(''.join(filter(str.isdigit, MW1)))
                numeric2 = int(''.join(filter(str.isdigit, MW2)))

                if levenshtein_distance(alphabetic1, alphabetic2) < 0.5:
                    if levenshtein_distance(str(numeric1), str(numeric2)) > 0.5:
                        similarity = 0
                        return similarity

        similarity = 1
        return similarity

    else:
        MWs_tit1 = modelWordExtraction(cl_titleDict[product1])[0]
        MWs_tit2 = modelWordExtraction(cl_titleDict[product2])[0]

        one_match = False

        MW_matches1 = []
        MW_matches2 = []

        for MW1 in MWs_tit1:
            for MW2 in MWs_tit2:
                alphabetic1 = "".join(re.split("[^a-zA-Z]*", MW1))
                alphabetic2 = "".join(re.split("[^a-zA-Z]*", MW2))

                numeric1 = int(''.join(filter(str.isdigit, MW1)))
                numeric2 = int(''.join(filter(str.isdigit, MW2)))

                if levenshtein_distance(alphabetic1, alphabetic2) < 0.5:
                    if levenshtein_distance(str(numeric1), str(numeric2)) > 0.5:
                        similarity = 0
                        return similarity
                    else:
                        MW_matches1.append(MW1)
                        MW_matches2.append(MW2)
                        one_match = True

    avgLvSim = get_avgLvSim(list(vec1.keys()), list(vec2.keys()))

    if one_match == False:
        finalNameSim = avgLvSim

    if one_match == True:
        delta = 0.5
        modelWordSimVal = get_avgLvSim(MW_matches1, MW_matches2)
        finalNameSim = delta * modelWordSimVal + (1 - delta) * avgLvSim

    theta1 = m / min(len(KV_featuremap[product1]), len(KV_featuremap[product2]))
    theta2 = 1 - theta1 - mu

    hSim = theta1 * avgSim + theta2 * mwPerc + mu * finalNameSim
    return hSim




