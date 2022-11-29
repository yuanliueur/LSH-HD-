from imports import *
from Functions import *
from data import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

no_bootstrap = 50

# Import brand list
my_file = open("brand_list.txt", "r")
data = my_file.read()
data_into_list = data.split('\n')
brand_list = set([x.lower() for x in data_into_list])

thresholds = np.arange(0, 1, 0.05)

with open("results.csv", 'w') as out:
    out.write("t,"
              "fraction, PQ, PC, F1star, precision, recall, F1,"
              "fraction_HD, PQ_HD, PC_HD, F1_star_HD, precision_HD, recall_HD, F1_HD,"
                "PQ_HDplus, PC_HDplus, F1_star_HDplus, n_reduced\n")


for t in thresholds:
    print('threshold: ', t)
    # results_base = np.zeros(7)
    results_HD = np.zeros(7)
    # results_HDplus = np.zeros(4)

    for i in range(no_bootstrap):
        print('boostrap number:', i + 1)

        # Take boostrap
        boot = bootstrap_sample(unpacked_data, 0.63)[0]

        # Generate required dictionaries / extract modelID matches
        title_dict, featureMaps_dict, titleFeatureMap_dict, modelID_dict, shop_dict, featureMapKV_dict = dictionary_generator(boot)
        modelIDMatchSet = modelIDCandidate(shop_dict, featureMaps_dict, modelWordDicts(title_dict)[1], 10)[1]
        correct, incorrect = modelIDCheck(modelIDMatchSet, modelID_dict)

        cl_title_dict = data_cleaning(title_dict)
        cl_features_dict = data_cleaning(featureMaps_dict)
        cl_combined_dict = data_cleaning(titleFeatureMap_dict)
        cl_modelWord_dict = modelWordDicts(cl_title_dict)[0]
        cl_features4dec_dict = decimal_dict(featureMaps_dict)
        cl_modelWordKV_dict = modelWordDicts(cl_features_dict)[0]
        cl_modelWordsKVred_dict = {}

        MW_set_title = modelWordSet(cl_modelWord_dict)
        MW_set_decs = modelWordSet(cl_features4dec_dict)

        for key, value in cl_modelWordKV_dict.items():
            MW_matches = []
            for item in value:
                if item in MW_set_title:
                    MW_matches.append(item)

            cl_modelWordsKVred_dict.update({key: MW_matches})

        cl_modelWordComb_set = MW_set_title.union(MW_set_decs)

        n_ini = round(len(cl_modelWordComb_set) / 2)
        if is_prime(n_ini):
            n_ini = n_ini + 1

        b, r, n, ini_guess = bands_row_finder(n_ini, t)

        ds = [cl_modelWord_dict, cl_features4dec_dict, cl_modelWordsKVred_dict]
        cl_modelWordComb_dict = {}
        for k in cl_modelWord_dict.keys():
            cl_modelWordComb_dict[k] = np.concatenate(list(cl_modelWordComb_dict[k] for cl_modelWordComb_dict in ds))

        model_words_title = modelWordDicts(cl_title_dict)[0]
        model_word_set_title = modelWordSet(model_words_title)

        binaryVector = binaryVectors(cl_modelWordComb_dict, cl_modelWordComb_set)
        binaryVectorList = list(binaryVector.values())

        # signature_matrix = minhash(binaryVectorList, n)
        signature_matrix_HD = HD_minhash(binaryVectorList, n, modelIDMatchSet)
        # signature_matrix_HDplus = HDplus_minhash(binaryVectorList, n, modelIDMatchSet)
#
#         potentialPairs = lsh(signature_matrix, b, r, False, modelID_dict)
        potentialPairs_HD = lsh(signature_matrix_HD, b, r, False, modelID_dict)
#         potentialPairs_HDplus = HDplus_lsh(signature_matrix_HDplus[0], b, r, False, modelID_dict, signature_matrix_HDplus[1])
#
#         disMatrix = dissimilarity_matrix(modelIDMatchSet, cl_features4dec_dict, potentialPairs, shop_dict,
#                                             featureMapKV_dict, cl_modelWord_dict, title_dict, cl_title_dict, brand_list)
        disMatrix_HD = dissimilarity_matrix(modelIDMatchSet, cl_features4dec_dict, potentialPairs_HD, shop_dict,
                                            featureMapKV_dict, cl_modelWord_dict, title_dict, cl_title_dict, brand_list)

#
        # clusters = clustering(disMatrix, 0.05)
        clusters_HD = clustering(disMatrix_HD, 0.05)

#
        # final_pairs = set()
        final_pairs_HD = set()

#
#         new_clusters = clusters - modelIDMatchSet
        new_clusters_HD = clusters_HD - modelIDMatchSet

#         for pair in new_clusters:
#             if shop_dict[pair[0]] != shop_dict[pair[1]]:
#                 if not differentBrand(pair, featureMapKV_dict, brand_list, cl_title_dict):
#                     if not differentModelID(pair, title_dict):
#                         final_pairs.add(pair)

        for pair in new_clusters_HD:
            if shop_dict[pair[0]] != shop_dict[pair[1]]:
                if not differentBrand(pair, featureMapKV_dict, brand_list, cl_title_dict):
                    if not differentModelID(pair, title_dict):
                        final_pairs_HD.add(pair)

        # combined_final_pairs = final_pairs.union(modelIDMatchSet)
        combined_final_pairs_HD = final_pairs_HD.union(modelIDMatchSet)

        results_HD += evaluateResults(potentialPairs_HD, combined_final_pairs_HD, modelID_dict, False)

#
#         results_base += evaluateResults(potentialPairs, combined_final_pairs, modelID_dict, False)
#         combined_results_HDplus = potentialPairs_HDplus[1] + [signature_matrix_HDplus[2]]
#         results_HDplus += combined_results_HDplus

#     statistics = results_base / no_bootstrap
    statistics_HD = results_HD / no_bootstrap
#     statistics_HDplus = results_HDplus / no_bootstrap

    with open("results.csv", 'a') as out:
        out.write(str(t))
#
#         for stat in statistics:
#             out.write("," + str(stat))
#
        for stat in statistics_HD:
            out.write("," + str(stat))
#
#         for stat in statistics_HDplus:
#             out.write("," + str(stat))

        out.write("\n")

#
#
#
# with open("resultsRB.csv", 'w') as out:
#     out.write("t, t_estimNEW, t_estimOLD\n")
#
# for t in thresholds:
#     results_base = np.zeros(2)
#
#     for i in range(no_bootstrap):
#         # Take boostrap
#         boot = bootstrap_sample_resample(unpacked_data, len(unpacked_data))
#
#         # Generate required dictionaries / extract modelID matches
#         title_dict, featureMaps_dict, titleFeatureMap_dict, modelID_dict, shop_dict, featureMapKV_dict = dictionary_generator(boot)
#         modelIDMatchSet = modelIDCandidate(shop_dict, featureMaps_dict, modelWordDicts(title_dict)[1], 10)[1]
#
#         cl_title_dict = data_cleaning(title_dict)
#         cl_features_dict = data_cleaning(featureMaps_dict)
#         cl_combined_dict = data_cleaning(titleFeatureMap_dict)
#         cl_modelWord_dict = modelWordDicts(cl_title_dict)[0]
#         cl_features4dec_dict = decimal_dict(featureMaps_dict)
#         cl_modelWordKV_dict = modelWordDicts(cl_features_dict)[0]
#         cl_modelWordsKVred_dict = {}
#
#         MW_set_title = modelWordSet(cl_modelWord_dict)
#         MW_set_decs = modelWordSet(cl_features4dec_dict)
#
#         for key, value in cl_modelWordKV_dict.items():
#             MW_matches = []
#             for item in value:
#                 if item in MW_set_title:
#                     MW_matches.append(item)
#
#             cl_modelWordsKVred_dict.update({key: MW_matches})
#
#         cl_modelWordComb_set = MW_set_title.union(MW_set_decs)
#
#         n_ini = round(len(cl_modelWordComb_set) / 2)
#         if is_prime(n_ini):
#             n_ini = n_ini + 1
#
#         b, r, n_new, ini_guess = bands_row_finder(n_ini, t)
#         b_old, r_old, ini_guess_old = bands_row_finder_old(n_ini, t)
#
#         with open("resultsRB.csv", 'a') as out:
#             out.write(str(t))
#             out.write("," + str(ini_guess))
#             out.write("," + str(ini_guess_old))
#             out.write("\n")
#
#
#
#
# # for row in range(len(b_list)):
# #     print('row number: ', row, ' out of ', len(b_list))
# #     b = b_list[row]
# #     r = r_list[row]
# #
# #     bands_rows = [b, r]
# #
# #     results_base = np.zeros(7)
# #     results_HD = np.zeros(7)
# #     results_HDplus = np.zeros(4)
# #
# #     for i in range(no_bootstrap):
# #         print('boostrap number:', i)
# #
# #         # Take boostrap
# #         boot = bootstrap_sample(unpacked_data, 0.63)[0]
# #
# #         # Generate required dictionaries / extract modelID matches
# #         title_dict, featureMaps_dict, titleFeatureMap_dict, modelID_dict, shop_dict, featureMapKV_dict = dictionary_generator(boot)
# #         modelIDMatchSet = modelIDCandidate(shop_dict, featureMaps_dict, modelWordDicts(title_dict)[1], 10)[1]
# #         correct, incorrect = modelIDCheck(modelIDMatchSet, modelID_dict)
# #
# #         cl_title_dict = data_cleaning(title_dict)
# #         cl_features_dict = data_cleaning(featureMaps_dict)
# #         cl_combined_dict = data_cleaning(titleFeatureMap_dict)
# #         cl_modelWord_dict = modelWordDicts(cl_title_dict)[0]
# #         cl_features4dec_dict = decimal_dict(featureMaps_dict)
# #         cl_modelWordKV_dict = modelWordDicts(cl_features_dict)[0]
# #         cl_modelWordsKVred_dict = {}
# #
# #         MW_set_title = modelWordSet(cl_modelWord_dict)
# #         MW_set_decs = modelWordSet(cl_features4dec_dict)
# #         # MW_set_KV = modelWordSet(cl_modelWordKV_dict)
# #
# #         for key, value in cl_modelWordKV_dict.items():
# #             MW_matches = []
# #             for item in value:
# #                 if item in MW_set_title:
# #                     MW_matches.append(item)
# #
# #             cl_modelWordsKVred_dict.update({key: MW_matches})
# #
# #         cl_modelWordComb_set = MW_set_title.union(MW_set_decs)
# #
# #         ds = [cl_modelWord_dict, cl_features4dec_dict, cl_modelWordsKVred_dict]
# #         cl_modelWordComb_dict = {}
# #         for k in cl_modelWord_dict.keys():
# #             cl_modelWordComb_dict[k] = np.concatenate(list(cl_modelWordComb_dict[k] for cl_modelWordComb_dict in ds))
# #
# #         model_words_title = modelWordDicts(cl_title_dict)[0]
# #         model_word_set_title = modelWordSet(model_words_title)
# #
# #         # binaryVector = binaryVectors(model_words_title, model_word_set_title)
# #         binaryVector = binaryVectors(cl_modelWordComb_dict, cl_modelWordComb_set)
# #         # binaryVectorList = list(binaryVector.values())
# #         binaryVectorList = list(binaryVector.values())
# #
# #         signature_matrix = minhash(binaryVectorList, n)
# #         signature_matrix_HD = HD_minhash(binaryVectorList, n, modelIDMatchSet)
# #         signature_matrix_HDplus = HDplus_minhash(binaryVectorList, n, modelIDMatchSet)
# #
# #         potentialPairs = lsh(signature_matrix, b, r, False, modelID_dict)
# #         potentialPairs_HD = lsh(signature_matrix_HD, b, r, False, modelID_dict)
# #         potentialPairs_HDplus = HDplus_lsh(signature_matrix_HDplus[0], b, r, False, modelID_dict, signature_matrix_HDplus[1])
# #
# #         # disMatrix = dissimilarity_matrix_NO(cl_features4dec_dict, potentialPairs, shop_dict,
# #         #                                  featureMapKV_dict, cl_modelWord_dict, cl_title_dict, brand_list)
# #         disMatrix = dissimilarity_matrix(modelIDMatchSet, cl_features4dec_dict, potentialPairs, shop_dict,
# #                                             featureMapKV_dict, cl_modelWord_dict, title_dict, cl_title_dict, brand_list)
# #         disMatrix_HD = dissimilarity_matrix(modelIDMatchSet, cl_features4dec_dict, potentialPairs_HD, shop_dict,
# #                                             featureMapKV_dict, cl_modelWord_dict, title_dict, cl_title_dict, brand_list)
# #
# #         clusters = clustering(disMatrix, 0.05)
# #         clusters_HD = clustering(disMatrix_HD, 0.05)
# #
# #         final_pairs = set()
# #         final_pairs_HD = set()
# #
# #         new_clusters = clusters - modelIDMatchSet
# #         new_clusters_HD = clusters_HD - modelIDMatchSet
# #
# #         for pair in new_clusters:
# #             if shop_dict[pair[0]] != shop_dict[pair[1]]:
# #                 if not differentBrand(pair, featureMapKV_dict, brand_list, cl_title_dict):
# #                     if not differentModelID(pair, title_dict):
# #                         final_pairs.add(pair)
# #
# #         for pair in new_clusters_HD:
# #             if shop_dict[pair[0]] != shop_dict[pair[1]]:
# #                 if not differentBrand(pair, featureMapKV_dict, brand_list, cl_title_dict):
# #                     if not differentModelID(pair, title_dict):
# #                         final_pairs_HD.add(pair)
# #
# #         # combined_final_pairs_NO = final_pairs
# #         combined_final_pairs = final_pairs.union(modelIDMatchSet)
# #         combined_final_pairs_HD = final_pairs_HD.union(modelIDMatchSet)
# #
# #         # results_base += evaluateResults(potentialPairs, combined_final_pairs_NO, modelID_dict, False)
# #         results_base += evaluateResults(potentialPairs, combined_final_pairs, modelID_dict, False)
# #         results_HD += evaluateResults(potentialPairs_HD, combined_final_pairs_HD, modelID_dict, False)
# #         combined_results_HDplus = potentialPairs_HDplus[1] + [signature_matrix_HDplus[2]]
# #         results_HDplus += combined_results_HDplus
# #
# #         # results_base = evaluateResults(potentialPairs, combined_final_pairs, modelID_dict, False)
# #         # results_HD = evaluateResults(potentialPairs_HD, combined_final_pairs_HD, modelID_dict, False)
# #         # combined_results_HDplus = potentialPairs_HDplus[1] + [signature_matrix_HDplus[2]]
# #         # results_HDplus = combined_results_HDplus
# #
# #         # with open("results.csv", 'a') as out:
# #         #     out.write(str(bands_rows[0]))
# #         #     out.write("," + str(bands_rows[1]))
# #         #
# #         #     for stat in results_base:
# #         #         out.write("," + str(stat))
# #         #
# #         #     for stat in results_HD:
# #         #         out.write("," + str(stat))
# #         #
# #         #     for stat in results_HDplus:
# #         #         out.write("," + str(stat))
# #         #     out.write("\n")
# #
# #     statistics = results_base / no_bootstrap
# #     statistics_HD = results_HD / no_bootstrap
# #     statistics_HDplus = results_HDplus / no_bootstrap
# #
# #     with open("results.csv", 'a') as out:
# #         out.write(str(bands_rows[0]))
# #         out.write("," + str(bands_rows[1]))
# #
# #         for stat in statistics:
# #             out.write("," + str(stat))
# #
# #         for stat in statistics_HD:
# #             out.write("," + str(stat))
# #
# #         for stat in statistics_HDplus:
# #             out.write("," + str(stat))
# #         out.write("\n")
#
