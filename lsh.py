# This is the code for the LSH project of TDT4305

import configparser
import itertools  # for reading the parameters file
import sys  # for system errors and printouts
from pathlib import Path  # for paths of files
import os  # for reading the input data
import time  # for timing
import numpy as np  # for matrix operations
import random  # for random numbers

from random import randint

random.seed(100)  # for reproducibility

# Global parameters
parameter_file = 'default_parameters.ini'  # the main parameters file
data_main_directory = Path('data')  # the main path were all the data directories are
parameters_dictionary = dict()  # dictionary that holds the input parameters, key = parameter name, value = value
document_list = dict()  # dictionary of the input documents, key = document id, value = the document


# DO NOT CHANGE THIS METHOD
# Reads the parameters of the project from the parameter file 'file'
# and stores them to the parameter dictionary 'parameters_dictionary'
def read_parameters():
    config = configparser.ConfigParser()
    config.read(parameter_file)
    for section in config.sections():
        for key in config[section]:
            if key == 'data':
                parameters_dictionary[key] = config[section][key]
            elif key == 'naive':
                parameters_dictionary[key] = bool(config[section][key])
            elif key == 't':
                parameters_dictionary[key] = float(config[section][key])
            else:
                parameters_dictionary[key] = int(config[section][key])


# DO NOT CHANGE THIS METHOD
# Reads all the documents in the 'data_path' and stores them in the dictionary 'document_list'
def read_data(data_path):
    for (root, dirs, file) in os.walk(data_path):
        for f in file:
            file_path = data_path / f
            doc = open(file_path).read().strip().replace('\n', ' ')
            file_id = int(file_path.stem)
            document_list[file_id] = doc


# DO NOT CHANGE THIS METHOD
# Calculates the Jaccard Similarity between two documents represented as sets
def jaccard(doc1, doc2):
    return len(doc1.intersection(doc2)) / float(len(doc1.union(doc2)))


# DO NOT CHANGE THIS METHOD
# Define a function to map a 2D matrix coordinate into a 1D index.
def get_triangle_index(i, j, length):
    if i == j:  # that's an error.
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    if j < i:  # just swap the values.
        temp = i
        i = j
        j = temp

    # Calculate the index within the triangular array. Taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # adapted for a 0-based index.
    k = int(i * (length - (i + 1) / 2.0) + j - i) - 1

    return k


# DO NOT CHANGE THIS METHOD
# Calculates the similarities of all the combinations of documents and returns the similarity triangular matrix
def naive():
    docs_Sets = []  # holds the set of words of each document

    for doc in document_list.values():
        docs_Sets.append(set(doc.split()))

    # Using triangular array to store the similarities, avoiding half size and similarities of i==j
    num_elems = int(len(docs_Sets) * (len(docs_Sets) - 1) / 2)
    similarity_matrix = [0 for x in range(num_elems)]
    for i in range(len(docs_Sets)):
        for j in range(i + 1, len(docs_Sets)):
            similarity_matrix[get_triangle_index(i, j, len(docs_Sets))] = jaccard(docs_Sets[i], docs_Sets[j])

    return similarity_matrix


# METHOD FOR TASK 1
# Creates the k-Shingles of each document and returns a list of them
def k_shingles():
    k = parameters_dictionary['k']
    docs_k_shingles = []
    for i in range(1,len(document_list)+1):
        doc = document_list[i]
        doc_k_shingles = []
        for j in range(len(doc) - k + 1):
            doc_k_shingles.append(doc[j:j + k])
        docs_k_shingles.append(doc_k_shingles)
    return docs_k_shingles


# METHOD FOR TASK 2
# Creates a signatures set of the documents from the k-shingles list
def signature_set(k_shingles):
    docs_signature_sets = []
    hash_values = []
    for i in range(0,len(document_list)):
        print(i)
        doc = k_shingles[i]
        doc_signature_set = set()
        for j in range(len(doc)):
            hash_value = hash(doc[j])
            if hash_value not in hash_values:
                hash_values.append(hash_value)
                doc_signature_set.add(hash_value)
        docs_signature_sets.append(doc_signature_set)
    return docs_signature_sets


# METHOD FOR TASK 3
# Creates the minHash signatures after simulation of permutations
def minHash(docs_signature_sets):
    
    num_hash_functions = parameters_dictionary['permutations']
    num_docs = len(document_list)


    # Generate random coefficients for each hash function
    coeffs = []
    for i in range(num_hash_functions):
        a = randint(1, pow(2, 32) - 1)
        b = randint(0, pow(2, 32) - 1)
        p = parameters_dictionary['permutations']
        coeffs.append((a, b, p))

    # Compute the MinHash signature matrix
    signature_matrix = np.full((num_hash_functions, num_docs), np.inf)
    for i in range(num_hash_functions):
        for j in range(num_docs):
            for hash_value in docs_signature_sets[j]:
                # Apply a random permutation to the hash value
                permuted_hash_val = (coeffs[i][0] * hash_value + coeffs[i][1]) % coeffs[i][2]
                # Update the current minimum hash value
                if permuted_hash_val < signature_matrix[i][j]:
                    signature_matrix[i][j] = permuted_hash_val

    return signature_matrix

#Når vi burker BBC må vi ha større p verdi
# METHOD FOR TASK 4
# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents

def lsh(m_matrix): 
    candidates = []  # list of candidate sets of documents for checking similarity
    r = parameters_dictionary['r'] 
    dics = []
    for shingle in range(0, len(m_matrix), r):
        dic_key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        my_dic = {key: [] for key in dic_key}
        for document in range(len(m_matrix[0])):
            x = (m_matrix[shingle][document]*10+m_matrix[shingle+1][document]) % parameters_dictionary['buckets']
            my_dic[x].append(document+1)
        for key in my_dic:
            if len(my_dic[key]) > 1:
                pairs = list(itertools.combinations(my_dic[key], 2))
                for pair in pairs:
                    if pair not in candidates:
                        candidates.append(pair)
        dics.append(my_dic)
    return candidates


# METHOD FOR TASK 5
# Calculates the similarities of the candidate documents
def candidates_similarities(candidate_docs, min_hash_matrix):
    similarity_matrix = []
    for pair in candidate_docs:
        jaccard_similarity = jaccard(set(min_hash_matrix[pair[0]]), set(min_hash_matrix[pair[1]]))
        similarity_matrix.append((pair[0], pair[1], jaccard_similarity))
    return similarity_matrix


# METHOD FOR TASK 6
# Returns the document pairs of over t% similarity
def return_results(lsh_similarity_matrix):
    document_pairs = []
    print("")
    for row in lsh_similarity_matrix:
        if row[2] > parameters_dictionary['t']:
            document_pairs.append((row[0], row[1]))
            print("Dokumentpar:", row[0], "og", row[1], "Likhet:", row[2])
    print("Antall matchene dokumenterpar:", len(document_pairs))
    return document_pairs


# METHOD FOR TASK 6
def count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix):
    false_negatives = 0
    false_positives = 0

    # implement your code here
    for row in lsh_similarity_matrix:
        if row[2] > parameters_dictionary['t']:
            if row not in naive_similarity_matrix:
                false_positives += 1
    for row in naive_similarity_matrix:
        if row not in lsh_similarity_matrix:
            false_negatives += 1
    
    return false_negatives, false_positives


# DO NOT CHANGE THIS METHOD
# The main method where all code starts
if __name__ == '__main__':
    # Reading the parameters
    read_parameters()

    # Reading the data
    print("Data reading...")
    data_folder = data_main_directory / parameters_dictionary['data']
    t0 = time.time()
    read_data(data_folder)
    document_list = {k: document_list[k] for k in sorted(document_list)}
    t1 = time.time()
    print(len(document_list), "documents were read in", t1 - t0, "sec\n")

    # Naive
    naive_similarity_matrix = []
    if parameters_dictionary['naive']:
        print("Starting to calculate the similarities of documents...")
        t2 = time.time()
        naive_similarity_matrix = naive()
        t3 = time.time()
        print("Calculating the similarities of", len(naive_similarity_matrix),
              "combinations of documents took", t3 - t2, "sec\n")

    # k-Shingles
    print("Starting to create all k-shingles of the documents...")
    t4 = time.time()
    all_docs_k_shingles = k_shingles()
    t5 = time.time()
    print("Representing documents with k-shingles took", t5 - t4, "sec\n")

    # signatures sets
    print("Starting to create the signatures of the documents...")
    t6 = time.time()
    signature_sets = signature_set(all_docs_k_shingles)
    t7 = time.time()
    print("Signatures representation took", t7 - t6, "sec\n")

    # Permutations
    print("Starting to simulate the MinHash Signature Matrix...")
    t8 = time.time()
    min_hash_signatures = minHash(signature_sets)
    t9 = time.time()
    print("Simulation of MinHash Signature Matrix took", t9 - t8, "sec\n")

    # LSH
    print("Starting the Locality-Sensitive Hashing...")
    t10 = time.time()
    candidate_docs = lsh(min_hash_signatures)
    t11 = time.time()
    print("LSH took", t11 - t10, "sec\n")

    # Candidate similarities
    print("Starting to calculate similarities of the candidate documents...")
    t12 = time.time()
    lsh_similarity_matrix = candidates_similarities(candidate_docs, min_hash_signatures)
    t13 = time.time()
    print("Candidate documents similarity calculation took", t13 - t12, "sec\n\n")

    # Return the over t similar pairs
    print("Starting to get the pairs of documents with over ", parameters_dictionary['t'], "% similarity...")
    t14 = time.time()
    pairs = return_results(lsh_similarity_matrix)
    t15 = time.time()
    print("The pairs of documents are:\n")
    for p in pairs:
        print(p)
    print("\n")

    # Count false negatives and positives
    if parameters_dictionary['naive']:
        print("Starting to calculate the false negatives and positives...")
        t16 = time.time()
        false_negatives, false_positives = count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix)
        t17 = time.time()
        print("False negatives = ", false_negatives, "\nFalse positives = ", false_positives, "\n\n")

    if parameters_dictionary['naive']:
        print("Naive similarity calculation took", t3 - t2, "sec")

    print("LSH process took in total", t13 - t4, "sec")
