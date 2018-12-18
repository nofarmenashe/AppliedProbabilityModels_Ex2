# Writen By:
# Yuval Maymon - 315806299
# Nofar Menashe - 205486210

import sys
import math
import collections


def get_articles_from_file(filename):
    file = open(filename, "r")
    lines = file.readlines()

    articles = []
    for line in lines:
        if line[:6] != '<TRAIN' and line[:5] != '<TEST':
            articles.append(line)

    return articles


def get_all_words_in_articles(articles):
    words = []
    for article in articles:
        for word in article.split():
            words.append(word)

    return words


# Lidstone Model
def MLE(event, dataset_collection, dataset_size):
    event_count = dataset_collection[event]
    return event_count / dataset_size


def lidstone_unigram_model(lambda_param, event, dataset_collection, dataset_size):
    mle_event = MLE(event, dataset_collection, dataset_size)
    miu = dataset_size / (dataset_size + (lambda_param * VOCABULARY_SIZE))
    P_lid = (miu * mle_event) + ((1 - miu) * (1 / VOCABULARY_SIZE))
    return P_lid


def get_lambda_with_min_perplexity(test_set, train_set_collection, train_set_size):
    lambda_array = [i / 100. for i in range(1, 200)]
    min_perplexity = math.inf
    min_lambda = None

    for lambda_param in lambda_array:
        curr_perplexity = lidstone_perplexity(lambda_param, test_set, train_set_collection, train_set_size)

        if curr_perplexity < min_perplexity:
            min_perplexity = curr_perplexity
            min_lambda = lambda_param

    return min_lambda, min_perplexity


def lidstone_perplexity(lambda_param, test_set, train_set_collection, train_set_size):
    sum_of_logs = 0
    for word in test_set:
        P_lidstone = lidstone_unigram_model(lambda_param, word, train_set_collection, train_set_size)
        sum_of_logs += math.log(P_lidstone, 2)

    return math.pow(2, -(1 / len(test_set)) * sum_of_logs)


def check_sum_of_lidstone_probabilities(lambda_param, train_set_collection, train_set_size):
    n0 = VOCABULARY_SIZE - len(training_set_collection)
    sum_of_probabilities = n0 * lidstone_unigram_model(lambda_param, UNSEEN_WORD, train_set_collection,
                                                       training_set_size)

    for word in train_set_collection.keys():
        sum_of_probabilities += lidstone_unigram_model(lambda_param, word, train_set_collection, train_set_size)

    print("on Lidstone Model with lambda", lambda_param,"sum of probabilities is: ", sum_of_probabilities)


# Heldout Model
def held_out_model(event, S_t_collection, S_h_collection, S_h_size, vocabulary_size):
    r = S_t_collection[event]
    t_r_sum = 0
    N_r = 0

    for event_S_t in S_t_collection:
        if S_t_collection[event_S_t] == r:
            N_r += 1
            t_r_sum += S_h_collection[event_S_t]

    if r == 0:
        N_r = vocabulary_size - len(S_t_collection)
        difference_set = S_h_collection - S_t_collection
        for word in difference_set:
            t_r_sum += S_h_collection[word]

    return (float(t_r_sum) / N_r) / S_h_size


def heldout_perplexity(test_set, S_t_collection, S_h_collection, S_h_size, vocabulary_size):
    sum_of_logs = 0
    for word in test_set:
        P_heldout = held_out_model(word, S_t_collection, S_h_collection, S_h_size, vocabulary_size)
        sum_of_logs += math.log(P_heldout, 2)

    return math.pow(2, -(1 / len(test_set)) * sum_of_logs)


def check_sum_of_heldout_probabilities(S_t_collection, S_h_collection, S_h_size, vocabulary_size):
    n0 = VOCABULARY_SIZE - len(S_t_collection)
    sum_of_probabilities = n0 * held_out_model(UNSEEN_WORD, S_t_collection, S_h_collection, S_h_size, vocabulary_size)

    for word in S_t_collection.keys():
        sum_of_probabilities += held_out_model(word, S_t_collection, S_h_collection, S_h_size, vocabulary_size)

    print("on Heldout Model sum of probabilities is: ", sum_of_probabilities)


if __name__ == "__main__":

    development_set_filename = sys.argv[1]
    test_set_filename = sys.argv[2]
    INPUT_WORD = sys.argv[3].lower()
    output_filename = sys.argv[4]

    VOCABULARY_SIZE = 300000  # number of events
    UNSEEN_WORD = 'unseen-word'

    developmentArticles = get_articles_from_file(development_set_filename)
    development_set_words = get_all_words_in_articles(developmentArticles)

    testArticles = get_articles_from_file(test_set_filename)
    test_set_words = get_all_words_in_articles(testArticles)

    outputs = [None for _ in range(29)]

    # 1. Init
    outputs[1] = development_set_filename
    outputs[2] = test_set_filename
    outputs[3] = INPUT_WORD
    outputs[4] = output_filename
    outputs[5] = VOCABULARY_SIZE

    P_uniform = 1 / VOCABULARY_SIZE
    outputs[6] = P_uniform

    # 2. Development set preprocessing
    outputs[7] = len(development_set_words)

    # 3. Lindstone model training
    training_set_size = round(0.9 * len(development_set_words))

    training_set = development_set_words[:training_set_size]
    validation_set = development_set_words[training_set_size:]

    training_set_collection = collections.Counter(training_set)

    outputs[8] = len(validation_set)
    outputs[9] = len(training_set)
    outputs[10] = len(training_set_collection)
    outputs[11] = training_set_collection[INPUT_WORD]

    outputs[12] = MLE(INPUT_WORD, training_set_collection, len(training_set))
    outputs[13] = MLE(UNSEEN_WORD, training_set_collection, len(training_set))

    outputs[14] = lidstone_unigram_model(0.1, INPUT_WORD, training_set_collection, len(training_set))
    outputs[15] = lidstone_unigram_model(0.1, UNSEEN_WORD, training_set_collection, len(training_set))

    outputs[16] = lidstone_perplexity(0.01, validation_set, training_set_collection, len(training_set))
    outputs[17] = lidstone_perplexity(0.1, validation_set, training_set_collection, len(training_set))
    outputs[18] = lidstone_perplexity(1, validation_set, training_set_collection, len(training_set))

    best_lambda, min_perplexity = get_lambda_with_min_perplexity(validation_set, training_set_collection,
                                                                 len(training_set))
    outputs[19] = best_lambda
    outputs[20] = min_perplexity

    # 4. Held out model training
    held_out_training_set_size = round(0.5 * len(development_set_words))

    S_t = development_set_words[:held_out_training_set_size]
    S_t_collection = collections.Counter(S_t)

    S_h = development_set_words[held_out_training_set_size:]
    S_h_collection = collections.Counter(S_h)

    outputs[21] = len(S_t)
    outputs[22] = len(S_h)

    outputs[21] = len(S_t)
    outputs[22] = len(S_h)

    outputs[23] = held_out_model(INPUT_WORD, S_t_collection, S_h_collection, len(S_h), VOCABULARY_SIZE)
    outputs[24] = held_out_model(UNSEEN_WORD, S_t_collection, S_h_collection, len(S_h), VOCABULARY_SIZE)

    # 5. debug models
    check_sum_of_lidstone_probabilities(0.1, training_set_collection, len(training_set))
    check_sum_of_heldout_probabilities(S_t_collection, S_h_collection, len(S_h), VOCABULARY_SIZE)

    # 6.
    outputs[25] = len(test_set_words)

    lidstone_perplexity = lidstone_perplexity(best_lambda, test_set_words, training_set_collection, len(training_set))
    outputs[26] = lidstone_perplexity
    print("start heldout proplexity")
    heldout_perplexity = heldout_perplexity(test_set_words, S_t_collection, S_h_collection, len(S_h), VOCABULARY_SIZE)
    print("end heldout proplexity")

    outputs[27] = heldout_perplexity

    outputs[28] = 'L' if lidstone_perplexity > heldout_perplexity else 'H'

    # Write final output
    output_string = "#Student\tYuval Maymon\tNofar Menashe\t315806299\t 205486210\n"

    for index, output in enumerate(outputs[1:]):
        output_string += "#Output" + str(index + 1) + "\t" + str(output) + "\n"

    with open(output_filename, 'w') as outputFile:
        outputFile.write(output_string)
