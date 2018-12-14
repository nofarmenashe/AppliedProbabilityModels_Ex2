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


def number_of_different_events(dataset):
    distinct_dataset = list(set(dataset))
    return len(distinct_dataset)

def MLE(event, dataset_collection, dataset_size):
    event_count = dataset_collection[event]
    return event_count / dataset_size


def lidstone_unigram_model(lambda_param, event, dataset_collection, dataset_size):
    mle_event = MLE(event, dataset_collection, dataset_size)
    miu = dataset_size / (dataset_size + (lambda_param * VOCABULARY_SIZE))
    P_lid = miu * mle_event + (1 - miu) * (1 / VOCABULARY_SIZE)
    return P_lid


def perplexity(lambda_param, test_set, train_set_collection, train_set_size):
    sum_of_logs = 0
    for word in test_set:
        P_lidstone = lidstone_unigram_model(lambda_param, word, train_set_collection, train_set_size)
        sum_of_logs += math.log(P_lidstone, 2)

    return math.pow(2, -(1 / len(test_set)) * sum_of_logs)


if __name__ == "__main__":

    development_set_filename = sys.argv[1]
    test_set_filename = sys.argv[2]
    INPUT_WORD = sys.argv[3].lower()
    output_filename = sys.argv[4]

    VOCABULARY_SIZE = 300000  # number of events
    UNSEEN_WORD = 'unseen-word'

    developmentArticles = get_articles_from_file(development_set_filename)
    development_set_words = get_all_words_in_articles(developmentArticles)

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

    outputs[16] = perplexity(0.01, validation_set, training_set_collection, len(training_set))
    outputs[17] = perplexity(0.1, validation_set, training_set_collection, len(training_set))
    outputs[18] = perplexity(1, validation_set, training_set_collection, len(training_set))


    # Write final output
    output_string = "#Student\tYuval Maymon\tNofar Menashe\t315806299\t 205486210\n"

    for index, output in enumerate(outputs[1:]):
        output_string += "#Output" + str(index+1) + "\t" + str(output) + "\n"

    with open(output_filename, 'w') as outputFile:
        outputFile.write(output_string)
