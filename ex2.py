# Wriiten By:
# Yuval Maymon - 315806299
# Nofar Menashe - 205486210

import sys


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


def number_of_times_event_appear(event, dataset):
    return dataset.count(event)


if __name__ == "__main__":

    development_set_filename = sys.argv[1]
    test_set_filename = sys.argv[2]
    INPUT_WORD = sys.argv[3].lower()
    output_filename = sys.argv[4]

    VOCABULARY_SIZE = 300000

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

    outputs[8] = len(validation_set)
    outputs[9] = len(training_set)
    outputs[10] = number_of_different_events(training_set)
    outputs[11] = number_of_times_event_appear(INPUT_WORD, training_set)

    # Write final output
    output_string = "#Student\tYuval Maymon\tNofar Menashe\t315806299\t 205486210\n"

    for index, output in enumerate(outputs[1:]):
        output_string += "#Output" + str(index+1) + "\t" + str(output) + "\n"

    with open(output_filename, 'w') as outputFile:
        outputFile.write(output_string)
