import sys


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


    # 3. Lindstone model training
    development_set = [None for _ in range(29)] # change to real development set
    training_set_size = round(0.9 * len(development_set))

    training_set = development_set[:training_set_size]
    validation_set = development_set[training_set_size:]

    outputs[8] = len(validation_set)
    outputs[9] = len(training_set)
    outputs[10] = number_of_different_events(training_set)
    outputs[11] = number_of_times_event_appear(INPUT_WORD, training_set)

    # Write final output
    outputString = "#Student\tYuval Maymon\tNofar Menashe\t315806299\t 205486210\n"

    for index, output in enumerate(outputs[1:]):
        outputString += "#Output" + str(index+1) + "\t" + str(output) + "\n"

    with open(output_filename, 'w') as outputFile:
        outputFile.write(outputString)
