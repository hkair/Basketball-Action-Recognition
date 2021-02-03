import matplotlib.pyplot as plt
import json

def action_distribution(annotation_paths, labels_path):
    annotation_dict = {}
    for annotations in annotation_paths:
        with open(annotations) as f:
            x = json.load(f).items()
            annotation_dict.update(x)

    with open(labels_path) as f:
        labels_dict = json.load(f, object_hook=keystoint)

    # Let's first visualize the distribution of actions in the
    count_dict = dict()
    for key in annotation_dict:
        if labels_dict[annotation_dict[key]] in count_dict:
            count_dict[labels_dict[annotation_dict[key]]] += 1
        else:
            count_dict[labels_dict[annotation_dict[key]]] = 1

    sorted_dict = {k: v for k, v in sorted(count_dict.items(), key=lambda item: item[1])}

    return sorted_dict

def keystoint(x):
    return {int(k): v for k, v in x.items()}

if __name__ == "__main__":

    # # Read Dictionary from dataset
    # with open('../dataset/annotation_dict.json') as f:
    #     annotation_dict = json.load(f)
    #
    # def keystoint(x):
    #     return {int(k): v for k, v in x.items()}
    #
    # with open('../dataset/labels_dict.json') as f:
    #     labels_dict = json.load(f, object_hook=keystoint)
    #
    # # Let's first visualize the distribution of actions in the
    # count_dict = dict()
    # for key in annotation_dict:
    #     if labels_dict[annotation_dict[key]] in count_dict:
    #         count_dict[labels_dict[annotation_dict[key]]] += 1
    #     else:
    #         count_dict[labels_dict[annotation_dict[key]]] = 1

    annotation_paths = ['../dataset/annotation_dict.json', '../dataset/augmented_annotation_dict.json']
    sorted_dict = action_distribution(annotation_paths, '../dataset/labels_dict.json')

    plt.bar(range(len(sorted_dict)), list(sorted_dict.values()), align='center')
    plt.xticks(range(len(sorted_dict)), list(sorted_dict.keys()))
    plt.show()