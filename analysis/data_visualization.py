import matplotlib.pyplot as plt
import json

# Read Dictionary from dataset
with open('./dataset/annotation_dict.json') as f:
    annotation_dict = json.load(f)

def keystoint(x):
    return {int(k): v for k, v in x.items()}

with open('./dataset/labels_dict.json') as f:
    labels_dict = json.load(f, object_hook=keystoint)

# Let's first visualize the distribution of actions in the
count_dict = dict()
for key in annotation_dict:
    if labels_dict[annotation_dict[key]] in count_dict:
        count_dict[labels_dict[annotation_dict[key]]] += 1
    else:
        count_dict[labels_dict[annotation_dict[key]]] = 1

sorted_dict = {k: v for k, v in sorted(count_dict.items(), key=lambda item: item[1])}

print(count_dict)
print("Length of Annotations:")
print(len(annotation_dict))
print("Length of counts (Annotations):")
print(sum(count_dict.values()))

plt.bar(range(len(sorted_dict)), list(sorted_dict.values()), align='center')
plt.xticks(range(len(sorted_dict)), list(sorted_dict.keys()))
plt.show()