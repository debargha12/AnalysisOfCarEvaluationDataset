from DecisionTree import *
import pandas as pd
from sklearn import model_selection
import random

header = ['buying ', 'maint', 'doors', 'persons', 'lug_boot','safety']
df = pd.read_csv('car.csv', header=None, names=['buying ', 'maint', 'doors', 'persons', 'lug_boot','safety'])
lst = df.values.tolist()

t = build_tree(lst, header)
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))

## TODO: You have to decide on a pruning strategy
sample=[]
for inner in innerNodes:
    if inner.id !=0:
        sample.append(inner.id)
import random
random.shuffle(sample)
t_pruned = prune_tree(t, sample[:7])

print("*************Tree after pruning*******")
print_tree(t_pruned)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))