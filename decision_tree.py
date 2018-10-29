import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

col_names = ["age", "workclass", "education",
             "marital-status", "occupation", "race",
             "sex", "hours-per-week",
             "country", "income"]
train_data = pd.read_csv("income.train.txt", header=None, names=col_names)
test_data = pd.read_csv("income.test.txt", header=None, names=col_names)
dev_data = pd.read_csv("income.dev.txt", header= None, names=col_names)

X_train = train_data.drop('income',axis=1)
Y_train = train_data['income']

X_test = test_data.drop('income',axis=1)
Y_test = test_data['income']

X_dev = dev_data.drop('income',axis=1)
Y_dev = dev_data['income']

data = pd.concat([X_train, X_test])
data_1 = pd.concat([X_train, X_dev])

data_ohe = pd.get_dummies(data)
data_ohe1 = pd.get_dummies(data_1)
X_train_ohe = data_ohe[:len(X_train)]
X_test_ohe = data_ohe[len(X_train):]
X_dev_ohe = data_ohe1[len(X_train):]
y_train_ohe = Y_train.replace([' <=50K', ' >50K'], [-1, 1])
y_test_ohe = Y_test.replace([' <=50K', ' >50K'], [-1, 1])
y_dev_ohe = Y_dev.replace([' <=50K', ' >50K'], [-1, 1])
X_train = np.array(X_train_ohe)
Y_train = np.array(y_train_ohe)
X_test  = np.array(X_test_ohe)
Y_test  = np.array(y_test_ohe)
X_dev = np.array(X_dev_ohe)
Y_dev= np.array(y_dev_ohe)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)
print("\nThe training accuracy is")
y_pred = classifier.predict(X_train)
print(np.mean(y_pred==Y_train)*100)

print("\nThe testing accuracy is")
y_pred = classifier.predict(X_test)
print(np.mean(y_pred==Y_test)*100)
test_accuracy=np.mean(y_pred==Y_test)*100

print("\nThe validation accuracy is")
y_pred = classifier.predict(X_dev)
print(np.mean(y_pred==Y_dev)*100)
#print(classification_report(Y_test, y_pred))

n_nodes = classifier.tree_.node_count
child_left = classifier.tree_.children_left
child_right = classifier.tree_.children_right
f = classifier.tree_.feature
t = classifier.tree_.threshold


# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    depth[node_id] = parent_depth + 1

    # If we have a test node
    if (child_left[node_id] != child_right[node_id]):
        stack.append((child_left[node_id], parent_depth + 1))
        stack.append((child_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print(" \nThe binary tree structure has %s nodes"
      % n_nodes)

node_indicator = classifier.decision_path(X_test)

# Similarly, we can also have the leaves ids reached by each sample.

leave_id = classifier.apply(X_test)

# Now, it's possible to get the tests that were used to predict a sample or
# a group of samples. First, let's make it for the sample.

sample_id = 0
node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]


#print(test_accuracy)
max_test_accuracy_pruning=0
depth=classifier.tree_.max_depth
test_error=[]
train_error=[]
validation_error=[]
test_nodes=[]
train_nodes=[]
validation_nodes=[]
dep=[]
print("After Pruning")
print("\nTesting accuracy is")
for i in range(1,depth):
    dep.append(i)
    classifier = DecisionTreeClassifier(max_depth=i)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    #print(np.mean(y_pred==Y_test)*100)
    test_accuracy=np.mean(y_pred==Y_test)*100
    test_nodes.append(classifier.tree_.node_count)
    test_error.append(100-test_accuracy)
    if test_accuracy>max_test_accuracy_pruning:
        max_test_accuracy_pruning = test_accuracy
print(max_test_accuracy_pruning)
#print(test_error)
#print(test_nodes)
max_train_accuracy_pruning =0
print("\nTraining")
for i in range(1,depth):
    classifier = DecisionTreeClassifier(max_depth= i)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_train)
    #print(np.mean(y_pred==Y_train)*100)
    train_accuracy=np.mean(y_pred==Y_train)*100
    train_nodes.append(classifier.tree_.node_count)
    train_error.append(100-train_accuracy)
    if train_accuracy>max_train_accuracy_pruning:
        max_train_accuracy_pruning = train_accuracy
print(max_train_accuracy_pruning)
#print(train_error)
#print(train_nodes)
max_dev_accuracy_pruning =0
print("\nValidation")
for i in range(1,depth):
    classifier = DecisionTreeClassifier(max_depth=i)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_dev)
    #print(np.mean(y_pred==Y_dev)*100)
    dev_accuracy=np.mean(y_pred==Y_dev)*100
    validation_nodes.append(classifier.tree_.node_count)
    validation_error.append(100-dev_accuracy)
    if dev_accuracy>max_dev_accuracy_pruning:
        max_dev_accuracy_pruning = dev_accuracy
print(max_dev_accuracy_pruning)
#print(validation_error)
#print(validation_nodes)
plt.xlabel("Number of nodes")
plt.ylabel("Error")
plt.plot(train_nodes,train_error,label = 'Train Error')
plt.plot(test_nodes,test_error, label = 'Test Error')
plt.plot(validation_nodes,validation_error, label = 'Validation Error')
plt.legend()
plt.show()

plt.xlabel("Depth of the tree")
plt.ylabel("Error")
plt.plot(dep,train_error,label = 'Train Error')
plt.plot(dep,test_error,label = 'Test Error')
plt.plot(dep,validation_error,label = 'Validation Error')
plt.legend()
plt.show()
