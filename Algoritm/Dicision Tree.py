import numpy as np


def tree_in_iris(x):
    if x[3] < 0.8:
        return 0
    elif x[3] < 1.75:
        if x[2] < 4.85:
            return 1
        elif x[2] >= 5.1:
            return 2
        else:
            return 1
    else:
        return 2


petal_width = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2,
     0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2,
     0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1., 1.3, 1.4, 1., 1.5,
     1., 1.4, 1.3, 1.4, 1.5, 1., 1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7,
     1.5, 1., 1.1, 1., 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3, 1.2,
     1.4, 1.2, 1., 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8,
     2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2., 1.9, 2.1, 2., 2.4, 2.3, 1.8,
     2.2, 2.3, 1.5, 2.3, 2., 2., 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6,
     1.9, 2., 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9,
     2.3, 2.5, 2.3, 1.9, 2., 2.3, 1.8])
petal_length = np.array([1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4,
                         1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1., 1.7, 1.9, 1.6,
                         1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.5, 1.3,
                         1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5,
                         4.9, 4., 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4., 4.7, 3.6,
                         4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4., 4.9, 4.7, 4.3, 4.4, 4.8, 5.,
                         4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4., 4.4,
                         4.6, 4., 3.3, 4.2, 4.2, 4.2, 4.3, 3., 4.1, 6., 5.1, 5.9, 5.6,
                         5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5., 5.1, 5.3, 5.5,
                         6.7, 6.9, 5., 5.7, 4.9, 6.7, 4.9, 5.7, 6., 4.8, 4.9, 5.6, 5.8,
                         6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1,
                         5.9, 5.7, 5.2, 5., 5.2, 5.4, 5.1])
sepal_width = np.array([3.5, 3., 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.,
                        3., 4., 4.4, 3.9, 3.5, 3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3.,
                        3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1, 4.2, 3.1, 3.2, 3.5, 3.1, 3.,
                        3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3., 3.8, 3.2, 3.7, 3.3, 3.2, 3.2,
                        3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2., 3., 2.2, 2.9, 2.9,
                        3.1, 3., 2.7, 2.2, 2.5, 3.2, 2.8, 2.5, 2.8, 2.9, 3., 2.8, 3.,
                        2.9, 2.6, 2.4, 2.4, 2.7, 2.7, 3., 3.4, 3.1, 2.3, 3., 2.5, 2.6,
                        3., 2.6, 2.3, 2.7, 3., 2.9, 2.9, 2.5, 2.8, 3.3, 2.7, 3., 2.9,
                        3., 3., 2.5, 2.9, 2.5, 3.6, 3.2, 2.7, 3., 2.5, 2.8, 3.2, 3.,
                        3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8, 3., 2.8, 3.,
                        2.8, 3.8, 2.8, 2.8, 2.6, 3., 3.4, 3.1, 3., 3.1, 3.1, 3.1, 2.7,
                        3.2, 3.3, 3., 2.5, 3., 3.4, 3.])
sepal_length = np.array([5.1, 4.9, 4.7, 4.6, 5., 5.4, 4.6, 5., 4.4, 4.9, 5.4, 4.8, 4.8,
                         4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5.,
                         5., 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5., 5.5, 4.9, 4.4,
                         5.1, 5., 4.5, 4.4, 5., 5.1, 4.8, 5.1, 4.6, 5.3, 5., 7., 6.4,
                         6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5., 5.9, 6., 6.1, 5.6,
                         6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7,
                         6., 5.7, 5.5, 5.5, 5.8, 6., 5.4, 6., 6.7, 6.3, 5.6, 5.5, 5.5,
                         6.1, 5.8, 5., 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3,
                         6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5,
                         7.7, 7.7, 6., 6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2,
                         7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6., 6.9, 6.7, 6.9, 5.8,
                         6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9])

target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])


def gini(array):
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    # Gini coefficient:
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


print(gini(petal_width))
print(gini(petal_length))
print(gini(sepal_width))
print(gini(sepal_length))


# Decision Tree Class

data = np.array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],
       [4.7, 3.2, 1.3, 0.2],
       [4.6, 3.1, 1.5, 0.2],
       [5. , 3.6, 1.4, 0.2],
       [5.4, 3.9, 1.7, 0.4],
       [4.6, 3.4, 1.4, 0.3],
       [5. , 3.4, 1.5, 0.2],
       [4.4, 2.9, 1.4, 0.2],
       [4.9, 3.1, 1.5, 0.1],
       [5.4, 3.7, 1.5, 0.2],
       [4.8, 3.4, 1.6, 0.2],
       [4.8, 3. , 1.4, 0.1],
       [4.3, 3. , 1.1, 0.1],
       [5.8, 4. , 1.2, 0.2],
       [5.7, 4.4, 1.5, 0.4],
       [5.4, 3.9, 1.3, 0.4],
       [5.1, 3.5, 1.4, 0.3],
       [5.7, 3.8, 1.7, 0.3],
       [5.1, 3.8, 1.5, 0.3],
       [5.4, 3.4, 1.7, 0.2],
       [5.1, 3.7, 1.5, 0.4],
       [4.6, 3.6, 1. , 0.2],
       [5.1, 3.3, 1.7, 0.5],
       [4.8, 3.4, 1.9, 0.2],
       [5. , 3. , 1.6, 0.2],
       [5. , 3.4, 1.6, 0.4],
       [5.2, 3.5, 1.5, 0.2],
       [5.2, 3.4, 1.4, 0.2],
       [4.7, 3.2, 1.6, 0.2],
       [4.8, 3.1, 1.6, 0.2],
       [5.4, 3.4, 1.5, 0.4],
       [5.2, 4.1, 1.5, 0.1],
       [5.5, 4.2, 1.4, 0.2],
       [4.9, 3.1, 1.5, 0.1],
       [5. , 3.2, 1.2, 0.2],
       [5.5, 3.5, 1.3, 0.2],
       [4.9, 3.1, 1.5, 0.1],
       [4.4, 3. , 1.3, 0.2],
       [5.1, 3.4, 1.5, 0.2],
       [5. , 3.5, 1.3, 0.3],
       [4.5, 2.3, 1.3, 0.3],
       [4.4, 3.2, 1.3, 0.2],
       [5. , 3.5, 1.6, 0.6],
       [5.1, 3.8, 1.9, 0.4],
       [4.8, 3. , 1.4, 0.3],
       [5.1, 3.8, 1.6, 0.2],
       [4.6, 3.2, 1.4, 0.2],
       [5.3, 3.7, 1.5, 0.2],
       [5. , 3.3, 1.4, 0.2],
       [7. , 3.2, 4.7, 1.4],
       [6.4, 3.2, 4.5, 1.5],
       [6.9, 3.1, 4.9, 1.5],
       [5.5, 2.3, 4. , 1.3],
       [6.5, 2.8, 4.6, 1.5],
       [5.7, 2.8, 4.5, 1.3],
       [6.3, 3.3, 4.7, 1.6],
       [4.9, 2.4, 3.3, 1. ],
       [6.6, 2.9, 4.6, 1.3],
       [5.2, 2.7, 3.9, 1.4],
       [5. , 2. , 3.5, 1. ],
       [5.9, 3. , 4.2, 1.5],
       [6. , 2.2, 4. , 1. ],
       [6.1, 2.9, 4.7, 1.4],
       [5.6, 2.9, 3.6, 1.3],
       [6.7, 3.1, 4.4, 1.4],
       [5.6, 3. , 4.5, 1.5],
       [5.8, 2.7, 4.1, 1. ],
       [6.2, 2.2, 4.5, 1.5],
       [5.6, 2.5, 3.9, 1.1],
       [5.9, 3.2, 4.8, 1.8],
       [6.1, 2.8, 4. , 1.3],
       [6.3, 2.5, 4.9, 1.5],
       [6.1, 2.8, 4.7, 1.2],
       [6.4, 2.9, 4.3, 1.3],
       [6.6, 3. , 4.4, 1.4],
       [6.8, 2.8, 4.8, 1.4],
       [6.7, 3. , 5. , 1.7],
       [6. , 2.9, 4.5, 1.5],
       [5.7, 2.6, 3.5, 1. ],
       [5.5, 2.4, 3.8, 1.1],
       [5.5, 2.4, 3.7, 1. ],
       [5.8, 2.7, 3.9, 1.2],
       [6. , 2.7, 5.1, 1.6],
       [5.4, 3. , 4.5, 1.5],
       [6. , 3.4, 4.5, 1.6],
       [6.7, 3.1, 4.7, 1.5],
       [6.3, 2.3, 4.4, 1.3],
       [5.6, 3. , 4.1, 1.3],
       [5.5, 2.5, 4. , 1.3],
       [5.5, 2.6, 4.4, 1.2],
       [6.1, 3. , 4.6, 1.4],
       [5.8, 2.6, 4. , 1.2],
       [5. , 2.3, 3.3, 1. ],
       [5.6, 2.7, 4.2, 1.3],
       [5.7, 3. , 4.2, 1.2],
       [5.7, 2.9, 4.2, 1.3],
       [6.2, 2.9, 4.3, 1.3],
       [5.1, 2.5, 3. , 1.1],
       [5.7, 2.8, 4.1, 1.3],
       [6.3, 3.3, 6. , 2.5],
       [5.8, 2.7, 5.1, 1.9],
       [7.1, 3. , 5.9, 2.1],
       [6.3, 2.9, 5.6, 1.8],
       [6.5, 3. , 5.8, 2.2],
       [7.6, 3. , 6.6, 2.1],
       [4.9, 2.5, 4.5, 1.7],
       [7.3, 2.9, 6.3, 1.8],
       [6.7, 2.5, 5.8, 1.8],
       [7.2, 3.6, 6.1, 2.5],
       [6.5, 3.2, 5.1, 2. ],
       [6.4, 2.7, 5.3, 1.9],
       [6.8, 3. , 5.5, 2.1],
       [5.7, 2.5, 5. , 2. ],
       [5.8, 2.8, 5.1, 2.4],
       [6.4, 3.2, 5.3, 2.3],
       [6.5, 3. , 5.5, 1.8],
       [7.7, 3.8, 6.7, 2.2],
       [7.7, 2.6, 6.9, 2.3],
       [6. , 2.2, 5. , 1.5],
       [6.9, 3.2, 5.7, 2.3],
       [5.6, 2.8, 4.9, 2. ],
       [7.7, 2.8, 6.7, 2. ],
       [6.3, 2.7, 4.9, 1.8],
       [6.7, 3.3, 5.7, 2.1],
       [7.2, 3.2, 6. , 1.8],
       [6.2, 2.8, 4.8, 1.8],
       [6.1, 3. , 4.9, 1.8],
       [6.4, 2.8, 5.6, 2.1],
       [7.2, 3. , 5.8, 1.6],
       [7.4, 2.8, 6.1, 1.9],
       [7.9, 3.8, 6.4, 2. ],
       [6.4, 2.8, 5.6, 2.2],
       [6.3, 2.8, 5.1, 1.5],
       [6.1, 2.6, 5.6, 1.4],
       [7.7, 3. , 6.1, 2.3],
       [6.3, 3.4, 5.6, 2.4],
       [6.4, 3.1, 5.5, 1.8],
       [6. , 3. , 4.8, 1.8],
       [6.9, 3.1, 5.4, 2.1],
       [6.7, 3.1, 5.6, 2.4],
       [6.9, 3.1, 5.1, 2.3],
       [5.8, 2.7, 5.1, 1.9],
       [6.8, 3.2, 5.9, 2.3],
       [6.7, 3.3, 5.7, 2.5],
       [6.7, 3. , 5.2, 2.3],
       [6.3, 2.5, 5. , 1.9],
       [6.5, 3. , 5.2, 2. ],
       [6.2, 3.4, 5.4, 2.3],
       [5.9, 3. , 5.1, 1.8]])
target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

class decision_tree:
    def __init__(self, max_depth, min_num_split):
        self.max_depth = max_depth
        self.min_num_sample = min_num_split

    def gini_score(self, groups, classes):
        n_samples = sum([len(group) for group in groups])
        gini = 0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = (group[:, -1] == class_val).sum() / size
                score += p * p
            gini += (1.0 - score) * (size / n_samples)
        return gini

    def split(self, feat, val, Xy):
        Xi_left = np.array([]).reshape(0, self.Xy.shape[1])
        Xi_right = np.array([]).reshape(0, self.Xy.shape[1])
        for i in Xy:
            if i[feat] <= val:
                Xi_left = np.vstack((Xi_left, i))
            if i[feat] > val:
                Xi_right = np.vstack((Xi_right, i))
        return Xi_left, Xi_right

    def best_split(self, Xy):
        classes = np.unique(Xy[:, -1])
        best_score = 999
        for feat in range(Xy.shape[1] - 1):
            for i in Xy:
                groups = self.split(feat, i[feat], Xy)
                gini = self.gini_score(groups, classes)
                if gini < best_score:
                    best_feat = feat
                    best_val = i[feat]
                    best_score = gini
                    best_groups = groups
        output = {'feat': best_feat, 'val': best_val, 'groups': best_groups}
        return output

    def terminal_node(self, group):
        classes, counts = np.unique(group[:, -1], return_counts=True)
        return classes[np.argmax(counts)]

    def split_branch(self, node, depth):
        left_node, right_node = node['groups']
        del (node['groups'])
        if not isinstance(left_node, np.ndarray) or not isinstance(right_node, np.ndarray):
            node['left'] = node['right'] = self.terminal_node(left_node + right_node)
            return
        if depth >= self.max_depth:
            node['left'] = self.terminal_node(left_node)
            node['right'] = self.terminal_node(right_node)
            return
        if len(left_node) <= self.min_num_sample:
            node['left'] = self.terminal_node(left_node)
        else:
            node['left'] = self.best_split(left_node)
            self.split_branch(node['left'], depth + 1)
        if len(right_node) <= self.min_num_sample:
            node['right'] = self.terminal_node(right_node)
        else:
            node['right'] = self.best_split(right_node)
            self.split_branch(node['right'], depth + 1)

    def build_tree(self):
        self.root = self.best_split(self.Xy)
        self.split_branch(self.root, 1)
        return self.root

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.Xy = np.column_stack((X, y))
        self.build_tree()

    def predict_sample(self, node, sample):
        if sample[node['feat']] < node['val']:
            if isinstance(node['left'], dict):
                return self.predict_sample(node['left'], sample)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_sample(node['right'], sample)
            else:
                return node['right']

    def predict(self, X_test):
        self.y_pred = np.array([])
        for i in X_test:
            self.y_pred = np.append(self.y_pred, self.predict_sample(self.root, i))
        return self.y_pred


dt = decision_tree(max_depth=2, min_num_split=30)
dt.fit(data[:150], target[:150])

predict = dt.predict(data[:150])
print((predict == target).sum())
