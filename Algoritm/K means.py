import numpy as np
import matplotlib.pyplot as plt



class K_means:

    def __init__(self, values, n_classter):
        self.x = values
        self.classter = self.clasters(values, n_classter)


    @staticmethod
    def clasters(x, n_class):

        max_ = x.max(axis=0)
        min_ = x.min(axis=0)

        np.random.seed(42)
        centers = np.zeros((n_class, 2))
        for i, k in enumerate(centers):
            centers[i] = np.random.randint(min_[0], max_[0]), np.random.randint(min_[1], max_[1])


        return centers

    @staticmethod
    def manhattan(x, y):
        total = 0
        for j in range(y.shape[0]):
            total += np.abs((x[j] - y[j]))
        return total


    def refresh_classter(self, x):
        c = np.zeros((len(self.classter), 2))
        for i, k in enumerate(x):
            m = np.mean(k, axis=0)
            c[i][0] = m[0]
            c[i][1] = m[1]

        return c

    def distance(self):

        new = []
        for i in range(len(self.classter)):
            new.append(np.empty(0))


        for k in range(len(self.x)):
            distance = np.empty(0)
            for i in self.classter:
                distance = np.append(distance, self.manhattan(self.x[k], i))
            min_ = np.argmin(distance)
            new[min_] = np.append(new[min_], self.x[k])

        for i in range(len(new)):
            new[i] = new[i].reshape(int(len(new[i]) / 2), 2)

        return new


    def pred(self):
        for i in range(20):
            x = self.distance()
            self.classter = self.refresh_classter(x)

        return x, self.classter


    def graph(self):
        x, c = self.pred()
        for i in range(len(c)):
            plt.scatter(c[i][0], c[i][1], marker='x', s=150, c='black')
        for i in x:
            plt.scatter(i[:, 0], i[:, 1], marker='o', s=50)
        plt.show()





df = np.array([[5.1, 1.4],
       [4.9, 1.4],
       [4.7, 1.3],
       [4.6, 1.5],
       [5. , 1.4],
       [5.4, 1.7],
       [4.6, 1.4],
       [5. , 1.5],
       [4.4, 1.4],
       [4.9, 1.5],
       [5.4, 1.5],
       [4.8, 1.6],
       [4.8, 1.4],
       [4.3, 1.1],
       [5.8, 1.2],
       [5.7, 1.5],
       [5.4, 1.3],
       [5.1, 1.4],
       [5.7, 1.7],
       [5.1, 1.5],
       [5.4, 1.7],
       [5.1, 1.5],
       [4.6, 1. ],
       [5.1, 1.7],
       [4.8, 1.9],
       [5. , 1.6],
       [5. , 1.6],
       [5.2, 1.5],
       [5.2, 1.4],
       [4.7, 1.6],
       [4.8, 1.6],
       [5.4, 1.5],
       [5.2, 1.5],
       [5.5, 1.4],
       [4.9, 1.5],
       [5. , 1.2],
       [5.5, 1.3],
       [4.9, 1.5],
       [4.4, 1.3],
       [5.1, 1.5],
       [5. , 1.3],
       [4.5, 1.3],
       [4.4, 1.3],
       [5. , 1.6],
       [5.1, 1.9],
       [4.8, 1.4],
       [5.1, 1.6],
       [4.6, 1.4],
       [5.3, 1.5],
       [5. , 1.4],
       [7. , 4.7],
       [6.4, 4.5],
       [6.9, 4.9],
       [5.5, 4. ],
       [6.5, 4.6],
       [5.7, 4.5],
       [6.3, 4.7],
       [4.9, 3.3],
       [6.6, 4.6],
       [5.2, 3.9],
       [5. , 3.5],
       [5.9, 4.2],
       [6. , 4. ],
       [6.1, 4.7],
       [5.6, 3.6],
       [6.7, 4.4],
       [5.6, 4.5],
       [5.8, 4.1],
       [6.2, 4.5],
       [5.6, 3.9],
       [5.9, 4.8],
       [6.1, 4. ],
       [6.3, 4.9],
       [6.1, 4.7],
       [6.4, 4.3],
       [6.6, 4.4],
       [6.8, 4.8],
       [6.7, 5. ],
       [6. , 4.5],
       [5.7, 3.5],
       [5.5, 3.8],
       [5.5, 3.7],
       [5.8, 3.9],
       [6. , 5.1],
       [5.4, 4.5],
       [6. , 4.5],
       [6.7, 4.7],
       [6.3, 4.4],
       [5.6, 4.1],
       [5.5, 4. ],
       [5.5, 4.4],
       [6.1, 4.6],
       [5.8, 4. ],
       [5. , 3.3],
       [5.6, 4.2],
       [5.7, 4.2],
       [5.7, 4.2],
       [6.2, 4.3],
       [5.1, 3. ],
       [5.7, 4.1],
       [6.3, 6. ],
       [5.8, 5.1],
       [7.1, 5.9],
       [6.3, 5.6],
       [6.5, 5.8],
       [7.6, 6.6],
       [4.9, 4.5],
       [7.3, 6.3],
       [6.7, 5.8],
       [7.2, 6.1],
       [6.5, 5.1],
       [6.4, 5.3],
       [6.8, 5.5],
       [5.7, 5. ],
       [5.8, 5.1],
       [6.4, 5.3],
       [6.5, 5.5],
       [7.7, 6.7],
       [7.7, 6.9],
       [6. , 5. ],
       [6.9, 5.7],
       [5.6, 4.9],
       [7.7, 6.7],
       [6.3, 4.9],
       [6.7, 5.7],
       [7.2, 6. ],
       [6.2, 4.8],
       [6.1, 4.9],
       [6.4, 5.6],
       [7.2, 5.8],
       [7.4, 6.1],
       [7.9, 6.4],
       [6.4, 5.6],
       [6.3, 5.1],
       [6.1, 5.6],
       [7.7, 6.1],
       [6.3, 5.6],
       [6.4, 5.5],
       [6. , 4.8],
       [6.9, 5.4],
       [6.7, 5.6],
       [6.9, 5.1],
       [5.8, 5.1],
       [6.8, 5.9],
       [6.7, 5.7],
       [6.7, 5.2],
       [6.3, 5. ],
       [6.5, 5.2],
       [6.2, 5.4],
       [5.9, 5.1]])


km = K_means(df, n_classter=3)
print(km.classter)
km.graph()



