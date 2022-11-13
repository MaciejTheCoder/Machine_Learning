import pandas as pd
import matplotlib.pyplot as plt
# % matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

iris = pd.read_csv(r"iris.data",
                   header = None, 
                   names = ['petal length', 'petal width', 
                            'sepal length', 'sepal width', 'species'])
                            
x_min, x_max = iris['petal length'].min() - .5, iris['petal length'].max() + .5
y_min, y_max = iris['petal width'].min() - .5, iris['petal width'].max() + .5
 
colors = {'Iris-setosa':'red', 'Iris-versicolor':'blue', 'Iris-virginica':'green'}
 
pd.plotting.scatter_matrix(iris, figsize=(8, 8), 
                           color = iris['species'].apply(lambda x: colors[x]));
plt.show()
