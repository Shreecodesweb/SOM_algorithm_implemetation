# Import necessary libraries
import numpy as np
import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

# Step 1: Load the Iris dataset
iris = load_iris()
data = iris.data  # Features
target = iris.target  # Labels (Not used in training, only for plotting)

# Step 2: Normalize the data
scaler = MinMaxScaler()  # Scale data between 0 and 1
data_normalized = scaler.fit_transform(data)

# Step 3: Initialize the SOM
# Create a 10x10 grid, input_len corresponds to the number of features (4 in the Iris dataset)
som = MiniSom(x=10, y=10, input_len=data_normalized.shape[1], sigma=1.0, learning_rate=0.5)

# Step 4: Initialize weights randomly and train the SOM
som.random_weights_init(data_normalized)
som.train_random(data_normalized, num_iteration=100)  # Training for 100 iterations

# Step 5: Visualize the results
plt.figure(figsize=(7, 7))

# Plot the distance map (U-Matrix), which shows the distance between neighboring neurons
plt.pcolor(som.distance_map().T, cmap='coolwarm')
plt.colorbar()

# Step 6: Plot each data point (with different markers and colors for each class)
markers = ['o', 's', 'D']  # Different shapes for each class
colors = ['r', 'g', 'b']  # Different colors for each class

for i, x in enumerate(data_normalized):
    # Get the winning node for each data sample
    winning_node = som.winner(x)
    
    # Plot the sample at the winning node with the corresponding marker and color
    plt.plot(winning_node[0] + 0.5, winning_node[1] + 0.5, markers[target[i]],
             markerfacecolor='None', markeredgecolor=colors[target[i]], 
             markersize=12, markeredgewidth=2)

plt.title('Self-Organizing Map for Iris Dataset')
plt.show()
