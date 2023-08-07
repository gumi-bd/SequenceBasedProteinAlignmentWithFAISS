import numpy as np

# Load the compressed .npy file
data = np.load('score.npy')
print(data.shape)
# Print the contents of the array
print(data)