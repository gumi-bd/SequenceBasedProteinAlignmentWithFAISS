import numpy as np

# Load the compressed .npy file
data = np.load('score.npy', allow_pickle=True)
print(data.shape)
# Print the contents of the array
print(data)