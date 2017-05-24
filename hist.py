import matplotlib.pyplot as plt
import numpy as np
from data_utils import load_data
# %matplotlib inline

print('Loading data...')
y,x = load_data('./training/t0/driving_log.csv')
# x = np.random.normal(size = 1000)
plt.hist(x, normed=True, bins=30)
plt.ylabel('Probability')

plt.show()
