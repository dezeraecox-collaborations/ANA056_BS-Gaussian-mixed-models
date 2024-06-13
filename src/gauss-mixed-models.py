import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.mixture import GaussianMixture

# Simulate dataset with three peaks
peak_1 = np.random.normal(0.2, 0.05, 5000)
peak_2 = np.random.normal(0.5, 0.03, 1000)
peak_3 = np.random.normal(0.9, 0.01, 500)

peaks = pd.concat([pd.DataFrame(peak_1), pd.DataFrame(peak_2), pd.DataFrame(peak_3)])

# Generate histogram to visualise
bins = np.linspace(0, 1, 200)
counts = pd.DataFrame(pd.cut(peaks[0].values, bins=bins, right=True, labels=bins[:-1]).value_counts()).reset_index().sort_index()
sns.lineplot(data=counts, x='index', y=0)
plt.show()

# Generate GMM and fit to values
gauss_model = GaussianMixture(n_components=3, covariance_type='full')
gauss_model.fit(peaks[0].values.reshape(-1, 1))

# Extract fitted parameters
means = gauss_model.means_
weights = gauss_model.weights_
covars = gauss_model.covariances_

# For each model, use fitted parameters to generate sample gaussian line data for plotting fit
fit_vals = []
for model in range(len(means)):
    vals = pd.DataFrame([np.linspace(0, 1, 1000), weights[model]*stats.norm.pdf(np.linspace(0, 1, 1000), means[model], np.sqrt(covars[model])).ravel()]).T
    vals['model'] = model
    # Optional normalisation to maxiximum height of the sigma
    vals['norm_val'] = vals[1] / vals[1].max()
    fit_vals.append(vals)
fit_vals = pd.concat(fit_vals)
fit_vals.columns = ['x_fit', 'y_fit', 'model', 'norm_val']
fit_vals.reset_index(drop=True, inplace=True)

# Visualise fit, overlayed on the original histogram
# Note, palette is specific for number of components
palette={0: '#FB8B24', 1: '#820263', 2: '#234E69', 3: '#1AB0B0', 4: '#D90368', 5: '#EA4746'}
sns.lineplot(data=fit_vals, x='x_fit', y='y_fit', hue='model', palette=palette, linewidth=2)
sns.distplot(peaks, kde=False, norm_hist=True, color='black')
plt.show()


