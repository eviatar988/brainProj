import numpy as np

# Example matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# Number of random columns to select
num_random_columns = 2

# Randomly select column indices
random_column_indices = np.random.choice(matrix.shape[1], num_random_columns, replace=False)

# Extract random columns
random_columns = matrix[:, random_column_indices]
print("Random Columns:\n", random_columns)

"""
pvalues = []
for freq in freq_dict.keys():
    rest_avg = []
    film_avg = []

    for i, patient in enumerate(patients):
        rest_data, film_data = data_extracts.data_extract(freq, freq, (i, i), data_extracts.data_fit)
        rest_avg.append(np.mean(rest_data))
        film_avg.append(np.mean(film_data))
    stat, pvalue = stats.ttest_rel(rest_avg, film_avg)
    pvalues.append(pvalue)
print(pvalues)
plt.bar(freq_dict.keys(), pvalues)
plt.title('t-test,pvalues for mean values of all the patients')
plt.axhline(y=0.05, color='red', linestyle='--')
plt.show()
pvalues = []
for freq in freq_dict.keys():
    rest_avg = []
    film_avg = []
    for i, patient in enumerate(patients):
        rest_data, film_data = data_extracts.data_extract(freq, freq, (i, i), data_extracts.data_fit)
        rest_avg.append(np.mean(rest_data))
        film_avg.append(np.mean(film_data))
    stat, pvalue = stats.f_oneway(rest_avg, film_avg)
    pvalues.append(pvalue)
print(pvalues)
plt.bar(freq_dict.keys(), pvalues)
plt.title('f-test,pvalues for mean values of all the patients')
plt.axhline(y=0.05, color='red', linestyle='--')
plt.show()

p_values_all = []
for freq in freq_dict:
    p_values = []
    for i, patient in enumerate(patients):
        rest_data, film_data = data_extracts.data_extract(freq, freq, (i, i), data_extracts.data_fit)
        rest_data = np.mean(rest_data, axis=0)
        film_data = np.mean(film_data, axis=0)
        stat, pvalue = stats.ttest_rel(rest_data, film_data)
        p_values.append(pvalue)
    p_values = np.array(p_values)
    p_values_all.append(p_values)
bars = []
labels = []
keys = list(freq_dict.keys())
for i, p_values in enumerate(p_values_all):
    bars.append(np.sum(p_values < 0.05))
    bars.append(np.sum(p_values >= 0.05))
    labels.append(f'{keys[i]}:success')
    labels.append(f'{keys[i]}:failure')
plt.figure(figsize=(15, 6))
plt.ylabel('patients')
plt.bar(labels, bars)
plt.title(f'paired t-test success rate')
plt.show()

svm_results = test1.pred_all_freqs(ml_algorithms.svm_classifier, data_extracts.max_indices)
sns.boxplot(data=svm_results)
plt.title(f'svm classifier')
plt.show()
rf_results = test1.pred_all_freqs(ml_algorithms.random_forest, data_extracts.max_indices)
sns.boxplot(data=rf_results)
plt.title(f'random forest')
plt.show()
svm_results = test1.pred_all_freqs(ml_algorithms.svm_classifier, data_extracts.max_diffrence_indices)
print(svm_results)
sns.boxplot(data=svm_results)
plt.title(f'svm classifier')
plt.show()
rf_results = test1.pred_all_freqs(ml_algorithms.random_forest, data_extracts.max_diffrence_indices)
sns.boxplot(data=rf_results)
plt.title(f'random forest')
plt.show()


acc_al = []
for freq in freq_dict.keys():
    acc = test1.pred_single_frequency(ml_algorithms.random_forest, data_extracts.max_indices, freq)
    print(np.mean(acc))
    acc_al.append(acc)

plt.figure(figsize=(10, 6))  # Optional: Adjust figure size

plt.boxplot(acc_al, positions=[1, 2, 3, 4, 5, 6])  # Positions for the groups

# Optional: Add labels to x-axis
plt.xticks([1, 2, 3, 4, 5, 6], list(freq_dict.keys()))
plt.xlabel('Frequency ranges')
plt.ylabel('Accuracy')
plt.title('Boxplot Of Accuracies for single patient case,Random Forest')
plt.grid(True)

plt.show()

acc = test1.pred_all_frequencys(ml_algorithms.svm_classifier, data_extracts.max_diffrence_indices)
print(np.mean(acc))
sns.boxplot(acc)
plt.show()
"""