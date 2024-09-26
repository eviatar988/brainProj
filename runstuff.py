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