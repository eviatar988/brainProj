"""upper_triangle_rest = matrix[np.triu_indices(matrix.shape[0], k=1)]
data = upper_triangle_film

# Compute kernel density estimate
kde = gaussian_kde(data)

# Create a range of values to evaluate the PDF
x_vals = np.linspace(min(data), max(data), 1000)

# Evaluate the PDF at the specified values
pdf_values = kde(x_vals)

pdf_percentage_film = pdf_values / np.sum(pdf_values) * 100

data = upper_triangle_rest

# Compute kernel density estimate
kde = gaussian_kde(data)

# Create a range of values to evaluate the PDF
x_vals = np.linspace(min(data), max(data), 1000)

# Evaluate the PDF at the specified values
pdf_values = kde(x_vals)

pdf_percentage_rest = pdf_values / np.sum(pdf_values) * 100
# Plot the estimated PDF
plt.plot(x_vals, pdf_percentage_film - pdf_percentage_rest, label='Estimated PDF')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Estimated Probability Density Function')

# Show plot
plt.legend()
plt.grid(True)
plt.show()"""