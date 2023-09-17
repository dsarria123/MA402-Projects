import numpy as np
import matplotlib.pyplot as plt

# #Problem 1
# # Initialize arrays to store s and Cn values
# n_values = np.arange(1, 31)
# s = np.zeros_like(n_values, dtype=float)
# Cn = np.zeros_like(n_values, dtype=float)

# # Define the initial value for s1
# s[0] = 1.0

# # Compute sn and Cn for n = 1 to 30
# for i in range(1, 30):
#     s[i] = np.sqrt(2 - np.sqrt(4 - s[i - 1]**2))
#     Cn[i] = 3 * 2**i * s[i]

# # Compute the absolute error
# epsilon = np.abs(Cn - 2 * np.pi)

# # Plot the absolute error on a semi-logarithmic plot
# plt.semilogy(n_values, epsilon)
# plt.xlabel('n')
# plt.ylabel('Absolute Error (log scale)')
# plt.show()

# #Problem 2
# s = 1.0  # Reset s to its initial value
# n_values = np.arange(1, 31)
# epsilon_values_formula2_corrected = np.zeros(30)

# for n in n_values:
#     e_n = 3 * 2**n
#     Cn = e_n * s
#     epsilon = abs(Cn - 2 * np.pi)
    
#     epsilon_values_formula2_corrected[n - 1] = epsilon
    
#     s = s / np.sqrt(2 + np.sqrt(4 - s**2))

# plt.semilogy(n_values, epsilon_values_formula2_corrected)
# plt.xlabel('n')
# plt.ylabel('Absolute Error (εn)')
# plt.title('Absolute Error vs. n (Using Corrected Formula 2)')
# plt.grid(True)
# plt.show()

#Problem 3
import numpy as np
import matplotlib.pyplot as plt

n_values = np.arange(1, 31)  # Values of n from 1 to 30
e_values = 3 * 2 ** n_values  # Values of en
s_values = np.zeros_like(n_values, dtype=float)
K_abs_values = np.zeros_like(n_values, dtype=float)
K_rel_values = np.zeros_like(n_values, dtype=float)

# Initialize s_1 and calculate K_abs and K_rel for n = 1
s_values[0] = 1
K_abs_values[0] = e_values[0]
K_rel_values[0] = e_values[0]

# Calculate K_abs and K_rel for n > 1
for i in range(1, len(n_values)):
    s_values[i] = np.sqrt(2 - np.sqrt(4 - s_values[i - 1] ** 2))
    K_abs_values[i] = e_values[i]
    K_rel_values[i] = e_values[i] / (s_values[i] / s_values[i - 1])

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.semilogy(n_values, K_abs_values, marker='o', label='Absolute Condition Number (K_abs)')
plt.xlabel('n')
plt.ylabel('K_abs')
plt.title('Absolute Condition Number vs. n')
plt.grid()

plt.subplot(2, 1, 2)
plt.semilogy(n_values, K_rel_values, marker='o', label='Relative Condition Number (K_rel)')
plt.xlabel('n')
plt.ylabel('K_rel')
plt.title('Relative Condition Number vs. n')
plt.grid()

plt.tight_layout()
plt.show()
