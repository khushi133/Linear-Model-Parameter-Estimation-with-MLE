import numpy as np
# Define the true coefficients a and b
true_a = np.random.uniform(-2, 2)
true_b = np.random.uniform(-2, 2)
# Generate X values in the range [-3, 3]
X = np.random.uniform(-3, 3, 500)

# Generate normally distributed random number in the range [-2, 2]
r = np.random.normal(0, 1, 500)
r = 2*r/max(abs(r))

# Generate Y values using Y = aX + b + r
Y = true_a * X + true_b + r


# Define the log-likelihood function for error
def log_likelihood(a, b):
    # Calculate the errors
    predicted_Y = a * X + b
    errors = Y - predicted_Y
    # Calculate the log-likelihood (assuming standard normally distributed errors)
    log_likelihood = -0.5 * np.sum(errors ** 2) - 0.5 * len(X) * np.log(2 * np.pi)
    return log_likelihood

# Use optimization to find MLE estimates for 'a' and 'b'
from scipy.optimize import minimize
result = minimize(lambda params: -log_likelihood(params[0], params[1]), [0, 0])
mle_a, mle_b = result.x

print("MLE Estimate for 'a':", mle_a)
print("True a :", true_a)
print("MLE Estimate for 'b':", mle_b)
print("True b :", true_b)