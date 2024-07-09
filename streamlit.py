import streamlit as st
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

st.title('MLE Estimation of Linear Model Parameters')
st.write('This app estimates the parameters of a linear model using Maximum Likelihood Estimation (MLE).')

# Sidebar for user inputs
st.sidebar.header('Simulation Parameters')
true_a = st.sidebar.slider('True value of a', -2.0, 2.0, 0.5)
true_b = st.sidebar.slider('True value of b', -2.0, 2.0, 0.5)
noise_scale = st.sidebar.slider('Noise scale', 0.1, 2.0, 1.0)

# Generate data
X = np.random.uniform(-3, 3, 500)
r = np.random.normal(0, noise_scale, 500)
r = 2 * r / max(abs(r))
Y = true_a * X + true_b + r

# Define the log-likelihood function
def log_likelihood(a, b):
    predicted_Y = a * X + b
    errors = Y - predicted_Y
    log_likelihood = -0.5 * np.sum(errors ** 2) - 0.5 * len(X) * np.log(2 * np.pi)
    return log_likelihood

if st.button('Run MLE Estimation'):
    with st.spinner('Running MLE estimation...'):
        # Use optimization to find MLE estimates for 'a' and 'b'
        result = minimize(lambda params: -log_likelihood(params[0], params[1]), [0, 0])
        mle_a, mle_b = result.x

    st.success('MLE estimation completed!')
    
    st.write('### Results')
    st.write(f"**MLE Estimate for 'a':** {mle_a:.4f}")
    st.write(f"**True a :** {true_a:.4f}")
    st.write(f"**MLE Estimate for 'b':** {mle_b:.4f}")
    st.write(f"**True b :** {true_b:.4f}")

    # Displaying the generated data and fit
    fig, ax = plt.subplots()
    ax.scatter(X, Y, label='Data', alpha=0.6)
    ax.plot(X, mle_a * X + mle_b, color='red', label='Fitted Line', linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    st.pyplot(fig)

    # Show the errors histogram
    st.write('### Error Distribution')
    predicted_Y = mle_a * X + mle_b
    errors = Y - predicted_Y
    fig, ax = plt.subplots()
    ax.hist(errors, bins=30, alpha=0.7)
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.write("### Data Description")
    st.write("The data is generated using the equation $Y = aX + b + r$, where $r$ is a normally distributed random noise.")
    st.write(f"Number of data points: {len(X)}")
    st.write(f"Noise scale: {noise_scale}")

# Information section
st.sidebar.header('About')
st.sidebar.info('This app demonstrates the Maximum Likelihood Estimation (MLE) technique for estimating the parameters of a simple linear regression model.')
