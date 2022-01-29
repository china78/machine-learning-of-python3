import numpy as np
import matplotlib.pyplot as plt

x_plot = np.linspace(-1, 6, 141)
y_plot = (x_plot - 2.5) ** 2 - 1

# plt.plot(x_plot, y_plot)
# plt.show()


def J(theta):
    try:
        return (theta - 2.5) ** 2 - 1
    except:
        return float('inf')


def DJ(theta):
    return 2*(theta - 2.5)


# theta = 0.0
# theta_history = [theta]
# eta = 0.01
# precision = 1e-8
# # i_iters = 0
# n_iters = 1e4


def gradient_descent(initial_theta, eta, n_iters=1e4, precision=1e-8):
    theta = initial_theta
    i_iters = 0
    theta_history = [initial_theta]

    while i_iters < n_iters:
        gredient = DJ(theta)
        last_theta = theta
        theta = theta - eta * gredient
        theta_history.append(theta)
        if (abs(J(last_theta) - J(theta)) < precision):
            break
        i_iters += 1
    return theta, theta_history


def plot_theta_history(x_plot, theta_history):
    plt.plot(x_plot, J(x_plot))
    plt.plot(np.array(theta_history), J(
        np.array(theta_history)), color='r', marker='+')
    plt.show()


theta, theta_history = gradient_descent(initial_theta=0.0, eta=0.01,
                                        n_iters=1e4, precision=1e-8)
print('theta: ', theta)
print('theta_history: ', len(theta_history))

plot_theta_history(x_plot, theta_history)
