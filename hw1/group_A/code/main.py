"""
About tool use: retain control, align with your goals. We highly recommend 
that you disable copilot and other such intrusive autocomplete technologies.
In contrast, it is fine to use LLMs to help you format plots, read and write
files, learn how to code better, and other such aspects which won’t adversely
affect your opportunity to learn by doing.

About your code: We will unzip your submission, add the data files in data/,
install any required packages (for Python), and then run your code from
group_ID/code/ (e.g., K/code/) with this command: python main.py

Keep code/ as the working directory: read data from ../data/ and write outputs
to ../results/, relative to that folder. This should work as is. We will not
debug your submission. The complete run should take no more than five minutes
on a standard laptop.

Your program should run to completion without requiring any user input, and stop
on its own. Your code should (re)create the requested files in ../results/ from
the supplied data rather than loading previously saved results when we run the
command above.

For full credit, your code should:
- Be vectorized (essentially, this means no loops over array indices),
except where requested;
- Avoid redundant computations when this affects overall running time;
- Not crash due to numerical issues such as NaN / Inf.
"""

from scipy.io import loadmat
import time
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Data

data = loadmat('../data/mnist_train_test.mat',
               squeeze_me=True, struct_as_record=False)
train, test = data['train'], data['test']

train_X, train_Y  = np.asarray(train.X), np.asarray(train.y, dtype=np.float64)
test_X, test_Y = np.asarray(test.X), np.asarray(test.y, dtype=np.float64)

d, m = train_X.shape # 785, 12665

l2_reg = 5e-3 # regularization

# Question 2

def phi_scalar(z): # z real
    if z <= -1.0:
        return 0.0
    elif -1.0 < z <= 0.0:
        return 0.5 * (1.0+z)**2
    else:
        return 0.5 + z
phi = np.vectorize(phi_scalar)

def dphi_scalar(z): # z real
    if z <= -1.0:
        return 0.0
    elif -1.0 < z <= 0.0:
        return 1.0+z
    else:
        return 1.0
dphi = np.vectorize(dphi_scalar)

def f_explicit(theta, data_X = train_X, data_Y = train_Y, l2_reg=l2_reg):
    loss = 0.0
    m = data_X.shape[1]

    for i in range(m):
        x_i = data_X[:, i]        #(785,)
        y_i = data_Y[i]           # Label (0 o 1)
        s_i = 1.0 - 2.0 * y_i     # Scale y_i from {0,1} to {1, -1}
        
        reg_i = s_i * np.dot(theta, x_i)   # s_i * <θ, x_i>
        loss += phi_scalar(reg_i)        
        
    reg = (l2_reg / 2.0) * np.dot(theta, theta)
    return reg + loss

def grad_f_explicit(theta, data_X=train_X, data_Y=train_Y, l2_reg=l2_reg):
    grad = np.zeros_like(theta)
    m = data_X.shape[1]
    
    for i in range(m):
        x_i = data_X[:, i]        #(785,)
        y_i = data_Y[i]           # Label (0 o 1)
        s_i = 1.0 - 2.0 * y_i     # Scale y_i from {0,1} to {1, -1}
        
        reg_i = s_i * np.dot(theta, x_i)   # s_i * <θ, x_i>
        grad += (s_i * dphi_scalar(reg_i)) * x_i
        
    grad += l2_reg * theta
    return grad
  
def f(theta, data_X = train_X, data_Y = train_Y, l2_reg = l2_reg):
    return (
        l2_reg/2 * np.sum(theta**2) 
        + np.sum(phi((1 - 2 * data_Y) * (theta @ data_X)))
    )

def grad_f(theta, data_X = train_X, data_Y = train_Y, l2_reg = l2_reg):
    s = 1 - 2 * data_Y
    return (
        l2_reg * theta
        + (s * data_X) @ dphi(s * (theta @ data_X))
    )
######
###if you want you can leave these tests but i did them below..
#####
X_test = np.array([
    [2.0, 0.5, 1.0],
    [1.0, 1.0, 1.0]
])

y_test = np.array([1.0, 1.0, 0.0])
theta_test = np.array([1.0, 0.0])
lam_test = 2.0

assert np.isclose(f(theta_test, X_test, y_test, lam_test), 2.625)
assert np.allclose(
    grad_f(theta_test, X_test, y_test, lam_test),
    np.array([2.75, 0.5])
)

#tests
theta_test = np.random.randn(d)

#check f
val_vec = f(theta_test)
val_exp = f_explicit(theta_test)

assert np.isclose(val_vec, val_exp)

#check grad_f
grad_vec = grad_f(theta_test)
grad_exp = grad_f_explicit(theta_test)

assert np.allclose(grad_vec, grad_exp)

#time for explicit version
t0 = time.time()
f_exp_val = f_explicit(theta_test)
grad_exp_val = grad_f_explicit(theta_test)
t_explicit = time.time() - t0

#time for vector form
t0 = time.time()
f_vec_val = f(theta_test)
grad_vec_val = grad_f(theta_test)
t_vectorized = time.time() - t0

# Results
print(f"explicit time:    {t_explicit:.4f} seconds")
print(f"vectorized time:  {t_vectorized:.4f} seconds")

# Question 3

theta_q3 = np.random.randn(d)

v = np.random.randn(d)
v = v / np.linalg.norm(v)

t_values = np.logspace(-8, 0, 101)

f_theta = f(theta_q3)
grad_theta = grad_f(theta_q3)

errors = np.array([
    abs(
        f(theta_q3 + t * v)
        - f_theta
        - t * (v @ grad_theta)
    )
    for t in t_values
])

os.makedirs('../results', exist_ok=True)

# Save numerical values
with open('../results/q3_gradient_check.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["t", "error"])

    for t, error in zip(t_values, errors):
        writer.writerow([t, error])

# Plot
fig = plt.figure(figsize=(6, 4))

plt.loglog(t_values, errors, label='Taylor error')

# Reference line with slope 2
reference = errors[-1] * (t_values / t_values[-1])**2
plt.loglog(t_values, reference, '--', label=r'$O(t^2)$')

plt.xlabel(r'$t$')
plt.ylabel(
    r'$|f(\theta+tv)-f(\theta)-t\,v^T\nabla f(\theta)|$'
)
plt.title('Gradient check')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()

plt.savefig('../results/q3_gradient_check.pdf')
plt.close(fig)

# Question 4

GRAD_TOLERANCE = 1e-3

def run_gradient_descent(theta_0, alpha = 1e-4, time_limit = 3.0 * 60):
    theta = theta_0
    grad = grad_f(theta)

    values_f = [f(theta)]
    values_grad_f_norm = [np.linalg.norm(grad)]

    start = time.perf_counter() #s
    while (
        values_grad_f_norm[-1] > GRAD_TOLERANCE * values_grad_f_norm[0]
        and time.perf_counter() - start < time_limit # 3 mins
    ):
        theta = theta - alpha * grad
        grad = grad_f(theta)

        values_f.append(f(theta))
        values_grad_f_norm.append(np.linalg.norm(grad))

    if values_grad_f_norm[-1] <= GRAD_TOLERANCE * values_grad_f_norm[0]:
        stopping_reason = "gradient_tolerance"
    else:
        stopping_reason = "time_limit"

    return theta, values_f, values_grad_f_norm, stopping_reason

theta_0 = np.random.randn(d)
"""
print("running grid_search:")
alpha_grid = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

grid_results = {}
for alpha in alpha_grid:
    _, values_f_a, values_grad_norm_a, stopping_reason_a = run_gradient_descent(
        theta_0, alpha = alpha, time_limit = 3 * 60.0
    )
    grid_results[alpha] = {
        "min_f": min(values_f_a),
        "n_iter": len(values_f_a) - 1,
        "stopping_reason": stopping_reason_a,
    }
"""
best_alpha = 1e-4
#min(grid_results, key=lambda a: grid_results[a]["min_f"])
theta_final, values_f, values_grad_norm, stopping_reason = run_gradient_descent(
    theta_0, best_alpha
)

# Grid search results

"""
print("Grid search over alpha (time budget "
      f"{15*60.0:.0f}s each):")
for alpha, res in grid_results.items():
    print(f"  alpha={alpha:g}: min f={res['min_f']:.6f}, "
          f"iterations={res['n_iter']}, stopping={res['stopping_reason']}")
"""
print(f"Selected alpha = {best_alpha:g}")
print(f"Final run ({3*60.0:.0f}s budget): "
      f"f={values_f[-1]:.6f}, grad_norm={values_grad_norm[-1]:.6e}, "
      f"iterations={len(values_f) - 1}, stopping={stopping_reason}")

# Save file

os.makedirs('../results', exist_ok=True)

with open('../results/q4_run_values.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["iteration", "alpha", "f", "grad_norm", "stopping_reason"])
    for k in range(len(values_f)):
        writer.writerow(
            [k, best_alpha, values_f[k], values_grad_norm[k], stopping_reason]
        )
"""
with open('../results/q4_grid_search.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["alpha", "min_f", "n_iter", "stopping_reason"])
    for alpha, res in grid_results.items():
        writer.writerow(
            [alpha, res["min_f"], res["n_iter"], res["stopping_reason"]]
        )
"""
## plots

iterations = np.arange(len(values_f))
 
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
 
axes[0].plot(iterations, values_f)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel(r'$f(\theta_k)$')
axes[0].set_title('Objective value')

axes[1].plot(iterations, values_grad_norm)
axes[1].set_yscale('log')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel(r'$\|\nabla f(\theta_k)\|$')
axes[1].set_title('Gradient norm')
 
fig.suptitle(f'Gradient descent convergence (alpha = {best_alpha:g})')
fig.tight_layout()
fig.savefig('../results/q5_convergence.pdf')
plt.close(fig)
