Run `python main.py` from `code/`, with `mnist_train_test.mat` in `data/`.
Required packages: NumPy, SciPy, and Matplotlib. The verified full run took
about 61 seconds on this machine (NumPy 2.1.3, SciPy 1.15.3, Matplotlib 3.10.0).

Question 3 saves the gradient-check values in `results/q3_gradient_check.csv`
and the plot in `results/q3_gradient_check.pdf`. The approximately straight
part of the log-log plot has slope 2, as expected from the Taylor remainder.
Question 4 saves the iteration history, including the initial point, in
`results/q4_run_values.csv`; the selected fixed step size is `0.0001`.
Question 5 plots iteration number on a linear horizontal axis and uses
logarithmic vertical axes because both quantities span several orders of
magnitude. The plot is saved in `results/q5_convergence.pdf`.

Question 7 uses the same `theta_final` as the Question 5 convergence plots:
predict 1 when `theta_final @ X > 0`, and 0 otherwise, including a zero score.
The run used step size 0.0001 and stopped after 1546 iterations by the gradient
tolerance criterion.

| Set | Incorrect / samples | Error rate | Percentage |
| --- | --- | --- | --- |
| Training | 2 / 12665 | 0.000157915515 | 0.0157916% |
| Test | 3 / 2115 | 0.001418439716 | 0.1418440% |

`results/q7_classification.npz` stores `theta_final`, `train_error_rate`,
`test_error_rate`, `train_incorrect`, `test_incorrect`, `train_samples`, and
`test_samples`. Load it with `np.load('../results/q7_classification.npz')`
from `code/`. Running `main.py` regenerates it alongside the earlier results.

The test error is higher than the training error by about 0.126 percentage
points, but both are very low. This suggests good generalization on this test
set, with little evidence of substantial overfitting. Only three test mistakes
were observed, so the precise test error should not be overinterpreted.
