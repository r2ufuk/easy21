# easy21
My solution to David Silver's Easy21 sssignment. I wrote this completely from scratch without looking at anything other than the book. Comments and suggestions are more than welcome!

***main.py*** should take care of everything in the assignment.

## Monte Carlo

### Progression of Q by Episode Number

* ####1E4
![Monte Carlo 1e4](plots/monte_carlo/Q(1e+04).png)

* ####1E5
![Monte Carlo 1e5](plots/monte_carlo/Q(1e+05).png)

* ####1E6
![Monte Carlo 1e6](plots/monte_carlo/Q(1e+06).png)

## Sarsa(λ) - Tabular

## Learning Curves

![Learning Curves](plots/sarsa/tabular/learning_curves.png)

## MSE by λ

* ###1E3
![MSE 1e3](plots/sarsa/tabular/mse(1e+03).png)

* ###2E4
![MSE 2e4](plots/sarsa/tabular/mse(2e+04).png)

## Sarsa(λ) - Approximate

## Learning Curves

![Learning Curves](plots/sarsa/approx/learning_curves.png)

## MSE by λ

* ###1E3
![MSE 1e3](plots/sarsa/approx/mse(1e+03).png)

* ###2E4
![MSE 2e4](plots/sarsa/approx/mse(2e+04).png)