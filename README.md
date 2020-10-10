# easy21
My solution to David Silver's Easy21 sssignment. I wrote this completely from scratch without looking at anything other than the book. Comments and suggestions are more than welcome!

***main.py*** should take care of everything in the assignment.

## Monte Carlo

### Progression of Q by Episode Number

**1e3** | **1e4**
:-:|:-:
![Monte Carlo 1e3](plots/monte_carlo/Q(1e+03).png) | ![Monte Carlo 1e4](plots/monte_carlo/Q(1e+04).png)

**1e5** | **1e6**
:-:|:-:
![Monte Carlo 1e5](plots/monte_carlo/Q(1e+05).png) | ![Monte Carlo 1e6](plots/monte_carlo/Q(1e+06).png)

## Sarsa(λ) - Tabular

## Learning Curves

![Learning Curves](plots/sarsa/tabular/learning_curves.png)

## MSE by λ

* **1E3**  
![MSE 1e3](plots/sarsa/tabular/mse(1e+03).png)

* **2E4**  
![MSE 2e4](plots/sarsa/tabular/mse(2e+04).png)

## Sarsa(λ) - Approximate

## Learning Curves

![Learning Curves](plots/sarsa/approx/learning_curves.png)

## MSE by λ

* **1E3**  
![MSE 1e3](plots/sarsa/approx/mse(1e+03).png)

* **2E4**  
![MSE 2e4](plots/sarsa/approx/mse(2e+04).png)

## Discussion

Warning: This is not investment advice.

* **What are the pros and cons of bootstrapping in Easy21?**  

Bootstrapping has an advantage of prioritizing later events in the session, which emprically performs better the smaller lambda gets. This is because previous card draws do not provide meaningful information for the decision of current action, perhaps only general learning of the probability distribution of this particular deck and table rules. However, since the environment is highly stochastic because every hit may yield one of various cards, future value prediction might never be precise enough.


* **Would you expect bootstrapping to help more in blackjack or Easy21? Why?**  

If the movie is correct, keeping track of finite decks is the best strategy for blackjack and I think bootstrapping will help less because we need to preserve the significance of previous states, cards that have already been drawn. But for that, we might have to expand our episode to deck attrition. I am not good at blackjack.

* **What are the pros and cons of function approximation in Easy21?**

Approximation seems to work better because state transitions are already non-deterministic so we need not compute the exact tabular value of state-actions. It still might be an overkill because state-action pairs in this environment is intrinsically tabular.

* **How would you modify the function approximator suggested in this section to get better results in Easy21?**

I would apply a grid search over the coarse codes to see if better bins might be generated. Tile coding might work better given the prior of incremental states.



