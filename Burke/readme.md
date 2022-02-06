Here is reported the script for building a surrogate model of the empirical temperature to GDP relationship based on Burke et al. 2015.
First the benchmark Burke model is simulated under different temperatures increase.
Then a polynomial is used to compute the increase in damages fraction based on temperature and actual damages fraction.
Relevant functions are defined in utils.py
To replicate:
- run BurkeModel.py
- run SurrogateBurke.py