# regularized-t-learner
Investigating T-learners with regularization

What's a T-learner? The t is for two, meaning a T-learner is actually two learners. 

What's a T-learner learning? The uplift. 


What's the uplift? In it's simplest form, its the difference between two binary classifications under different data regimes. Specifically, 
Uplift = P(y|x,T=1) - p(y|x,T=0)

Usually in binarly classification we are just learning P(y|x), say what are the chances a customer makes purchase y. But for measuring the uplift between two different conditions we want to know how do the chances of the purchase change if a coupon is shown or not. 
We learn two separate binary classifiers with two different datasets (T=0, x, y) and (T=1, x, y). 
