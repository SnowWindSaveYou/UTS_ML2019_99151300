sconsideration, time complexty, memory, importance, challenge , Feasible, which tech, why



#introduction

along with the  network information rapid growth, the rich information caused the problem of information overload, the manifestation of it is people are hard to make decision on various items, and  an item will have less prosibility meet by target user. in this situation, a decision support system for recommend approprate items to user are important. in application, this problem can  transfer into a problem of predict the rating of the user would give to each item.

Netflix price dataset is the dataset provided by Netflix company for movie recomendation, 



# Exploration

Netflix price dataset contains seven files include the ratings of users give to each movie with its date and the movie information include their publish date and name. 





 $\left[ \begin{matrix} 0 & 3 & 0 & 0 & 0 & \cdots & 0\\ 0 & 0 & 5 &0 &0& \cdots & 0 \\ \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 &0 &2& \cdots & 0 \end{matrix} \right]$



Since the low activate users are useless for training model, I filtered the user have rated movies lower than 50. And for evaluate the models I will explain in next section, I decided to split the data set into three sets.

- Train: include most data except the data in test set.
- Test: randomly choose 1/10 users, and split their data by time '2005/12/1', so that dataset have include 1/10 users without data after '2005/12/1'.
- Target: target is the full data of test data, it include full data of 1/10 users chosen by test set

This split way allows me can evaluate the accuracy of model predict the rating of users would give to the items they were not seen before.





# Methodology

the formal recommendation system have two main ways, one is content based system that use the   tags of items and historical preference of user to recommend the item have same tags as user browsed before, but those kind of system have the problem of over-specialization that can only find known preference while can't find potential preference, and easy to make other problem of information cocoon. (Adomavicius & Tuzhilin 2005) So, for solve the problem of information overload I think that collaborative filter based systems are more appropriated, those kind of methods make recommendation by sharing information between user or items, main idea of them are make recommendation by find other user have same preference. 

in this case, I choose to use the matrix factorization models for 

### Matrix fatorization

 



![](/Users/wujingyi/UTS_document/AdvDataAnalytis/UTS_ML2019_99151300/img/fm_principle.png)

Basic

Multi-Multi-Layer Perception based



### Autoencoder 

Restricted Boltzmann machine

# Evaluation

to evaluate the models,

Experiment Design

Accuracy Metrics

MSE

RMSE

光丿羽  16:49:45
$\operatorname { RMSE } = \frac { \sqrt { \sum _ { ( u , i ) \in T } \left( r _ { u i } - \hat { r } _ { u i } \right) ^ { 2 } } } { | \text { Test } | }$





Coverage Metric





# Ethic issue

spread bad information

privacy





