sconsideration, time complexty, memory, importance, challenge , Feasible, which tech, why



#introduction

along with the  network information rapid growth, the rich information caused the problem of information overload, the manifestation of it is people are hard to make decision on various items, and  an item will have less prosibility meet by target user. in this situation, a decision support system for recommend approprate items to user are important. in application, this problem can  transfer into a problem of predict the rating of the user would give to each item.

Netflix price dataset is the dataset provided by Netflix company for movie recomendation, 



# Exploration

Netflix price dataset contains seven files include the ratings of users give to each movie with its date and the movie information include their publish date and name. 





 $\left[ \begin{matrix} 0 & 3 & 0 & 0 & 0 & \cdots & 0\\ 0 & 0 & 5 &0 &0& \cdots & 0 \\ \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 &0 &2& \cdots & 0 \end{matrix} \right]$











# Methodology

the formal recommendation system have two main ways, one is content based system that use the   tags of items and historical preference of user to recommend the item have same tags as user browsed before.

but for solve the problem of information overload the collaborative filter based systems are more approprated, they share the information between user or items to make recommendation by similar instence.

i choose to use the matrix fatorized models for 

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





