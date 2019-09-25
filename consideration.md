sconsideration, time complexty, memory, importance, challenge , Feasible, which tech, why



# introduction

Along with the  network information rapid growth, the rich information caused the problem of information overload, the manifestation of it is people are hard to make decision on various items, and  an item will have less possibility meet by target user. An example of this consequence can be a good movie only have limited box office income because of no one heard about it. in this situation, a decision support system for recommend appropriate items to user are important, because of it can provide the chance of the item can contact  with the target user over huge information sea, that active the economic flows over internet and finally can benefits to the platform maintaining recommendation system.

In application, this problem can  transfer into the problem of predict the rating or ranking of the user would give to each item. because of the reliable application prospects and business values, there are various methods have designed for make recommendation. The major recommendation systems have two main ways, one is content based system that use the tags of items and historical preference of user to recommend the item have same tags as user browsed before, but those kind of system have the problem of over-specialization that can only find known preference while can't find potential preference, and easy to make other problem of information cocoon. (Adomavicius & Tuzhilin 2005) So, for solve the problem of information overload I think that collaborative filter based systems are more appropriated, those kind of methods make recommendation by sharing information between user or items, main idea of them are make recommendation by find other users have same preference. in this case, the machine learning technologies are also widely used in recommendation system, they are usually used for extract the features or modelling auxiliary information, but recent research also analysis how to use them to interact user and item features. (He et al. 2017)

Netflix price dataset is the dataset provided by Netflix company for movie recommendation. in this report I will analysis how to build the models over Netflix dataset, the objective is to learn the good recommendation by predict the rating of user would give to movies through collaborative filtering methods those sharing user and item features.



 I identifier movie recommendation problem as a rating prediction problem, 

in this section I had







# Exploration

Netflix price dataset contains seven files include the ratings of users give to each movie with its date and the movie information include their publish date and name. 





 $\left[ \begin{matrix} 0 & 3 & 0 & 0 & 0 & \cdots & 0\\ 0 & 0 & 5 &0 &0& \cdots & 0 \\ \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 &0 &2& \cdots & 0 \end{matrix} \right]$



Since the low activate users are useless for training model, I filtered the user have rated movies lower than 50. And for evaluate the models I will explain in next section, I decided to split the data set into three sets.

- Train: include most data except the data in test set.
- Test: randomly choose 1/10 users, and split their data by time '2005/12/1', so that dataset have include 1/10 users without data after '2005/12/1'.
- Target: target is the full data of test data, it include full data of 1/10 users chosen by test set

This split way allows me can evaluate the accuracy of model predict the rating of users would give to the items they were not seen before.





# Methodology

in this section I will build and train data models over structured Netflix dataset. first I will build the user-based and item-based similarity models as a base line, and then introduce an build two different network based matrix factorization models.

## baseline

As the fundamental of the collaborative filtering methods, the item-based and user-based similarity models can be the good baseline when evaluate other models. those two are both the memory based collaborative filtering methods, they are measure the similarity of users and items by cosine, Pearson or other correlation coefficient matric to make a matrix record the similarity between users or items.

r=\frac{\sum_{i=1}^{n}\left(X_{i}-\overline{X}\right)\left(Y_{i}-\overline{Y}\right)}{\sqrt{\sum_{i=1}^{n}\left(X_{i}-\overline{X}\right)^{2}} \sqrt{\sum_{i=1}^{n}\left(Y_{i}-\overline{Y}\right)^{2}}}



### Matrix factorization

The traditional Matrix factorization algorithm is a well-known collaborative filtering algorithm using the latent factor model (LFM). The LFM is predict the rating of the user would give to the  item by characterizing both of them into their latent factor. through describe the attributes and preference of user and item into their factor matrix, the dot product of user and item factor matrix would be the predicted rating matrix.  following with this idea, the matrix factorization algorithm is a way of factorize the observed into user and item matrixes.



![](/Users/wujingyi/UTS_document/AdvDataAnalytis/UTS_ML2019_99151300/img/fm_principle.png)

The traditional Matrix factorization algorithm are used gradient decent or alternating least squares methods to minimizing the dot product result observed data. since the final goal of the algorithm is to calculate the inner product, and its non-linear, so it seems that algorithm also can implement by neural network, and even get better performance. (gink, 2018) In this case, He et al. 2017 point out that traditional dot product have limited the expression ability of MF, but through repeat it by neural network can solve that problem.

Multi-Multi-Layer Perception based



### Autoencoder 

Restricted Boltzmann machine

# Evaluation

Since the recommendation system are predict the unknown knowledge, its impossible to makeup data and hard for individual researcher to test models in actual environment, the evaluation of recommendation system become an challenge. if is an enterprise developer, the best way is to develop the model online and to check the metrics of recall rate and do A/B test. But since the online evaluation methods are impossible for individual, in this report I'm only use the off-line evaluation methods.

- accuracy

  - the accuracy is an important matric for measure the correctness of system predict the rating of user would give to items. similar with the regression problem, the main matrices of it are MAE and RMSE. 

  



By measure the MAE and RMSE, the item based is obviously accurate than user based. And the prediction time and the training time  of item-based model are also obviously longer than user based, however this calculation time cost is depend on the number of users and items, if I got the relatively lower number of items, the item based model would be better choose, vice versa.



By measure them with MAE and RMSE metrics, they have obviously high accuracy than baseline item and user based memory model, the MLP based model is more accurate than simple repeat dot product with neural network, however later has higher coverage which is better for mining potential user interesting even its usually are opposite to accuracy. They also got very shot prediction time in the 'ms' metric, and since the neural network is an O(1) algorithm, so their prediction time will not increase with the user or item increase. 

further more,  the MF have higher 







MSE

RMSE

光丿羽  16:49:45
$\operatorname { RMSE } = \frac { \sqrt { \sum _ { ( u , i ) \in T } \left( r _ { u i } - \hat { r } _ { u i } \right) ^ { 2 } } } { | \text { Test } | }$





Coverage Metric





# Ethic issue

The ethical issue is an important consideration for the technologies, The main ethical issues of recommendation system are mainly located as spread bad information and invasion of privacy. to stat with discuss those ethical issue in recommendation system, first thing to do is to briefly identifier the stakeholders and their perspectives with the consideration of those ethic issues.

- user: care about self privacy, want to see interesting information but not proactive want.
- provider: want learn benefits, sale their product to users. want their items can observed by target users.
- platform: want learn benefits and their benefits are create by users and provider. want user stickiness.
- government: want the stable and active society, disgust with bad information. less interesting with highest power.

By sort out those stakeholders, its easier to see that main beneficiary are provider and platform, and user also have benefit from it. so, I can determine that develop a recommendation system are right for enterprises by both of utilitarian and duty based ethical approaches.  In the duty based approach the objective of enterprise or employees are make the prise, while in utilitarian approach spend less prise to got more benefits to the society also is right thing, that recommendation system can active the online economic flows and leads the benefits to all user, platform and producer.

Even so, those ethical issues still are the problem must be consider with, because of spread bad information will make the society not stable and leads the government dissatisfied, also the dissatisfaction from user also is bad for any enterprises.  those issues are challenge to solve, my suggestions are the enterprise must improve their self-management don't torch the bottom line.





spread bad information

privacy



overall report, I have build four models for predict the rating of users would give to movies for make the recommendation system, and that system are solved the problem of information overload on internet, and also provides the benefits of active economical flows. as each models have their own limitation and the benefits, those system actually can collaborative together to make better prediction, for example the matrix factorization model not allowed to predict new users, but one way to solve it is to calculate the similarity of this new user with other, than use the similar user's data for make prediction.

