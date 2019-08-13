# Boston Crimes Dataset 

<p align = 'center'>
<img src= https://c1.staticflickr.com/5/4163/33559981643_1e7d0bd3ac_b.jpg>
</p>

## Introduction

The following dataset contains records of crimes commited in the city of Boston, MA. Some of the columns included are incident number, district, date, and offence code group. This repository will take a cursory look into the data for some insights, and follow up on a binary classification modelled to predict whether or not a given crime will involve shootings. 

As of late, there has been a large public outcry over gun control issues. Though this is a sensitive subject, perhaps it would be beneficial to know what kind of crimes involve gun shootings. 

### Data Exploration

For the large part, we will look at crimes where the UCR_PART is categorized as "Part One". This is the most "serious" types of crimes, and the insights gathered from this particular dive may not apply to the part two and part three classified crimes. 

Entries with nan values were ignored. The nan values usually occrued in irrelevant columns, and even after deleting the rows with nan's we were still left with over 58k observations.

From the seaborn catplot below, we can see that the most commong type of serious crime occuring in Boston is larceny. Larceny is defined as "a crime involving the unlawful taking of personal property of another person or business" It makes sense that even out of the most serious crimes, a simple "taking of others' stuff" is the most common.

![image](https://user-images.githubusercontent.com/49466466/62897554-8685d900-bd21-11e9-87af-0b38bb637923.png)

The plot below displays the timeline of when the crimes occur. During the day, the lowest number of crimes occur at 5am, and gradually peak at near 6pm. During the week, the highest number of crimes occur on Friday. As for the month, the highest number of crimes peak during July.

![image](https://user-images.githubusercontent.com/49466466/62897713-d6fd3680-bd21-11e9-8029-1df76f996387.png)

Finally, looking at the scatterplot below, we observe that the highest number of crimes seems to happen in districts A1 and D4, which are the most crowded downtown areas of Boston. There is also an unusually high concentration of crimes occuring in district D14.

![image](https://user-images.githubusercontent.com/49466466/62897803-f005e780-bd21-11e9-82be-4b1f76f1f873.png)

**Data Summary**
- Larceny is the most common type of serious crime.
- Serious crimes are most likely to occur in the afternoon and evening.
- Serious crimes are most likely to occur on Friday and least likely to occur on Sunday.
- Serious crimes are most likely to occur in the summer and early fall, and least likely to occur in the winter (with the exeption of - January, which has a crime rate more similar to the summer).
- Serious crimes are most common in the city center, especially districts A1 and D4.

### Imbalanced Data - Shootings

In our dataset, we observe that most of the serious crimes don't involve a shooting (fortunately). The ratio is 58208 crimes without shooting, and 636 crimes with shooting.

![image](https://user-images.githubusercontent.com/49466466/62898024-8508e080-bd22-11e9-97bd-b4657f6203d8.png)

This makes it so that any classification model built to predict on the binary variable of Shooting will almost always predict no shooting. Even if it **only ever** predicts 0, the model accuracy would be over 98%.

IF we were to just simply built a logistic regression model on this dataset, the outcome would look like below:

![image](https://user-images.githubusercontent.com/49466466/62898257-0fe9db00-bd23-11e9-9bde-3d6ef0e15b7a.png)

The accuracy looks good at **99%**, but that is because the majority of the predictions were 0 to begin with. Even if the model predicted all 0's (using the baseline model which just predicts the majority outcome no matter what) **the model would still have 98.9% accuracy.**

We can see that the recall rate of predictions for shootings being 1's are 0.2! **Only 20% of cases where there were actually a shooting involved were correctly identified.**

We can see the dangers of imbalanced datasets from this model's prediciton. We would like to know when a shooting will be involved in a crime, so let's try to "balance out" the dataset.

To deal with this issue, we can balance out the dataset using the random under sampling technique. We take our majority case (no shooting) and downsample to the number of minority case (yes-shooting) such that the new balanced dataset has a 50/50 ratio of both events.

![image](https://user-images.githubusercontent.com/49466466/62898177-e7fa7780-bd22-11e9-9b00-0217f0bfb039.png)

We can also see a slight change in the relevant feature correlations when we look at the difference in correlation between the imbalanced dataset and the new balanced dataset.

![image](https://user-images.githubusercontent.com/49466466/62898356-46275a80-bd23-11e9-9c0c-4f3589f0e89f.png)

#### Dimensionality reduction and clustering:

[t-SNE: intro-video](https://www.youtube.com/watch?v=NEaUSP4YerM)

- t-SNE algorithm can accurately cluster cases where there were shooting and non shooting cases in our dataset.
- Although the random downsampled subset is pretty small (around 600 cases each), the t-SNE algorithm detects clusters in almost every case.
- This give an indication that further predictive models will perform well in separating shooting cases from non shooting cases.

[credit to this kaggle kernel](https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)

![image](https://user-images.githubusercontent.com/49466466/62898484-9b636c00-bd23-11e9-917c-c922cc2c74a3.png)

So we now build a logistic regression model on the newly balanced dataset, and arrive at the following:

![image](https://user-images.githubusercontent.com/49466466/62898525-b209c300-bd23-11e9-8bb9-04f01c9fd623.png)

While the accuracy of the overall model went down to **92.9%**, the **recall rate of the true shooting cases has significantly increased to 95%.** This recall metric carries much more weight, knowing which crimes involve a shooting is what we are actually interested in.

Another performance metric that we can look at is the Area Under ROC Curve.

Intuitively, the AUROC represents the likelihood of your model distinguishing observations from the two classes. In other words, if you randomly select one observation from each class, what's the probability that your model will be able to rank them correctly.

![image](https://user-images.githubusercontent.com/49466466/62898569-cf3e9180-bd23-11e9-88d1-605841460350.png)

Finally, to get back to what we were initially interested in, we can look at the feature importances of the decision tree classifier to understand which feature is most important in determining whether or not a crime will involve shooting.

![image](https://user-images.githubusercontent.com/49466466/62898628-f2694100-bd23-11e9-9174-e7365518e60a.png)

We can see that crimes that are aggravated assaults and homicides carry the most amount of weight. They account for more than 95% of feature importances between them. Latitude seems to carry a small amount of importance as well, hinting that that area in which the crime occurs in may have a factor as well.

### Future Work & Conclusion

Conclusively, most crimes that happened (even the most serious ones) did not involve a shooting. However, if we were to build a predictive model to detect whether or not a crime involved shootings, we could look at aggravated assaults, homicides, and the latitude of the crimes. 

For the future, we could look to implement some more imbalanced dataset practices. A popular method is the SMOTE method: synthetic minority oversampling technique. This would kind of be the opposite of what we did here, since we would be synthetically upsampling the number of minority cases to match the majority case to balance out the dataset. 

Another venture could be to look at different UCR_PART code crimes; less serious crimes may have different feature importances and different occurance over time. We could also look to incorporate disparate data sources, such as weather into the predictive modeling scheme. There may or may not be less crimes occuring in the dead of winter, since it is too cold for anyone to stay outside that long. 

Finally, we could look to implement neural networks or other sort of predictive measures other than the 4 major classification schemes covered in this repository. 
