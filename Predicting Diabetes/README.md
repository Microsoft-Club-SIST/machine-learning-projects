# Predicting Diabetes
We are using the [PIMA Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database) to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset, **Naive Bayes**.
# Naive Bayes Classification
A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. It is built on the principles of Bayes Theorem.

## Bayes Theorem

<p align = "center">
<a href="https://www.codecogs.com/eqnedit.php?latex=P(A/B)&space;=&space;\frac{P(B/A)P(A)}{P(B)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A/B)&space;=&space;\frac{P(B/A)P(A)}{P(B)}" title="P(A/B) = \frac{P(B/A)P(A)}{P(B)}" /></a>
</p>

Using Bayes theorem, we can find the probability of **A** happening, given that **B** has occurred. **The assumption made here is that the features are independent**. That is presence of one particular feature does not affect the other. Hence it is called **naive**.

# Types of Naive Bayes Classifier:

## Gaussian Naive Bayes:

When the predictors take up a continuous value and are not discrete, we assume that these values are sampled from a gaussian distribution.

## Multinomial Naive Bayes:

This is mostly used for document classification problem, i.e whether a document belongs to the category of sports, politics, technology etc. The features/predictors used by the classifier are the frequency of the words present in the document.

## Bernoulli Naive Bayes:

This is similar to the multinomial naive bayes but the predictors are boolean variables. The parameters that we use to predict the class variable take up only values yes or no, for example if a word occurs in the text or not.
