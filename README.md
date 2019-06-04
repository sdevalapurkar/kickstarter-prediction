# Predicting Kickstarter Campaign Success

Predicting the success of kickstarter campaigns!

## The Dataset

The data we had to work with was a list of over 350,000 recent Kickstarter campaigns obtained from Kaggle but collected from the Kickstarter platform, with their features available in a tabular format. The features of these campaigns included, but were not limited to, title, country, main category, success state, monetary goal, amount of money pledged, and the number of backers.

## Problem Statement and Objective

The main problem that made our project interesting was that although our dataset contained many features, not all of these features would be available for analysis purposes upon a campaign’s inception. Features such as the number of backers or the amount of money pledged would only be known when the success state of a campaign had been determined. Taking this into account, we decided to only examine features that would be available at campaign inception and solve the problem of understanding what makes a Kickstarter campaign a success or a failure.

Out of all the attributes available to us upon campaign inception, we decided to focus most on the campaign title. Our hypothesis was that the campaign title would be an extremely important feature because anytime someone views an advertising campaign or petition, the first thing that they look at is its title or slogan. As once said by Paul Hoffman (CTO, Space-Time Insight), “if you want to understand people, especially your customers, then you have to be able to possess a strong capability to analyze text [3].” As a result, we believed that the way a title is phrased could have a significant impact on campaign success. Our goal was to use the campaign title along with other features to predict with relative accuracy, the success or failure of a kickstarter campaign.

We used the positivity score as well as the reading ease score of the campaign title to predict campaign success.

## Conclusion

Our initial hypothesis was proven to be partially correct, while also being slightly inaccurate. It has been shown that the title of a Kickstarter campaign is a very important feature when it comes to being able to effectively predict whether or not a campaign will be successful. Using a Linear Support Vector classification approach, the success state of a campaign can be predicted with 66% accuracy when the positivity and reading ease scores are included in the feature set, and only with an accuracy of 36% when they are not. Interestingly, although these features are instrumental in allowing the model to accurately predict campaign success, neither the positivity or reading ease of a kickstarter campaign title are very influential in creating a successful campaign; although it might benefit a campaign starter to create a simple-to-read title.
