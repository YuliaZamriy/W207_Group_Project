# Integrating Data Science Into Your Training Protocol

## How Machine Learning Algorithms Can Help Improve Quatlity of Data and Provide Insights for Smarter Training Protocols

*Author: Yulia Zamriy*

### Table of Contents

- [Introduction](#introduction)
- [What is OpenBarbell?](#what-is-openbarbell)
- [What do we want to achieve?](#what-do-we-want-to-achieve-here)
- [Exercise Classification](#exercise-classification)
- [Lifter Clustering](#lifter-clustering)
- [Conclustions](#conclusions)

### Introduction

Data is the belle of the ball these days, and it's time to apply data science in the weight room. This could mean anything from getting more accurate 1RM estimate to preventing injury. 

Data science has many faces. Randomized controlled trials, the [only reliable method](https://www.bmj.com/content/316/7126/201) to prove cause-effect relationship, have been employed by sports science for many years. But the findings tend to be limited to a small group of participants, most likely young male. On the other hand, there is machine learning that works with big data to make predictions and detect patterns. And that's where OpenBarbell can shine. 

This paper is our first attempt to demonstrate how we are using data collected from the device to improve your experience in the weight room. First, we will demonstrate how metrics captured in the background can detect performed exercise with 96% accuracy. Why does it matter? User inputs are infamously unreliable, unfortunately. But a lot of our analytics reports depend on accurate exercise specification. In the second section, we will share our first efforts to perform lifter clustering. This is a hard problem to solve, but it has large implications for building effective and cost-efficient custom training programs.

#### Acknowledgments

This white paper is the result of a project developed by data science students at UC Berkeley: [Renzee Reyes](https://www.ischool.berkeley.edu/people/renzee-reyes), [Tim Witthoefft](https://www.ischool.berkeley.edu/people/tim-witthoefft), [Jack Workman](https://www.ischool.berkeley.edu/people/jack-workman) and [Yulia Zamriy](https://www.ischool.berkeley.edu/people/yulia-zamriy). 

### What is OpenBarbell?

If you are reading this white paper, you are most likely best friends with OpenBarbell, or at least its good acquaintance. But a quick refresher can only help with what we are going to cover in this paper:

>*OpenBarbell is a device with a retractable string that attaches to your workout equipment to deliver data such as speed or range of motion<sup id="a1">[1](#f1)</sup>.*

The key word in the statement above is *data*. By the way, you have access to that data as well (assuming you are one of the proud owners) through the OpenBarbell app. And if you are curious about the full list of metrics the device is capturing, check out our [Wiki page](https://github.com/squatsandsciencelabs/OpenBarbell-V3/wiki). 

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/OB.png?raw=true" width="350" height="350" title="OpenBarbell">
  <img height="350" hspace="50"/>
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/store_app.png?raw=true" width="200" height="350" title="App">
</p>

### What do we want to achieve here?

For the purposes of this analysis we will focus on the *big three*: `squat`, `bench press` and `deadlift` (including sumo) performed with a bar. The OpenBarbell is usually attached by a string to the bar, and the data is recorded during the concentric part of the movement. Every user of the app will get a quick snapshot of the data immediately after each rep:

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/app_inside.png?raw=true=true" width="200" height="350" title="InsideApp">
</p>

In general, there are three types of data for each set:

| Rep-specific | Set aggregates | User inputs |
|---|---|---|
| Rep number | Peak-end velocty<sup id="a2">[2](#f2)</sup> | Exercise name |
| Average velocity | Estimated 1RM | Weight |
| Peak velocity | Minimum range of motion (**check**) | Metric (kgs/lbs) |
| Peak velocity height | Last rep peak velocity (**check**) | RPE |
| Randge of motion | Minimum set duration (**check**) | Tags |
| Duration |  | Video Log |
| [Other](https://github.com/squatsandsciencelabs/OpenBarbell-V3/wiki/OpenBarbell-Data-Storage-Format) | |  |

The app does not have any restrictions on the inputs in the text fields (**check**) (i.e., exercise and tags), and OpenBarbell users can get very creative with their naming conventions. However, if we wanted to analyze the good old squat, we would need to sift through all the records and extract the correct ones, which is time-consuming and prone to error. Moreover, what if a user did not impute any exercise name, but it was still a valid set? It's a valuable data point that cannot be used without exercise label. 

A supervised classification algorithm can help us solve the above stated problems. And if it's good enough (we'll cover that in the next section), it can eliminate the need for user input in general. AI in action! But not really (forgive us for our eagerness to be on top of the trend to call everything data-related AI). 

Hence, the first part of the analysis is all about exercise classification:

>Can we use machine learning to accurately classify exercise type based on measurements taken by the OpenBarbell device?

The second question is a more complex and does not have one correct answer:

>Is there an inherent grouping of lifters based on their lifting parameters (i.e., velocity, range of motion, etc.) that could be leveraged to calibrate their training?

There are fast lifters and there are grinders. There are people who fail deadlifts off the floor, and there are those who struggle with a lockout. And yet programs prescribing everyone the same percent of 1RM for the same number of sets/reps are still common because everyone is different and it's almost impossible to customize percentages on case by case basis. Maybe big data can change that.

###  Exercise Classification

#### Algorithm

As mentioned above, we used [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) to solve our first problem. The dataset contained a subset of OpenBarbell database that had sets with clearly labeled exercises. Clear labels for squats, bench press and deadlift were used as a source of truth in detecting patterns distinguishing the big three lifts. The model was trained on 80% of the data<sup id="a3">[3](#f3)</sup>, while the remaining 20% were used to validate it. The accuracy of the model was determined by whether the patterns detected on the 80% were distinct enough to correctly classify exercises on the subset of data that the model has not seen before.

There are a lot of supervised classifiction algorithms out there. We tested four classic ones: [KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm), [Random Forest](https://en.wikipedia.org/wiki/Random_forest), [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression), and [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier). The first two were distinct winners (their accuracies were almost identical), and the final choice of the algorithm will depend on the implementation. 

#### Results

<p align="float">
  <img align="right" src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/RF_matrix1.png?raw=true" width="350" height="350" title="matrix">
</p>

How well did the model work? 96% of the sets in the data put aside for validation were correctly classified by exercise<sup id="a4">[4](#f4)</sup>. However, there are some interesting details to discover about each of three exercises on a more granular rep level. 

For example, as illustrated by the matrix on the right, bench press is the easiest to detect (99% of all bench press exercises were correctly identified). On the other hand, the model had some trouble distinguishing deadlifts from squat: only 85% of deadlifts were correctly identified, and 11% of actual deadlifts were classified as squats. We can speculate about why this happens (similar range of motion, for example), but instead we'll just say that there is some room for improvement and more interesting insights to look for. 

And what were the most important features (a.k.a measurements) that differentiated the three lifts? `Range of motion` was by far the most significant one. `Peak Velocity Location`, `Peak Velocity` and `Lifted Weight` were also on the top 5 list (the distant fifth was `Rep Duration`).

There are different ways of using these results. The most obvious and immediate benefit is cleaner data for analysis (and more interesting reports for you as a result). Moreover, we could take this analysis further and start looking at how distinguishable the exercises are by lifter. This actually brings us to the next topic.

### Lifter Clustering

#### Framework

For this part of the analysis we aggregated the data to user level. As a result our dataset consisted of 250-300 records depending on the exercise. This was more than sufficient to start investigating which captured mesaurements differentiate athletes. 

Why would this analysis matter to you? Have you ever used a program that was based on your 1RM? For example, you would use a table similar to the below to calculate what weight to use for your set of 5 at RPE 8.0.

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/pct_1rm.png?raw=true" width="750" height="250" title="1rm">
</p>

Or if you have a velocity measurement device, you probably used target peak-end velocity to inform your training<sup id="a5">[5](#f5)</sup>:

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/vel_cal.png?raw=true" width="750" height="250" title="velocity">
</p>

Both methods have been developed for an *average* lifter. Unfortunately, this *average* can mean a lot of different things and is defined by a group of people used in developing the above tables. 

What if we could analyze all the lifters in our database, determine *natural* groupings based on their performance metrics and create multiple versions of the tables above for each distinct lifter profile? Then we would provide you with a small set of rules to detect your profile membership and pick the right calibration table for you. But that's only one of possible applications of lifter profiles. We are also working on using them to improve [1RM calculation](https://en.wikipedia.org/wiki/One-repetition_maximum) because there is no one formula that fits all. 

#### Algorithm 

This analysis can be described as exploratory because our goal was to discover natural groupings of lifters based on various performance metrics. At the beginning, we didn't even know how many different profiles there should be. To start things off we created a dataset split by exercise and aggregated it to OpenBarbell user level (and stripped all the data that was not relevant to lift performance). The idea of splitting data by exercise was driven by the assumption that the same lifter can belong to different groups in all three exercises.

The next step was to apply [hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) to determine how many distinct clusters we can identify in the data. This part is more of an art than science because the results are dependent on wide range of factors (selected features, distance metric, linkage function etc.). Hence, it's important to set clear goals for the analysis from the get-go. Another key to a good set of clusters is being able to interpret what makes these groups of lifters distinct. The solution can't be a black box.

The outcome of hierarchical clustering is captured by [dendrogram](https://en.wikipedia.org/wiki/Dendrogram) (see below). It has three key elements:

1. Three colors (green, red and cyan) identify potential clusters of lifters (the number is an arbitrary choice based on visual inspection of the chart)
2. Vertical lines represent individual lifters. The chart has been truncated at the bottom for simplification (the numbers along the X-axis represent how many individuals were grouped together to create that line)
3. The length of the vertical lines amounts to the magnitude of differences between groups of lifters

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/bench_dendr.png?raw=true" width="500" height="350" title="dendr">
</p>

#### Results

For simplicity let's focus on bench press only. Our current solution is work in progress. But we want to show you the power of data analysis and its potential in the weight room.

Our first workable solution for bench press consists of three clusters of lifters with 57, 116 and 117 members in each (they are captured by the chart above). Four metrics were used to form these clusters: `Peak Velocity`, `Peak Velocity Location`, `Rep Duration` and `Lifted Weight`. 

Since there are four metrics included in the solution, it's hard to visualize selected clusters with more than two metrics at a time (3D plots are cool, but even harder to read). The chart below is not easy to interpret at first glance either. It consists of three parts:

1. 4 diagonal plots: density distributions of the above stated metrics by cluster (color). The further apart the three (almost) bell-shaped curves are, the more difference exists among clusters
2. 6 upper corner plots (above diagonal): bivariate scatterplots. Each dot represents one lifter colored by cluster membership
3. 6 lower corner plots (lower diagonal): bivariate density distributions. Each blob is a separate cluster with the darkest area representing metric mean, while contours represent metric variance

The more you stare at it, the more insights you can extract. For example, the one metric where all three clusters look distinctly separated appears to be `Peak Velocity Location` (plot in row 3, column 3). That means that people differ significantly at a point where they reach peak velocity on bench press. However, `lifted weight` can also tell us a lot about these clusters. The red cluster has a heavier tail on the left, which means that it captures athletes that have lighter bench. Heavy benchers, on the other hand, are grouped into the purple cluster with a long tail.

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/pairgrid.png?raw=true" width="700" height="600" title="pairs">
</p>

A simpler way to look at the clusters is to compare averages of the selected metrics (see table below). It's a small set of numbers, but there is a lot of insight to drive from it. For example, *cluster 1* has the widest `Range of Motion`, highest `Velocities` and lightest `Lifted Weight`. Would that mean that this cluster captures relatively new to lifting athletes? We'll leave this question for you to contemplate.

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/clusters.png?raw=true" width="700" height="150" title="pairs">
</p>

This solution is not final. We might want to split up those clusters even further, or try including more metrics for higher precision. As mentioned previously, one of our goals for these clusters is to develop 1RM calculation that is more customizable to various types of lifters. If we do succeed, it will be exciting for us as it proves the power of data, and for you because it means better programming.

#### Conclusions

If you got to this section, it means you found something interesting in this paper. We don't know if it's because you like data analysis or because you want to know how to apply this in your weight room. In either case, we hope you found something useful and you'll stay tuned for more reports like that.

In terms of next steps, we are planning to write two blog posts about exercise prediction and lifter clustering. They will go into more technical details covering techniques and our approach. In the meantime, we'll keep crunching the numbers to get the rest of the answers (improve squat vs deadlift classification, finalize clusters for three lifts, revise 1RM calculation, etc.). Hence, there will be something for everyone.

If you have any questions, please reach out! We'll be more than happy to discuss our approach, take suggestions for improvement and answer any questions.


**Footnotes**

<b id="f1">1.</b> OpenBarbell is no longer in production. S&S is working on [RepOne](https://www.reponestrength.com/), full weight room analytics platform.[↩](#a1)

<b id="f2">2.</b> The mean of the slowest and last repetitions (average velocity) of a set.[↩](#a2)

<b id="f3">3.</b> Our full analysis dataset for exercise classification consisted of around 180,000 repetitions.[↩](#a3)

<b id="f4">4.</b> At the first stage of the analysis, exercise prediction was done for each rep. But the final prediction was established on a set level by selecting the most common exercise assignment among reps in a set.[↩](#a4)

<b id="f5">5.</b> We recently wrote how to customize these tables based on your historical metrics. You can check out [this blog post](https://blog.squatsandscience.com/calibrate-your-strength-6a7d9d83f993).[↩](#a5)
