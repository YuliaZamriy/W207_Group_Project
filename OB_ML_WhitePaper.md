#Integrating Data Science Into Your Training Protocol

##How Machine Learning Algorithms Can Help Improve Quatlity of Data and Provide Insights for Smarter Training Protocols

*Authors:*

### Table of Contents

1. Introduction  
2. What is OpenBarbell?
3. What do we want to achieve? Motivation
4. Exercise Classification
5. Lifter Clustering
6. Next Steps

### 1. Introduction

Data is belle of the ball these days, and it's time to apply data science in the weight room. This could mean anything from estimating your 1RM to preventing injury. 
Data science has many faces. For example, randomized control trials, the only reliable method to prove cause-effect relationship (**insert reference**), have been employed by Sports Science for many years. But the findings tend to be limited to a small group of participants, most likely young male. 
On the other hand, there is machine learning that works with big data to make predictions and detect patterns. And that's where OpenBarbell can shine. 
This paper is our first attempt to demonstrate how we are using all the data collected from the device to improve your experience in the weight room. First, we will demonstrate how metrics captured in the background can detect performed exercise with 96% accuracy. Why does it matter? User inputs are infamously unreliable, unfortunately. But a lot of our analytics reports depend on accurate exercise specification. 
And in the second section, we will share our first efforts to perform lifter clustering. This is a hard problem to solve, but it has large implications for building effective and cost-efficient custom training programs.

#### Acknowledgments

This white paper is the result of a project developed by data science students from UC Berkeley: Renzee Reyes, Tim Witthoefft, Jack Workman and Yulia Zamriy (**insert links to their UC Berkeley profiles?**). 

### 2. What is OpenBarbell?

If you are reading this white paper, you are most likely best friends with OpenBarbell. Or at least its good acquaintance. But a brief refresher can only help with what we are going to cover in this paper.

No one can explain it better than [Squats and Science](http://squatsandscience.com/openbarbell-v3/) itself (**insert reference to RepOne)**: 

>*OpenBarbell is a device with a retractable string that attaches to your workout equipment to deliver data such as speed or range of motion.*

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/OB.png?raw=true" width="350" height="350" title="OpenBarbell">
</p>

The key word in the statement above is *data*. By the way, you have access to that data as well (assuming you are one of the proud owners) through the OpenBarbell app:

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/store_app.png?raw=true" width="200" height="350" title="App">
</p>

If you are curious about what the device is capturing, check out our [Wiki page](https://github.com/squatsandsciencelabs/OpenBarbell-V3/wiki). It's a lot! And we are making a good use of it. How? Keep reading.

### 3. What do we want to achieve here?

For the purposes of this analysis we will focus on the *big three*: squat, bench press and deadlift(**include reference to a video of three lifts**) (including sumo whether you like it or not) performed with a bar. The OpenBarbell is usually attached by a string to the bar, and the data is recorded during the concentric part of the movement.

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/InTheGym.png?raw=true" width="400" height="150" title="InTheGym">
</p>

Every user of the app will get a quick snapshot of the data immediately after each rep.

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/app_inside.png?raw=true=true" width="200" height="350" title="InsideApp">
</p>

In general, there are three types of data for each set (**create table**):

1. Rep-specific: 

- Rep number
- Average velocity
- Peak velocity
- Peak velocity height
- Randge of motion
- Duration
- And there is a lot [more!](https://github.com/squatsandsciencelabs/OpenBarbell-V3/wiki/OpenBarbell-Data-Storage-Format) in the background

2. Set aggregates:

- Peak-end velocty (insert reference)
- Estimated 1RM (insert reference)
- Minimum range of motion (**check**)
- Last rep peak velocity (**check**)
- Minimum set duration (**check**)

3. User inputs:

- Exercise name
- Weight
- Metric (kgs/lbs)
- RPE
- Tags
- Video log

The app does not have any restrictions on the inputs in the text fields (**check**) (i.e., exercise and tags). And that's where OB users get creative with their namin conventions (**can we include examples?**). 

However, if we want to analyze good old *squat*, how do we filter it out? Moreover, what if a user did not input any exercise name, but it was still a valid set? A supervised classification algorithm can help us with that! And if it's good enough (we'll cover that as well), it can eliminate the need for user input in general. It's almost AI! But not really (forgive us our eagerness to be on top of the trend to call everything data-related AI). 

Hence, the first part of the analysis is all about exercise classification:

>Can we use machine learning to accurately classify exercise type based on measurements taken by the OpenBarbell device?

The answer is yes.

The second question we are still trying to answer is a lot more complex and does not have one correct answer:

>Is there an inherent grouping of lifters based on their lifting parameters (i.e., velocity, range of motion, etc.) that could be leverage for calibrating their training?

It is obvious to everyone who ever worked with a bar that every lifter is different. There are fast lifters and there are grinders. There are people who fail deadlifts off the floor, and there are those who struggle with a lockout. However, the programs prescribing everyone the same percent of 1RM for the same number of sets/reps are still common. 

### 4. Exercise Classification

#### The Algorithm (a.k.a., you can skip it if this is too boring)

As mentioned above, we used supervised classification algorithms to predict excercise name. No, we didn't higher a supervisor. It just means that to build and verify the model we used a subset of OB dataset that has clear labels for squats, bench press and deadlift. The idea was that those clearly labeled sets can help us detect patterns distinguishing the big three lifts. This stage is called training the model, and we used 80% of the clean dataset for that (**include number of records**). The remaining 20% of the sets were used to validate it: are the patterns detected on the 80% are distinct enough to correctly classify exercises on the subset of data that the model has not seen before?

There are lot of supervised classifiction algorithms out there. We tested four classic ones: KNN, Random Forest, Logistic Regression, and Gaussian Naive Bayes (**insert references**). The first two were distinct and almost tied winners. However, for reasons that will be explained later, Random Forest has more potential for actual implementation.

#### The Results

So how well did the model work? On the validation set 96% of our sets were correctly classified by exercise (**insert explanation about sets vs. reps**). However, there are some interesting details to discover about each of three exercises on a more granular (rep) level. For example, bench press is the easiest to detect (99% of all bench press exercises were correctly identified). On the other hand, the model had some trouble distinguishing deadlifts from squat: only 85% of deadlifts were correctly identified, and 11% of actual deadlifts were classified as squats. We can speculate about why this happens (similar range of motion, for example), but we instead we'll just say that there is some room for improvement and more interesting insights to look for. 

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/RF_matrix1.png?raw=true" width="250" height="250" title="matrix">
</p>

So what were the most important features (a.k.a measurements) that differetiated the three lifts? Range of motion was by far the most significant one. Peak Velocity Location, Peak Velocity and Lifted Weight were also on the top 5 list (the distant fifth was rep duration).

#### So what?

There are different ways of using these results. The most obvious and immediate benefit is cleaner data for analysis (and more interesting reports for you as a result). Moreover, we could take this analysis much further and start looking at how distinguishable the exercises are by lifter.

**Should we talk about integrating this into the app?**

### 5. Lifter Clustering

Squats And Science produced 1,000 OpenBarbells, and 200-300 (**confirm the number**) are actively used by lifters. This provides sufficient data to start annalyzing what captured mesaurements differentiate athletes. Why would that matter to you? Have you ever used a program that was based on your 1RM? For example, you would use a table similar to below to calculate what weight to use for your set of 5 at RPE 8.0.

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/pct_1rm.png?raw=true" width="750" height="250" title="1rm">
</p>

Or if you have a velocity measurement device, you probably used target peak-end velocity to inform your training (**include reference to the white paper**):

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/vel_cal.png?raw=true" width="750" height="250" title="velocity">
</p>

But both tables above have been developed for an *average* lifter. Unfortunately, this *average* can mean a lot of different things and is defined by a group of people used in developing the above tables. 

What if we could analyze all the lifters in our database, determine *natural* groupings based on their performance metrics and create multiple versions of the tables above for each distinct lifter profile? Then we would provide you with a small set of rules to detect your profile membership and pick the right calibration table for you.

But that's only one of possible applications of lifter profiles. We are actually working on using them to re-evaluate how 1RM should be calculated (**include reference to different calculations**) as no one formula fits all. Moreover, a lot of lifters like to compare themselves to others (let's blame it on Instagram). But what if instead comparing to those who have very different parameters, you could compare yourself to similar lifters (Disclaimer: comparing yourself to other might not be good for your physical or mental health!). 

#### The Algorithm (again, it's skippable)

This part of our analysis can be desribed as exploratory since there is no one set definition of a lifter profile. At the beginning, we didn't even know how many different profiles should be there. To start things off we created a dataset was split by exercise and aggregated to OB user level (**include a note on keeping only performance metrics and stripping any all other information**). Hence, all the lifter profiles would be exercise specific (**user X can belong to different groups in all three exercises since it's hard to imagine that people are similar in all three lifts**).

The next step was to apply hierarchical clustering (**reference**) to determine how many distinct clusters we can identify in the data. This part is more of an art than science because the results are highly impacted by various factors (features, distance metric, linkage function etc.). Hence, it's important to know ahead of time the ultimate goal: how these clusters will be used. Another key to a good set of clusters: can we interpret what differentiates this groups of lifters? The solution can't be a black box.

Our initial solution is shown below in a chart called dendrogram. It has three key elements to focus on:

1. Three colors green, red and cyan identify potential clusters of lifters (this is an arbitrary decision based on visual inspection of the chart)
2. Vertical lines represent individual lifters. The chart has been truncated at the bottom for simplification (the numbers along the X-axis represent how many individuals were grouped together to create that line)
3. The length of vertical lines represent how different lifters are in terms of chosen measurements (more on that below)

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/bench_dendr.png?raw=true" width="500" height="350" title="dendr">
</p>

#### Initial Findings

For simplicity (as much as it's possible) let's focus on the bench press only. Our current solution is work in progress. But we want to show you the power of data analysis and its potential inthe weight room.

Our first workable solution for bench press consists of three clusters of lifters with 57, 116 and 117 members in each. It was determined based on including four metrics: Peak Velocity, Peak Velocity Location, Rep Duration and Lifted Weight. 

The chart below requires is not easy to interpret at first glance. It consists of three parts:

1. Diagonal: density distributions of the above stated metrics by cluster (color). The further apart the three (almost)bell-shaped curves, the more difference exists among clusters.
2. Upper corner (above diagonal): bivariate scatterplot. Each dot represents one lifter colored by cluster membership. 
3. Lower corner (lower diagonal): bivariate density distribution. Each blob is a separate cluster with the darkest area representing metric mean, while contours represent metric variance. 

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/bench_clusters_pair.png?raw=true" width="700" height="600" title="pairs">
</p>

Since there are four metrics included in the solution, it's hard to visualize selected clusters with more than two metrics at a time (3D plots are cool, but even harder to read). But the more you stare at it, the more insights you can extract. For example, the only metric where all three clusters a distinctly separate appears to be Peak Velocity Location (plot in row 3, column 3). That means that people differ significantly at a point where they reach peak velocity on bench press. 

#### What's Next?

