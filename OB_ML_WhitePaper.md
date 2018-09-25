<center> <h2>Integrating Data Science Into Your Training Protocol</h2> </center>

<center> <h3>How Machine Learning Algorithms Can Help Improve Quatlity of Data and Provide Insights for Smarter Training Protocols</h3> </center>

<center> <h4>Authors:</h4></center>

### Table of Contents

1. Introduction  
2. What is OpenBarbell?
3. What do we want to achieve? Motivation
4. Exercise Classification
5. Lifter Clustering
6. Next Steps

### 1. Introduction

Data is belle of the ball these days, and it's time to apply data science in the weight room. It can mean anything from estimating your 1RM to preventing injury. 
Data science has many faces. For example, randomized control trials, the only reliable method to prove cause-effect relationship (insert reference), have been employed by Sports Science for many years. But the findings tend to be limited to a small group of participants, most likely young male. 
On the other hand, there is machine learning that works with big data to make predictions and detect patterns. And that's where OpenBarbell can shine. 
This paper is our first attempt to demonstrate how we are using all the data collected from the device to improve your experience in the weight room. First, we will demonstrate how metrics captured in the background can detect performed exercise with 96% accuracy. Why does it matter? User inputs are infamously unreliable, unfortunately. But a lot of our analytics reports depend on accurate exercise definition. 
And in the second section, we will share our first efforts to perform lifter clustering. This is a hard problem to solve, but it has large implications for building effective and cost-efficient custom training programs.

#### Acknowledgments

This white paper is the result of a project developed by data science students from UC Berkeley: Renzee Reyes, Tim Witthoefft, Jack Workman and Yulia Zamriy (insert links to their UC Berkeley profiles?). 

### 2. What is OpenBarbell?

If you are reading this white paper, you are most likely best friends with OpenBarbell. Or at least a good acquaintance. But a brief refresher can only help with what we are going to cover in this paper.

No one can explain it better than [Squats and Science](http://squatsandscience.com/openbarbell/) itself: 

>*OpenBarbell is a device with a retractable string that attaches to your workout equipment to deliver data such as speed or range of motion.*

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/OB.png?raw=true" width="350" height="350" title="OpenBarbell">
</p>

The key word in the statement above is *data*. By the way, you have access to that data as well (assuming you are one of the proud owners of this device) though the OpenBarbell app:

<p align="center">
  <img src="https://github.com/YuliaZamriy/W207_Group_Project/blob/master/images/store_app.png?raw=true" width="200" height="350" title="App">
</p>

If you are curious about what the device is capturing, check out our [Wiki page](https://github.com/squatsandsciencelabs/OpenBarbell-V3/wiki). It's a lot! And we are making a good use of it. How? Keep reading.

### 3. What do we want to achieve?

For the purposes of this analysis we will focus on the *big three*: squat, bench press and deadlift (including sumo whether you like it or not) performed with a bar. The OpenBarbell is usually attached by a string to the bar, and the data is recorded during the concentric part of the movement.

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

The app does not have any restrictions on the inputs in the text fields (**check**) (i.e., exercise and tags). And that's where we see a wide range of inputs for the same things (**can we include examples?**). 

However, if we want to analyze *squat*, how do we filter it out? Moreover, what if a user did not input any exercise name, but it was still a valid set? A supervised classification algorithm can help us with that! And if it's good enough (we'll cover that as well), it can eliminate the need for user input in general. It's almost AI! But not really (forgive us our eagerness to be on top of the trend to call everything data-related AI). 

Hence, the first part of the analysis is all about exercise classification:

>Can we create an algorithm that will accurately classify exercise type based on measurements taken by the OpenBarbell device?

The answer is yes.

The second question we are still trying to answer is a lot more complex and does not have one correct answer:

>Is there an inherent grouping of lifters based on their lifting parameters (i.e., velocity, range of motion, etc.) that could be leverage for calibrating their training?

It is obvious to everyone who ever lifted any weights that every lifter is different. There are fast lifters and there are grinders. There are people who fail deadlifts off the floor, and there are those who struggle with a lockout. However, the programs prescribing everyone the same percent of 1RM for the same number of sets/reps are still common. 
