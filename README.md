# W207_Group_Project
W207 Summer 2018 Group Project

The data have been pre-processed:

 - keep only exercises that can be classified into big 3 lifts (squat, bench and deadlift)
 - remove invalid sets and reps
 - unroll nested json

Sets data:

 - filename: set_data_w207.csv
 - unique on setID
 - contains data relevant to each set (weight, exercise, RPE etc.)

Reps data:

 - filenames (based on different app releases):
    - rep_data_w207_new.csv (contains 22 columns)
    - rep_data_w207_old17.csv (contains 17 columns)
    - rep_data_w207_old19.csv (contains 19 columns)
 - unique on setID + RepCount (RepCount starts with 0)
 - contains data relevant to each rep (average velocity, range of motion etc.)

How to combine datasets:

 1. Stack reps data (order is not important)
 2. Merge with sets data by setID

Caveats (ideas for future processing):

 - Some numeric fields are strings (weight, RPE)
 - Those strings can have , or . to indicate fractional numbers
 - Weight can be in kgs or lbs (use metric column to determine which one it is)
 - Weight and RPE can have extreme (invalid) values that need to be removed
 - Need to create cleaner labels for exercises (and potentially remove sets with tags)

