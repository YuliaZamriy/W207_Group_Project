import pandas as pd
import numpy as np

data_df = pd.read_json('/home/yulia/Documents/SNS/Data/obdatadump.json', lines = True)
list(data_df) 

# deletinng unnecessary columns
del data_df['videoType']
del data_df['videoFileURL']

data_df.shape
# (127833, 16)

"""Selecting big 3 lifts"""

exercise = data_df['exercise'].value_counts()
exercise.shape
# (6251,)

exercise_names = [ex.lower() for ex in list(exercise.keys())]
len(exercise_names)
# 6521
len(set(exercise_names))
# 5727

#convdl = ["conventional deadlift", "conv deadlift", "conv dl", "conventional dl",
#          "competition deadlift", "comp deadlift", "comp dl"]
#convdl_list = []
#for ex in data_df['exercise']:
#    if type(ex) == str and ex not in convdl_list and any(word in ex for word in convdl):
#       convdl_list.append(ex)
#convdl_list

valid_exercise_list = ['bench', 'bench press', 'bp', 'competition bench', 'comp bench',
              'squat', 'back squat', 'competition squat', 'comp squat',
              "deadlift", "sumo_deadlift", "conventional deadlift", "comp deadlift",
              "competition deadlift"]

data_df['exercise_valid'] = data_df['exercise'].str.lower().str.strip().isin(valid_exercise_list)
np.sum(data_df['exercise_valid'])
# 52842

dirty_exercise_list = []
for ex in data_df['exercise']:
    if type(ex) == str and ex not in dirty_exercise_list and any(word in ex for word in valid_exercise_list):
        dirty_exercise_list.append(ex)

len(dirty_exercise_list)
# 1328

data_df['exercise_dirty'] = data_df['exercise'].str.lower().str.strip().isin(dirty_exercise_list)
np.sum(data_df['exercise_dirty'])
# 70560

data_df_study = data_df.loc[data_df['exercise_dirty'] == 1]
data_df_study.shape
# (70560, 18)

"""Removing invalid work sets"""

data_df_study = data_df_study.loc[data_df_study['removed'] == 0]
data_df_study.shape
# (68593, 18)

data_df_study['deleted'].value_counts(dropna=False)
#NaN     41710
# 0.0    26447
# 1.0      436

data_df_study = data_df_study.loc[(data_df_study['deleted'].isnull()) | (data_df_study['deleted'] == 0)]
data_df_study.shape
# (68157, 18)

# write out sets data to another csv without rep data
set_df = data_df_study.drop('reps', axis = 1)
set_df.shape
# (68157, 17)
set_df.to_csv('/home/yulia/Documents/SNS/Data/set_data_w207.csv')

"""Reps: New Data Format"""

data_df_new = data_df_study.loc[pd.notna(data_df['initialStartTime'])]
data_df_new.shape
# (62439, 18)

reps_list = data_df_new['reps'].tolist()
sets_list = data_df_new['setID'].tolist()

# setting names for the first 22 metrics in reps['data']
colnames = ['StartMessg',   'RepN',         'AvgVel',   'ROM',          'PeakVel',  
            'PeakVelLoc',   'PeakAccel',    'RepDur',   'TimeBWReps',   'TimeRepComp', 
            'TimeRepWait',  'SlowAllow',    'Backlight','MinAllow',     'ComprEnable', 
            'FiltrEnable',  'CodeV',        'UnitN',    'LED',          'Bright', 
            'LowPow',       'BulkStart']

# creating a dictionary for all sets and reps
# primary key: setID 
# secondary key: rep count
# values: rep values and rep['data'] values
# keeping only the first 13 rep['data'] values
# I'm getting rid of those with less than 13 elements
set_dict = {}
for set_index in range(len(reps_list)):
    set_dict[sets_list[set_index]] = {}
    for n, rep in enumerate(reps_list[set_index]):
        if len(rep.get('data')) > 12:
            set_dict[sets_list[set_index]][n] = rep.copy()
            for col_index in range(13):
                set_dict[sets_list[set_index]][n][colnames[col_index]] = rep.get('data')[col_index]
            del set_dict[sets_list[set_index]][n]['data']

len(set_dict)
# 62439 number of unique sets

# creating a list of lists
# each inner list = one rep with setID and rep count 
set_dict_list = []
for setid in set_dict:
    for repnum in set_dict[setid]:
        temp_list = [setid, repnum]
        for metric in set_dict[setid][repnum]:
            temp_list.append(set_dict[setid][repnum][metric])
        set_dict_list.append(temp_list)
len(set_dict_list)
# 255269 number of unique reps

# creating column names for the output dataframe
some_set_id = list(set_dict.keys())[0]
colnames_full = ['setID', 'RepCount']
for metric in set_dict[some_set_id][0]:
    colnames_full.append(metric)
len(colnames_full)
# 22

# converting list of lists to a dataframe
rep_df = pd.DataFrame(set_dict_list)
rep_df.columns = colnames_full
rep_df.shape
# (255269, 22)

rep_df['isValid'].value_counts(dropna=False)
#True     255108
#False       161

rep_df['removed'].value_counts(dropna=False)
# False    241838
# True      13431

rep_df = rep_df.loc[rep_df['removed'] == 0]
rep_df.shape
# (241838, 22)

rep_df = rep_df.loc[rep_df['isValid'] == 1]
rep_df.shape
# (241761, 22)

# writing out reps data to a csv
rep_df.to_csv('/home/yulia/Documents/SNS/Data/rep_data_w207_new.csv')

"""Reps: Old Data Format"""

data_df_old = data_df_study.loc[pd.isna(data_df_study['initialStartTime'])]
data_df_old.shape
# (5718, 18)

reps_list = data_df_old['reps'].tolist()
sets_list = data_df_old['setID'].tolist()

set_dict = {}
# first record is nan, hence, skip it
for set_index in range(1, len(reps_list)):
    set_dict[sets_list[set_index]] = {}
    for n, rep in enumerate(reps_list[set_index]):
        if len(rep.get('data')) > 12:
            set_dict[sets_list[set_index]][n] = rep.copy()
            for col_index in range(13):
                set_dict[sets_list[set_index]][n][colnames[col_index]] = rep.get('data')[col_index]
            del set_dict[sets_list[set_index]][n]['data']
len(set_dict)
# 5717

set_dict_list17, set_dict_list19 = [], []
for setid in set_dict:
    for repnum in set_dict[setid]:
        temp_list = [setid, repnum]
        for metric in set_dict[setid][repnum]:
            temp_list.append(set_dict[setid][repnum][metric])
        if len(temp_list) == 17:      
            set_dict_list17.append(temp_list)
        elif len(temp_list) == 19:
            set_dict_list19.append(temp_list)
len(set_dict_list17)
# 8519
len(set_dict_list19)
# 17493

# creating column names for the output dataframe
colnames_full17 = ['setID', 'RepCount']
for metric in set_dict[set_dict_list17[0][0]][0]:
    colnames_full17.append(metric)
len(colnames_full17)
# 17

colnames_full19 = ['setID', 'RepCount']
for metric in set_dict[set_dict_list19[0][0]][0]:
    colnames_full19.append(metric)
len(colnames_full19)
# 19

rep_df17 = pd.DataFrame(set_dict_list17)
rep_df17.columns = colnames_full17
rep_df17.shape
# (8519, 17)

rep_df19 = pd.DataFrame(set_dict_list19)
rep_df19.columns = colnames_full19
rep_df19.shape
# (17493, 19)

rep_df17['isValid'].value_counts(dropna=False)
#True     8516
#False       3

rep_df17['removed'].value_counts(dropna=False)
# False    8309
# True      210

rep_df17 = rep_df17.loc[rep_df17['removed'] == 0]
rep_df17.shape
# (8309, 17)

rep_df17 = rep_df17.loc[rep_df17['isValid'] == 1]
rep_df17.shape
# (8306, 22)

rep_df19['isValid'].value_counts(dropna=False)
#True     17478
#False       15

rep_df19['removed'].value_counts(dropna=False)
# False    16120
# True      1373

rep_df19 = rep_df19.loc[rep_df19['removed'] == 0]
rep_df19.shape
# (16120, 19)

rep_df19 = rep_df19.loc[rep_df19['isValid'] == 1]
rep_df19.shape
# (16113, 19)

# writing out reps data to a csv
rep_df17.to_csv('/home/yulia/Documents/SNS/Data/rep_data_w207_old17.csv')
rep_df19.to_csv('/home/yulia/Documents/SNS/Data/rep_data_w207_old19.csv')

#%%
def ob_json_to_csv(json_path):
    """
    Converts the raw, openbarbell json data into a csv file. Performs basic cleanup and
    transformations, but keeps the data as close to raw as possible.
    
    Args:
        json_path (string): the path to the raw json data file
        
    Returns:
        None
    """
    
    # Read json into a dataframe
    data_df = pd.read_json(json_path, lines = True)
    
    # Filter dataframe to rows that are not flagged as removed
    data_df = data_df.loc[data_df['removed'] == 0]
    
    # Filter dataframe to rows where deleted is null or flagged as 0
    data_df = data_df.loc[(data_df['deleted'].isnull()) | (data_df['deleted'] == 0)]
    
    # Filter dataframe to rows that have an initialStartTime value
    data_df = data_df.loc[pd.notna(data_df['initialStartTime'])]
    
    # Extract the the reps data, transform it into its own dataframe
    rep_df = pd.DataFrame(data_df['reps'].values.tolist())
    rep_col_mapping = {
        	0:	'StartMessg'
        	,1:	'RepN'
        	,2:	'AvgVel'
        	,3:	'ROM'
        	,4:	'PeakVel'
        	,5:	'PeakVelLoc'
        	,6:	'PeakAccel'
        	,7:	'RepDur'
        	,8:	'TimeBWReps'
        	,9:	'TimeRepComp'
        	,10:	'TimeRepWait'
        	,11:	'SlowAllow'
        	,12:	'Backlight'
        	,13:	'MinAllow'
        	,14:	'ComprEnable'
        	,15:	'FiltrEnable'
        	,16:	'CodeV'
        	,17:	'UnitN'
        	,18:	'LED'
        	,19:	'Bright'
        	,20:	'LowPow'
        	,21:	'BulkStart'
    }
    rep_df = rep_df.rename(columns=rep_col_mapping)
    
    # Append it to the existing dataframe
    data_df = pd.concat([data_df, rep_df], axis=1)
     
    # Writing out data to a csv
    data_df.to_csv('./ob_data_w207_raw.csv')
    
    
def csv_filter_columns(csv_path):
    """
    Reads in a csv of transformed openbarbell data and filters it down to
    specific columns
    
    Args:
        csv_path (string): path the the source csv file
        
    Returns:
        None
    """
    
    # Specify the columns that should be kept in our data
    cols_to_keep = [
        'setID'
        ,'RepCount'
        ,'isValid'
        ,'removed'
        ,'hardware'
        ,'appVersion'
        ,'deviceName'
        ,'deviceIdentifier'
        ,'time'
        ,'exercise'
        ,'StartMessg'
        ,'RepN'
        ,'AvgVel'
        ,'ROM'
        ,'PeakVel'
        ,'PeakVelLoc'
        ,'PeakAccel'
        ,'RepDur'
        ,'TimeBWReps'
        ,'TimeRepComp'
        ,'TimeRepWait'
        ,'SlowAllow'
        ,'Backlight'
    ]
    
    # Read in the source csv file, filtering it to only the columns
    # specified
    data_df = pd.read_csv(csv_path, usecols=cols_to_keep)
    
    # Write out the filtered data to a new csv
    data_df.to_csv('./ob_data_w207_filtered.csv')
    
    
def add_labels(csv_path):
    """
    Reads in a csv file, ideally transformed and filtered, and applies 
    labeling logic to the dataset. The native exercise labels are unreliable,
    so custom logic is required to get a clearer sense around the exercise being
    performed
    
    Args:
        csv_path (string): path to the source csv file
        
    Returns:
        None
    """
    
    # Read csv file into dataframe
    data_df = pd.read_csv(csv_path)
    
    #  Apply Tim's logic for simple labeling
    data_df['exercise'] = data_df['exercise'].str.lower().str.strip()
    data_df['simple_label'] = data_df['exercise'].apply(ex_name)
    
    # Write labeled dataframe to a csv
    data_df.to_csv('./ob_data_w207_labeled.csv')

 
def ex_name(exercise):
    """
    Helper function that looks for specific keywords in the provided
    'exercise' argument to help identify which lift is being performed.
    
    Args:
        exercise (string): the original exercise string
        
    Returns:
        string: the clean exercise string
    """
    
    if "bench" in exercise.lower():
        return "bench"
    elif "dead" in exercise.lower():
        return "dead lift"
    elif "squat" in exercise.lower():
        return "squat"
    else:
        return exercise.lower()


#%%
def partition_data(csv_path, train_data, train_labels, test_data, test_labels):
    """
    docstring
    
    Args:
        
    Returns:
        
    """

    print('foo')
    
