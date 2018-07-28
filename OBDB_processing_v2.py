import pandas as pd
import numpy as np

def read_ob_json(json_path):
    """
    Reads raw json file and filters out:
        - sets that were removed by user 
        - sets that were not flagged as squat/bench/deadlift
    
    Args:
        json_path (string): the path to the raw json data file
        
    Returns:
        pandas dataframe
    """
    
    # Read json into a dataframe
    data_df = pd.read_json(json_path, lines = True)
    print(f"{data_df.shape[0]} rows in the raw dataset")
    
    # Filter dataframe to rows that are not flagged as removed
    data_df = data_df.loc[data_df['removed'] == 0]
    print(f"{data_df.shape[0]} rows after removing rows with 'removed' flag")
    
    # Filter dataframe to rows where deleted is null or flagged as 0
    data_df = data_df.loc[(data_df['deleted'].isnull()) | (data_df['deleted'] == 0)]
    print(f"{data_df.shape[0]} rows after removing rows with 'deleted' flag")
    
    # Filter on the rows that are at least somehow related to the big 3 lifts
    # bp is short for bench press
    big3 = ["squat", "bench", "bp", "deadlift"]
    exercises = data_df["exercise"].str.lower().str.strip().tolist()
    keep_flag = np.zeros(data_df.shape[0])    
    for index in range(len(exercises)):
        if type(exercises[index]) == str and any(word in exercises[index] for word in big3):
            keep_flag[index] = 1
    
    print("Number of rows with big 3 exercise:", sum(keep_flag))
    data_df = data_df.loc[keep_flag == 1]
    print(f"{data_df.shape[0]} sets with big 3 exercises")
        
    return data_df
    
def get_sets_df(data_df):
    """
    Removes 'reps' column from input dataframe to build set-specific dataframe
    
    Args:
        data_df: pandas dataframe (output from read_ob_json)
        
    Returns:
        pandas dataframe 
    """
    
    sets_df = data_df.drop('reps', axis = 1)
    print(f"Set data frame dimensions: {sets_df.shape}")
    print("Column names are", list(sets_df))
    
    return sets_df

def get_reps_df(data_df, colnames, appver):
    """
    Unrolls `reps` field from the main dataframe and extracts
    `data` list based on each OB version specifics
    
    Args:
        data_df:  pandas dataframe (output from read_ob_json)
        colnames: list of fields to keep from reps dictionary
        appver:   OB app version
        
    Returns:
        reps data frame specific to each app version
    """
    
    # setID is the primary key to merge sets and reps data
    sets_list = data_df['setID'].tolist()
    # `reps` field is a nested dictionary
    reps_list = data_df['reps'].tolist()

    # creating a dictionary for all sets and reps
    # primary key: setID; secondary key: rep count
    # values: rep values and rep['data'] values
    set_dict = {}
    for set_index in range(data_df.shape[0]):
        set_dict[sets_list[set_index]] = {}
        for n, rep in enumerate(reps_list[set_index]):
            # each OB version had different metrics captured by `data` list
            if len(rep.get('data')) > len(colnames):
                set_dict[sets_list[set_index]][n] = rep.copy()
                for col_index in range(len(colnames)):
                    set_dict[sets_list[set_index]][n][colnames[col_index]] = rep.get('data')[col_index]
                del set_dict[sets_list[set_index]][n]['data']      
                
    print(f"{len(set_dict)} sets recorded")
    
    # unroll nested set_dict into pandas dataframe
    reps_df = pd.DataFrame.from_dict({(i,j): set_dict[i][j] 
                                        for i in set_dict.keys() 
                                        for j in set_dict[i].keys()},
                                   orient='index')
    # name dataframe indices
    reps_df.index.names = ["setID", "RepCount"]
    # convert indices into columns
    reps_df = reps_df.reset_index(level=["setID", "RepCount"])
    
    # keep only rows specific to each OB app version
    reps_df = reps_df.loc[(reps_df['StartMessg'] == appver[0]) | (reps_df['StartMessg'] == appver[1])]
    print(f"{reps_df.shape[0]} reps extracted")
    
    # remove reps deleted by a user
    reps_df = reps_df.loc[reps_df['removed'] == 0]
    print(f"{reps_df.shape[0]} rows after removing rows with 'removed' flag")
    
    # remove reps marked by the app as invalid
    reps_df = reps_df.loc[reps_df['isValid'] == 1]
    print(f"{reps_df.shape[0]} rows after removing rows with 'deleted' flag")
    
    print("Column names are", list(reps_df))

    return reps_df

def combine_reps_dfs(data_df):
    """
    Combines rep dataframes from three different versions of OB database
    
    Args:
        data_df: pandas dataframe (output from read_ob_json) 
        
    Returns:
        reps dataframe
    """
    
    # data metrics captured by OB v1
    colnames_obv1 = [
            'StartMessg'
            ,'RepN'
            ,'AvgVel'
            ,'ROM'
            ,'PeakVel'
            ]
    
    # data metrics captured by OB v2
    colnames_obv2 = [
            'StartMessg'
            ,'RepN'
            ,'AvgVel'
            ,'ROM'
            ,'PeakVel'
            ,'PeakVelLoc'
            ,'StartData'
            ,'RepDur'
            ,'TimeBWReps'
            ,'TimeRepComp'
            ,'TimeRepWait'
            ,'SlowAllow'
            ,'Backlight'
            ,'MinAllow'
            ]
    
    # data metrics captured by OB v3
    colnames_obv3 = [
            'StartMessg'
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
            ,'MinAllow'
            ]
    
    print("\nProcessing OBV1 reps\n")
    # 1234 is the flag used to identify v1
    obv1_df = get_reps_df(data_df, colnames_obv1, ('-1234', '-1234.0'))
    print("\nProcessing OBV2 reps\n")
    # 2345 is the flag used to identify v2
    obv2_df = get_reps_df(data_df, colnames_obv2, ('-2345', '-2345.0'))
    print("\nProcessing OBV3 reps\n")
    # 3456 is the flag used to identify v3
    obv3_df = get_reps_df(data_df, colnames_obv3, ('-3456', '-3456.0'))

    rep_check = obv1_df.shape[0]+obv2_df.shape[0]+obv3_df.shape[0]
    print("\nTotal reps across 3 datasets:", rep_check)
    reps_all = pd.concat([obv1_df, obv2_df, obv3_df], sort=False)
    print("Dimensions of the final dataset:", reps_all.shape)
    
    return reps_all

def convert_columns(data_df):
    """
    Converts certain fields from strings to datetime and numeric objects
    
    Args:
        pandas dataframe with all the reps combined
        
    Returns:
        reps dataframe with properly typed fields
    """
    
    # datatime fields
    date_cols = [
            'endTime'
            ,'startTime'
            ,'initialStartTime'
            ,'time'
            ]
    
    # numeric fields
    num_cols = [
            'RepN'
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
            ,'MinAllow'
            ]
    
    data_df[date_cols] = data_df[date_cols].apply(pd.to_datetime, errors = 'coerce')
    data_df[num_cols] = data_df[num_cols].apply(pd.to_numeric, errors = 'coerce')
    
    # some of the velocity fields have 'infinity' values
    # replacing them with nans
    data_df=data_df.replace(np.inf, np.nan)
    
    return data_df

def fix_rpe_weight(data_df):
    """
    Fixes rpe and weight fields:
        both rpe and weight are catptured as strings in the database
        to allow decimal point and commas (for European lifters)
    
    Args:
        data_df: reps dataframe
    
    Returns:
        reps dataframe with new fields for rpe and weight
    """

    # Converting RPE (rate of perceived exertion) into float
    rpe = data_df['rpe'].tolist()
    
    # converting RPE into numbers and storing them in a list
    # capturing RPE error entries in a separate list for QA
    rpe_num, rpe_errors = [], []
    for r in rpe:
        # values less than 5.5 are not permitted in the database
        # because this low rpe is usually inaccurate
        # hence they are captured by '< 5.5' value
        if r in ('< 5.5', '< 5,5'):
            r_num = 5.5
        # capturing empty entries
        elif r in ('','..'):
            r_num = np.nan
        # replacing decimal comma with a period to convert to a float
        # checking for the length of 3 because the only valid values here
        # would be '6,5', '7,5', '8,5', '9,5'
        elif type(r) == str and len(r) == 3 and r.find(',') > 0:
            r_num = float(r.replace(',','.'))
        # this is to capture values separated by '-' and containing commas
        elif type(r) == str and r.find(',') > 0:
            r2 = r.replace(",",".")
            if r2.find("-") > 0:
                # take the average of multiple rpes separated by '-'
                r_num = np.mean([float(n) for n in r2.split("-")])
        # separate other multiple rpes
        elif type(r) == str and r.find('-') > 0:
            r_num = np.mean([float(n) for n in r.split("-")])
        elif type(r) == str:
            # try converting to float all remaining values
            try:
                r_num = float(r)
            except ValueError:
                # if this is not working, check what those values are
                r_num = np.nan
                rpe_errors.append(r)
        elif type(r) in (float, int):
            r_num = r
        else:
            r_num = np.nan
        rpe_num.append(r_num)
    
    # setting all values between 5.5 and 10
    for index in range(len(rpe_num)):
        if rpe_num[index] < 5.5: 
            rpe_num[index] = np.nan
        elif rpe_num[index] > 10: 
            rpe_num[index] = 10
    
    print("="*50)
    print("Results of processing 'rpe'\n")
    print(f"Number of values to be replaced: {len(rpe)}")
    print(f"Number of invalid values in 'rpe' list: {len(rpe_errors)}")
    print(f"Number of valid values in 'rpe_num' list: {sum(~np.isnan(rpe_num))}")
    print(f"Number of missing in 'rpe_num' list: {sum(np.isnan(rpe_num))}")
    print("Invalid rpe entries:",set(rpe_errors))
    
    data_df = data_df.assign(rpe_num = rpe_num)
    print(f"Average rpe excluding missing values: {np.mean(data_df['rpe_num'])}")
    
    
    # Converting Weight into float
    weight = data_df['weight'].tolist()
    # weight values can be lbs or kgs depending on 'metric' value
    metric = data_df['metric'].tolist()
    
    # converting weight into numbers and storing them in a list
    # capturing weight error entries in a separate list for QA
    weight_num = []
    weight_errors = []
    for w in weight:
        # replace decimal comma with point to get a float
        if type(w) == str and w.count(',') == 1:
            w_num = float(w.replace(',','.'))
        # sometimes comma is used to separate multiple weight values
        elif type(w) == str and w.count(',') > 1:
            w_num = np.mean([float(n) for n in w.split(",")])
        # sometimes '-' is used to separate multiple weight values        
        elif type(w) == str and w.find('-') > 0:
            w_num = np.mean([float(n) for n in w.split("-")])
        # sometimes '.' is used to separate multiple weight values        
        elif type(w) == str and w.count('.') > 1:
            # taking an average of weight values separated by '.'
            # in case the weight has half a unit (.5), ignore it
            # this is not accurate but no other way to differentiate
            w_num = np.mean([float(n) for n in w.split(".") 
                             if (n != '') and (n != '5')])
        else:
            try:
                # try converting to float all remaining values
                w_num = float(w)
            except (ValueError, TypeError):
                # if this is not working, check what those values are
                w_num = np.nan
                weight_errors.append(w)
        weight_num.append(w_num)
        
    # convert all weight values to lbs
    weight_lbs = weight_num[:]
    for index in range(len(weight_num)):
        if metric[index] == 'kgs':
            weight_lbs[index] = weight_num[index]*2.20462
        # cap all values at 1000
        # potentially exclude them from the analysis
        if weight_lbs[index] > 1000:
            weight_lbs[index] = 1000

    print("="*50)
    print("Results of processing 'weight'\n")

    print("Total mean of 'weight_num':", np.nanmean(weight_num))
    print("Max of 'weight_num':", np.nanmax(weight_num))
    print("Mean of weight in lbs:", np.nanmean(weight_lbs))
    print("Max of weight in lbs:", np.nanmax(weight_lbs))

    print(f"Number of values to be replaced: {len(weight)}")
    print(f"Number of invalid values in 'weight' list: {len(weight_errors)}")
    print(f"Number of valid values in 'weight_num' list: {sum(~np.isnan(weight_num))}")
    print(f"Number of missing in 'weight_num' list: {sum(np.isnan(weight_num))}")
    print("Invalid rpe entries:",set(weight_errors))

    data_df = data_df.assign(weight_lbs = weight_lbs)
    print(f"Average weight_lbs excluding missing values: {np.mean(data_df['weight_lbs'])}")
    
    return data_df

def remove_outliers(data_df):
    
    outliers = np.zeros(data_df.shape[0])
    AvgVel = data_df['AvgVel'].tolist()
    ROM = data_df['ROM'].tolist()
    PeakAccel = data_df['PeakAccel'].tolist()
    PeakVel = data_df['PeakVel'].tolist()
    PeakVelLoc = data_df['PeakVelLoc'].tolist()
    for index in range(data_df.shape[0]):
        if AvgVel[index] > 3:
            outliers[index] = 1
        elif ROM[index] > 2000:
            outliers[index] = 1
        elif PeakAccel[index] > 3000:
            outliers[index] = 1
        elif PeakVel[index] > 10:
            outliers[index] = 1
        elif PeakVelLoc[index] > 100:
            outliers[index] = 1

    print(f"{sum(outliers)} rows with outliers will are removed")    
    data_df = data_df.loc[outliers == 0]
    print(f"Dataframe dimensions is {data_df.shape}")
    
    return data_df

def add_labels(data_df):
    
    bench_clean = ['bench', 'bench press', 'bp', 'competition bench', 'comp bench']
    squat_clean = ['squat', 'back squat', 'competition squat', 'comp squat']
    deadlift_clean = [ "deadlift", "sumo deadlift", "conventional deadlift",
                      "comp deadlift", "competition deadlift"]
    exercises_dirty = data_df["exercise"].str.lower().str.strip().tolist()
    exercises_clean = ['']*data_df.shape[0]
    
    for index in range(data_df.shape[0]):
        if exercises_dirty[index] in bench_clean:
            exercises_clean[index] = "bench"
        elif exercises_dirty[index] in squat_clean:
            exercises_clean[index] = "squat"
        elif exercises_dirty[index] in deadlift_clean:
            exercises_clean[index] = "deadlift"
        elif "squat" in exercises_dirty[index]:
            exercises_clean[index] = "squat other"
        elif "bench" in exercises_dirty[index]:
            exercises_clean[index] = "bench other"
        elif "bp" in exercises_dirty[index]:
            exercises_clean[index] = "bench other"
        elif "deadlift" in exercises_dirty[index]:
            exercises_clean[index] = "deadlift other"
        else:
            exercises_clean[index] = ""
    
    print("Labels version1 counts:")
    ex, c = np.unique(exercises_clean, return_counts = True)
    for k in range(ex.shape[0]):
        print(ex[k], c[k])    
        
    tags = data_df['tags'].tolist()
    bench_tags, squat_tags, deadlift_tags = [], [], []
    for index in range(len(exercises_clean)):
        if exercises_clean[index] == "bench":
            bench_tags.append(tags[index])
        elif exercises_clean[index] == "squat":
            squat_tags.append(tags[index])
        elif exercises_clean[index] == "deadlift":
            deadlift_tags.append(tags[index])
            
    def top_list(mylist, top=20):
        for index in range(len(mylist)):
            if type(mylist[index]) != str:
                mylist[index] = str(mylist[index])
                
        member, count = np.unique(mylist, return_counts = True)
        for m, c in zip(member[np.argsort(-count)][:top],count[np.argsort(-count)][:top]):
            print(m, c)
    
#    print("\nBench tags:")
#    top_list(bench_tags)
#    print("\nSquat tags:")
#    top_list(squat_tags)
#    print("\nDeadlift tags:")
#    top_list(deadlift_tags)
    
    valid_tags = ['warm', 'belt', 'sleeve', 'single', 'workset', 'competition']
    clean_tags_flag = [0]*len(tags)
    for index in range(len(tags)):
        if type(tags[index]) == float:
            clean_tags_flag[index] = 1
        elif len(tags[index]) == 0:
            clean_tags_flag[index] = 1
        elif len(tags[index]) == 1:
            if any(word in tags[index][0].lower().strip() for word in valid_tags):
                clean_tags_flag[index] = 1
        elif len(tags[index]) == 2:
            clean_tags = ''.join(tags[index]).lower().strip()
            if any(word in clean_tags for word in valid_tags):
                clean_tags_flag[index] = 1
    
    exercises_clean2 = exercises_clean.copy()  
    for index in range(len(exercises_clean)):
        if clean_tags_flag[index] == 0:
            if exercises_clean[index] == "bench":
                exercises_clean2[index] = "bench other"
            elif exercises_clean[index] == "squat":
                exercises_clean2[index] = "squat other"
            elif exercises_clean[index] == "deadlift":
                exercises_clean2[index] = "deadlift other"

    print("Labels version2 counts:")
    ex, c = np.unique(exercises_clean2, return_counts = True)
    for k in range(ex.shape[0]):
        print(ex[k], c[k])    
    
#    check = []
#    for index in range(len(exercises_clean)):
#        if exercises_clean[index] == "squat" and exercises_clean2[index] != "squat":
#            check.append(tags[index])
#    top_list(check)
        
    data_df = data_df.assign(exercise_clean = exercises_clean)
    data_df = data_df.assign(exercise_clean2 = exercises_clean2)
    
    return data_df


def get_desriptives(var):
    
    descr_summary = {}
    descr_summary['min'] = round(np.min(var),3)
    descr_summary['per25'] = round(np.nanpercentile(var, 25),3)
    descr_summary['per50'] = round(np.nanpercentile(var, 50),3)
    descr_summary['mean'] = round(np.mean(var),3)
    descr_summary['per75'] = round(np.nanpercentile(var, 75),3)
    descr_summary['per99'] = round(np.nanpercentile(var, 99),3)
    descr_summary['per999'] = round(np.nanpercentile(var, 99.9),3)
    descr_summary['max'] = round(np.max(var),3)
    descr_summary['std'] = round(np.std(var),3)
    descr_summary['missing'] = sum(np.isnan(var))
    
    print(f"\nDescriptive summary for {var.name}:")
    for stat in descr_summary:
        print(stat, descr_summary[stat])
    

json_path = '/home/yulia/Documents/SNS/Data/obdatadump.json'
data_df = read_ob_json(json_path)
sets_df = get_sets_df(data_df)
reps_df = combine_reps_dfs(data_df)
full_data = sets_df.merge(reps_df, on = "setID", how = "left", 
                          suffixes=('_set', '_rep'), indicator = True)
full_data.shape
list(full_data)
np.unique(full_data['_merge'], return_counts=True)
full_data_clean = full_data.loc[full_data['_merge'] == 'both']
full_data_clean.shape

full_data_clean = convert_columns(full_data_clean)
full_data_clean = fix_rpe_weight(full_data_clean)
full_data_clean = remove_outliers(full_data_clean)
full_data_clean = add_labels(full_data_clean)
full_data_clean = full_data_clean.set_index("_id")

full_data_clean.to_csv('./ob_data_w207_filtered.csv')
#full_data_clean.to_json('./ob_data_w207_filtered.json', orient='records')


#%%

for col in list(full_data_clean):
    print(col, full_data_clean[col].dtype)

num_cols = [
        'RepN'
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
        ,'MinAllow'
        ,'rpe_num'
        ,'weight_lbs'
        ]

full_data_clean[num_cols].apply(get_desriptives)

exercises, count = np.unique(full_data_clean["exercise"].str.lower().str.strip(), return_counts = True)
top = 50
for ex, c in zip(exercises[np.argsort(-count)][:top],count[np.argsort(-count)][:top]):
    print(ex, c)