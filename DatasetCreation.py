import numpy as np

# Initialise indexes lists
fileIndex = ["%02d" % x for x in range(1,61)]
personIndex = ["%01d" % x for x in range(1,9)]
activityIndex = ["%02d" % x for x in range(1,20)]

datasetHeader = ["T_xacc", "T_yacc", "T_zacc", "T_xgyro", "T_ygyro", "T_zgyro", "T_xmag", "T_ymag", "T_zmag",
        "RA_xacc", "RA_yacc", "RA_zacc", "RA_xgyro", "RA_ygyro", "RA_zgyro", "RA_xmag", "RA_ymag", "RA_zmag",
        "LA_xacc", "LA_yacc", "LA_zacc", "LA_xgyro", "LA_ygyro", "LA_zgyro", "LA_xmag", "LA_ymag", "LA_zmag",
        "RL_xacc", "RL_yacc", "RL_zacc", "RL_xgyro", "RL_ygyro", "RL_zgyro", "RL_xmag", "RL_ymag", "RL_zmag",
        "LL_xacc", "LL_yacc", "LL_zacc", "LL_xgyro", "LL_ygyro", "LL_zgyro", "LL_xmag", "LL_ymag", "LL_zmag",
        "var_T_xacc", "var_T_yacc", "var_T_zacc", "var_T_xgyro", "var_T_ygyro", "var_T_zgyro", "var_T_xmag", "var_T_ymag", "var_T_zmag",
        "var_RA_xacc", "var_RA_yacc", "var_RA_zacc", "var_RA_xgyro", "var_RA_ygyro", "var_RA_zgyro", "var_RA_xmag", "var_RA_ymag", "var_RA_zmag",
        "var_LA_xacc", "var_LA_yacc", "var_LA_zacc", "var_LA_xgyro", "var_LA_ygyro", "var_LA_zgyro", "var_LA_xmag", "var_LA_ymag", "var_LA_zmag",
        "var_RL_xacc", "var_RL_yacc", "var_RL_zacc", "var_RL_xgyro", "var_RL_ygyro", "var_RL_zgyro", "var_RL_xmag", "var_RL_ymag", "var_RL_zmag",
        "var_LL_xacc", "var_LL_yacc", "var_LL_zacc", "var_LL_xgyro", "var_LL_ygyro", "var_LL_zgyro", "var_LL_xmag", "var_LL_ymag", "var_LL_zmag",
        "activity_index"]

# Create the train dataset: 7980 x 91
# Person from 1 to 7
trainingActivities = []
print("Importing training data")
for k in range(19):
    print("Elaborating activity number: ", activityIndex[k])
    for j in range(7):
        print("Elaborating person number: ", personIndex[j], end = "\r")
        for i in range(60):
            filename = f"./sportsDataset/a{activityIndex[k]}/p{personIndex[j]}/s{fileIndex[i]}.txt"

            data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=float)
            dataT = data.transpose()            
            average = np.mean(dataT, axis = 1)
            variance = np.var(dataT, axis = 1)
            
            index = np.array([int(activityIndex[k])])
            #newData = np.append(average,variance, index, axis = 0)
            newData = np.concatenate((average, variance, index), axis = None)
            
            trainingActivities.append(newData)

print("\nTraining data correctly stored")

activitiesTrainDataset = np.array(trainingActivities)
np.savetxt("./sportsDataset/TrainingDataset.csv", activitiesTrainDataset, delimiter=",", header = ','.join(datasetHeader), comments='')
print("Training dataset created\n")


# Create the test dataset: 1140 x 91
# Person 8
testingActivities = []
print("Importing testing data")
for k in range(19):
    print("Elaborating activity number: ", activityIndex[k])
    print("Elaborating person number: ", personIndex[7], end = "\r")
    for i in range(60):
        filename = './sportsDataset/a' + activityIndex[k] + '/p'+ personIndex[7] + '/s' + fileIndex[i] + '.txt'

        data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=float)
        dataT = data.transpose()            
        average = np.mean(dataT, axis = 1)
        variance = np.var(dataT, axis = 1)
        
        index = np.array([int(activityIndex[k])])
        #newData = np.append(average, variance, index, axis = 0)
        newData = np.concatenate((average, variance, index), axis = None)
        
        testingActivities.append(newData)

print("\nTesting data correctly stored")

activitiesTestDataset = np.array(testingActivities)
np.savetxt("./sportsDataset/TestDataset.csv", activitiesTestDataset, delimiter=",", header = ','.join(datasetHeader), comments='')
print("Testing dataset created\n")

# Create the entire dataset: 9120 x 91
# Person from 1 to 8
allActivities = []
print("Importing all data")
for k in range(19):
    print("Elaborating activity number: ", activityIndex[k])
    for j in range(8):
        print("Elaborating person number: ", personIndex[j], end = "\r")
        for i in range(60):
            filename = f"./sportsDataset/a{activityIndex[k]}/p{personIndex[j]}/s{fileIndex[i]}.txt"

            data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=float)
            dataT = data.transpose()            
            average = np.mean(dataT, axis = 1)
            variance = np.var(dataT, axis = 1)
            
            index = np.array([int(activityIndex[k])])
            #newData = np.append(average,variance, index, axis = 0)
            newData = np.concatenate((average, variance, index), axis = None)
            
            allActivities.append(newData)

print("\nData correctly stored")

ActivitiesDataset = np.array(allActivities)
np.savetxt("./sportsDataset/ActivitiesDataset.csv", ActivitiesDataset, delimiter=",", header = ','.join(datasetHeader), comments='')
print("Activity dataset created\n")
