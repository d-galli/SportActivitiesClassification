import numpy as np

# Initialise indexes lists
file_index = ["%02d" % x for x in range(1,61)]
person_index = ["%01d" % x for x in range(1,9)]
activity_index = ["%02d" % x for x in range(1,20)]

dataset_header = ["T_xacc", "T_yacc", "T_zacc", "T_xgyro", "T_ygyro", "T_zgyro", "T_xmag", "T_ymag", "T_zmag",
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
training_activities = []
print("Importing training data")
for k in range(19):
    print("Elaborating activity number: ", activity_index[k])
    for j in range(7):
        print("Elaborating person number: ", person_index[j], end = "\r")
        for i in range(60):
            filename = f"./sports_dataset/a{activity_index[k]}/p{person_index[j]}/s{file_index[i]}.txt"

            data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=float)
            dataT = data.transpose()            
            average = np.mean(dataT, axis = 1)
            variance = np.var(dataT, axis = 1)
            
            index = np.array([int(activity_index[k])])
            #newData = np.append(average,variance, index, axis = 0)
            newData = np.concatenate((average, variance, index), axis = None)
            
            training_activities.append(newData)

print("\nTraining data correctly stored")

activities_train_dataset = np.array(training_activities)
np.savetxt("./sports_dataset/training_dataset.csv", activities_train_dataset, delimiter=",", header = ','.join(dataset_header), comments='')
print("Training dataset created\n")


# Create the test dataset: 1140 x 91
# Person 8
testing_activities = []
print("Importing testing data")
for k in range(19):
    print("Elaborating activity number: ", activity_index[k])
    print("Elaborating person number: ", person_index[7], end = "\r")
    for i in range(60):
        filename = f"./sports_dataset/a{activity_index[k]}/p{person_index[7]}/s{file_index[i]}.txt"

        data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=float)
        data_t = data.transpose()
        average = np.mean(data_t, axis = 1)
        variance = np.var(data_t, axis = 1)
        
        index = np.array([int(activity_index[k])])
        #new_data = np.append(average, variance, index, axis = 0)
        new_data = np.concatenate((average, variance, index), axis = None)
        
        testing_activities.append(new_data)

print("\nTesting data correctly stored")

activities_test_dataset = np.array(testing_activities)
np.savetxt("./sports_dataset/test_dataset.csv",
           activities_test_dataset,
           delimiter=",", header = ','.join(dataset_header), comments='')
print("Testing dataset created\n")

# Create the entire dataset: 9120 x 91
# Person from 1 to 8
all_activities = []
print("Importing all data")
for k in range(19):
    print("Elaborating activity number: ", activity_index[k])
    for j in range(8):
        print("Elaborating person number: ", person_index[j], end = "\r")
        for i in range(60):
            filename = f"./sports_dataset/a{activity_index[k]}/p{person_index[j]}/s{file_index[i]}.txt"

            data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=float)
            data_t = data.transpose()
            average = np.mean(data_t, axis = 1)
            variance = np.var(data_t, axis = 1)
            
            index = np.array([int(activity_index[k])])
            #new_data = np.append(average,variance, index, axis = 0)
            new_data = np.concatenate((average, variance, index), axis = None)
            
            all_activities.append(new_data)

print("\nData correctly stored")

activities_dataset = np.array(all_activities)
np.savetxt("./sports_dataset/activities_dataset.csv", activities_dataset,
           delimiter=",", header = ','.join(dataset_header), comments='')
print("Activity dataset created\n")
