# Comparison of common Machine Learning Algorithms considering a sport activities Classification Task

*Group project for Machine Learning class of Galli Davide and Papousek Jiri.*

## Introduction to the Project ##
Even if Deep Learning is continuosly growing in terms of importance, classical Machine Learning technquis still represent a cornerstone for people who approach computer science for the first time. There are loads of different techniques, some of them are specifically devoted to tackle given types of problems; some others are instead multi-porpose.

One popular task is the so called *classification problem*, where the model’s output is a category with a semantic meaning. A classification model attempts to draw some conclusion from observed values.
Different methods can be implemented to tackle this problem.

Our focus is a brief comparative study over four different machine learning supervised techniques:
1. Logistic Regression
2. K Nearest Neighbors
3. Decision Trees
4. Multilayer Perceptron

## Brief Description of the Dataset ##

The choosen [dataset](https://archive-beta.ics.uci.edu/ml/datasets/daily+and+sports+activities) comprises motion sensor data of 19 daily and sports activities performed by 8 subjects (between 20 and 30 years old) *in their own style* for 5 minutes. Five Xsens MTx units are used on the torso, arms, and legs. This kind of sensor embeds a gyroscope, an accelerometer and a magnetometer.

Since activities are perfomed as the subject desires, there might be inter-subject variations in the speeds and amplitudes of some activities.

The activities are performed at the Bilkent University Sports Hall, in the Electrical and Electronics Engineering Building, and in a flat outdoor area on campus. Sensor units are calibrated to acquire data at 25 Hz sampling frequency. The 5-min signals are divided into 5-sec segments so that 480(=60x8) signal segments are obtained for each activity.

The 19 activities are:

 A1 &rarr; sitting

 A2 &rarr; standing

 A3 &rarr; lying on back

 A4 &rarr; laying on right side

 A5 &rarr; ascending stairs

 A6 &rarr; descending stairs

 A7 &rarr; standing in an elevator still

 A8 &rarr; moving around in an elevator

 A9 &rarr; walking in a parking lot
 
 A10 &rarr; walking on a treadmill with a speed of 4 km/h on flat
 
 A11 &rarr; walking on a treadmill with a speed of 4 km/h on a 15 deg inclined ositions
 
 A12 &rarr; running on a treadmill with a speed of 8 km/h
 
 A13 &rarr; exercising on a stepper
 
 A14 &rarr; exercising on a cross trainer
 
 A15 &rarr; cycling on an exercise bike in horizontal position
 
 A16 &rarr; cycling on an exercise bike in vertical positions
 
 A17 &rarr; rowing
 
 A18 &rarr; jumping
 
 A19 &rarr; playing basketball
 
 ## File structure ##
 
 19 activities (a) (in the order given above) 8 subjects (p) 60 segments (s) 5 units:
 - on torso (T)
 - right arm (RA)
 - left arm (LA)
 - right leg (RL)
 - left leg (LL)
 
 There are 9 sensors on each unit (x,y,z accelerometers, x,y,z gyroscopes, x,y,z magnetometers).
 
 Folders a01, a02, ..., a19 contain data recorded from the 19 activities. For each activity, the subfolders p1, p2, ..., p8 contain data from each of the 8 subjects. In each subfolder, there are 60 text files s01, s02, ..., s60, one for each segment. In each text file, there are 5 units x 9 sensors = 45 columns and 5 sec x 25 Hz = 125 rows. Each column contains the 125 samples of data acquired from one of the sensors of one of the units over a period of 5 sec. Each row contains data acquired from all of the 45 sensor axes at a particular sampling instant separated by commas.
 
 Columns 1-45 correspond to:
  T_xacc, T_yacc, T_zacc, T_xgyro, ..., T_ymag, T_zmag, RA_xacc, RA_yacc, RA_zacc, RA_xgyro, ..., RA_ymag, RA_zmag, LA_xacc, LA_yacc, LA_zacc, LA_xgyro, ..., LA_ymag, LA_zmag, RL_xacc, RL_yacc, RL_zacc, RL_xgyro, ..., RL_ymag, RL_zmag, LL_xacc, LL_yacc, LL_zacc, LL_xgyro, ..., LL_ymag, LL_zmag.
 
 Therefore, columns 1-9 correspond to the sensors in unit 1 (T), columns 10-18 correspond to the sensors in unit 2 (RA), columns 19-27 correspond to the sensors in unit 3 (LA), columns 28-36 correspond to the sensors in unit 4 (RL), columns 37-45 correspond to the sensors in unit 5 (LL). 

 ## References ##
1. [Comparative Study on Classic Machine learning Algorithms](https://towardsdatascience.com/comparative-study-on-classic-machine-learning-algorithms-24f9ff6ab222)
2. [Daily and Sports Activities Dataset](https://archive-beta.ics.uci.edu/ml/datasets/daily+and+sports+activities)
3. [Scikit Learn Python Library](https://scikit-learn.org/stable/)
4. [Keras Python Library](https://keras.io/)
5. [Project GitHub Repository](https://github.com/d-galli/SportActivitiesClassification)
