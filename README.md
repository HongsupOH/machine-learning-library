This is a machine learning library developed by Hongsup Oh for CS5350/6350 in University of Utah
# 1. Decision Tree
Please move to Experiment folder, and implement run.sh
## 1.1 Input variables
Sample data (S): It is represented by 2D list/array <br />
label: It is the label of each line of sample. It is represented by 1D list/array <br />
Attributes : It is represented by dictionaly. Key is attribute and it value is values of attribute <br />
Tree: Result tree is saved. It is dictionaly <br />
maxDepth: Tree's maximum depth limit<br />
cur depth: The depth of current depth. The root node is depth 1<br />
parent: Parent node of current node<br />
link: The linked edge between current node and parent node<br />
majority: Majority trend of parent node<br />
## 1.2 Generate Tree
At IG3.py, IG3 function is main function to generate Tree. Tree is dictionaly. Each path from root to current node is used as key of the Tree.
## 1.3 Evaluate data
At IG3.py, predict function helps evaluate data. Generated Tree, test sample, test label and attribute are required
## 1.4 Measure error
At IG3.py, error function works to measure error. Predict answer and label are required

# 2. Ensemble Learning
Please move to Experiment folder, and implement run.sh
## 2.1 AdaBoost
Stumps are generated t times. Current stump affect the weight of the next stump. Based on the new weight, Information gain and majority are maesured. After building t stumps, test data is evaluated with votes. 
## 2.2 Bagging
First, m' size random samples are selected with replacement. Full growth trees are generated t times with random sample.Based on t trees, test data is evaluated with votes. 
## 2.3 Random Forest
First, m' size random samples are selected with replacement. Then, speficied number of feautures are selected. Full growth trees are generated t times with random sample.Based on t trees, test data is evaluated with votes.

# 3. Linear regression
Please move to Experiment folder, and implement run.sh. The code for bias, variance decomposition is located at Experiment folder.
## 3.1 Batch Gradient Descent
It calculates gradient of the cost function by one step and update new weight vector.
## 3.2 Stochastic Gradient Descent
It calculated gradietn of the cost function one by one and immediately update new weight vector.

# 4. Perceptron learning
Please move to Experiment folder, and implement run.sh. The results of each methods - standard, voted, average- are saved at each folder as csv file.
## 4.1 Standard perceptron
Result Q2 2 a folder has all results. 
## 4.2 Voted perceptron
Result Q2 2 b folder has all results.
## 4.3 Average perceptron
Result Q2 2 c folder has all results.
## 4.4 Compare three methods
Reuslt Q2 2 d folder has all results. 
# 5. SVC
Please move to Experiment folder, and implement run.sh.
## 5.1 Stochastic sub gradient descent
First, suffle the data every epoch. Second, measure prediction. Third, update weight vector.
## 5.2 Linear classification
First, define boundary and constraints. Second, implement scipy optimize to minimize object function. Third, measure optimal weight vector and optimal bias. Finally, measure predictions. Everythins are based on linear kernel.
## 5.3 Non linear classification
First, define boundary and constraints. Second, implement scipy optimize to minimize object function. Third, measure optimal weight vector and optimal bias. Finally, measure predictions. Everythins are based on Gaussian kernel.
# 6. Neural Network
Please move to Experiment folder, and implement run.sh.
First, you need call NeuralNetwork class. Next, you need call add layer function. Neural network will find best weight through stochastic gradient descent with forward/back propagation. 
## 6.1 Gaussian
Initial weight is Gaussian distribution. It can be defined at add layer function.
## 6.2 Zero
Initial weight is zero array. It can be defined at add layer function.

# 7. Logistic Regression
Please move to Experiment folder, and implement run.sh.
## 7.1 MAP
Both prior and posterior parts are considered to objective function. We need to tune gamma, d and variance to find the good answer. We just optimize the objective function and update weight vector.
## 7.2 ML
Only posterior part is consitered to objective function. We need to turn gamm and d to get good ansdwer. We just optimize the objective function and update weight vector.  




