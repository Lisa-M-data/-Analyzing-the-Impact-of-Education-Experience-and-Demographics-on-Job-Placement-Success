Sourced and imported a dataset from Kaggle

[ ]
import pandas as pd # importing pandas
Job_placement=pd.read_csv('/content/drive/MyDrive/data/job_placement.csv') #uploading dataset

[ ]

Start coding or generate with AI.
Title: Analyzing the Impact of Education, Experience, and Demographics on Job Placement Success

Hypothesis: There is a significant relationship between job placement success and factors such as previous work experience, GPA, and degree type, with work experience being the strongest predictor of job placement.

Variables: Dependent Variable:

Job Placement Status (Placed/Not Placed)

Independent Variables:

Age
GPA
Degree
Gender
Major
Previous Work Experience
College Name
Why I Chose This Topic: Through my experience as an Enrollment Advisor, Career Coach, and Talent Sourcer, I have closely observed the factors that influence candidates' career outcomes. This project seeks to identify the key independent variables—such as GPA, major, and previous work experience—that most significantly impact job placement success. By analyzing these factors, I aim to uncover insights that can help refine career support strategies and improve job placement rates for future job seekers.

Data cleansing and data review:

[ ]
Job_placement.head() ##I wanted a view of the keys for the dataset

Next steps:
Using dataframe:
Job_placement
suggest a plot
0 / 1000

[ ]

Start coding or generate with AI.

[ ]
Job_placement.tail() ## To get a view of the end of the dataset


[ ]
Job_placement.shape
(700, 8)

[ ]
Job_placement.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 700 entries, 0 to 699
Data columns (total 8 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   gender               700 non-null    object 
 1   age                  700 non-null    int64  
 2   degree               700 non-null    object 
 3   stream               700 non-null    object 
 4   college_name         700 non-null    object 
 5   placement_status     700 non-null    object 
 6   gpa                  700 non-null    float64
 7   years_of_experience  699 non-null    object 
dtypes: float64(1), int64(1), object(6)
memory usage: 43.9+ KB

[ ]
Job_placement.describe()


[ ]
Job_placement.dtypes #To see all of the data types in the dataset


[ ]
Job_placement.keys() #List of the keys
Index(['gender', 'age', 'degree', 'stream', 'college_name', 'placement_status',
       'gpa', 'years_of_experience'],
      dtype='object')

[ ]
Job_placement.isna().sum() #to check if there were many missing factors. The results came back as just one.


[ ]
Job_placement.dropna(inplace=True) # Drop null value
Job_placement.head()

Next steps:

[ ]
import pandas as pd
import matplotlib.pyplot as plt #Imported matplotlib to show visulations

[ ]
import seaborn as catplot
from matplotlib import pyplot as plt
import seaborn as sns
Developed visuals for exploratory analysis of the data

[ ]
Job_placement.gender.value_counts().plot(kind='bar',color='red',figsize=(3,3))
plt.title("Distribution of Gender")
plt.ylabel("Number of students")
plt.show() # shows the Females and Male students a nearly equal for this dataset.


[ ]
# To show what the students majors are.
Job_placement.age.value_counts().plot(kind='bar',color='red',figsize=(3,3))
plt.title("Distribution of Ages")
plt.ylabel("Number of students")
plt.show() # shows thedistribution ages amongst the students


[ ]
# To show what the students majors are.
Job_placement.stream.value_counts().plot(kind='bar',color='red',figsize=(3,3))
plt.title("Distribution of Majors")
plt.ylabel("Number of students")
plt.show() # shows the major distribution amongst the students


[ ]
sns.displot(data=Job_placement, x="placement_status", col="stream", kde=True)


[ ]
sns.displot(data=Job_placement, x="placement_status", col="gender", kde=True)


[ ]
sns.displot(data=Job_placement, x="placement_status", col="years_of_experience", kde=True)


[ ]
sns.pairplot(Job_placement)


[ ]

Start coding or generate with AI.

[ ]
sns.displot(Job_placement, x="placement_status", col="college_name", kde=True)


[ ]
from scipy.stats import chi2_contingency #imported chi2 code for crosstab and stats to compute a simple cross tabulation of two (or more) factors
import scipy.stats as stats
import numpy as np

[ ]
from pandas.core.reshape.pivot import crosstab
crosstab = pd.crosstab(Job_placement['placement_status'], Job_placement['degree']) # Used crosstab look at CGPAs and Panic to see if there is any correlation.
crosstab
#mental_health = mental_health.sort_values(by='CGPA', ascending=False)


Next steps:

[ ]
stats.chi2_contingency(crosstab)
#Pvalue is higher than the significance level of 0.05 this means that we cannot reject the null hypothesis that panic disorders affects CGPAs.
Chi2ContingencyResult(statistic=0.0, pvalue=1.0, dof=0, expected_freq=array([[130.],
       [569.]]))

[ ]
crosstab = pd.crosstab(Job_placement['gpa'], Job_placement['placement_status']) #A look at CGPAs and Anxiety to see if there is any correlation.

[ ]
stats.chi2_contingency(crosstab)
#Pvalue is slightly lower than the significance level of 0.05 this means that we can reject the null hypothesis that anxiety affects CGPAs.
Chi2ContingencyResult(statistic=0.0, pvalue=1.0, dof=0, expected_freq=array([[130.],
       [569.]]))

[ ]
crosstab = pd.crosstab(Job_placement['years_of_experience'], Job_placement['placement_status']) #A look at CGPAs and Depression to see if there is any correlation.
crosstab

Next steps:

[ ]
stats.chi2_contingency(crosstab)
#Pvalue is slightly lower the significance level of 0.05 this means that we can reject the null hypothesis that depression affects CGPAs.

[ ]
crosstab = pd.crosstab(Job_placement['gender'], Job_placement['placement_status']) #A look at CGPAs and treatment to see if there is any correlation.
crosstab

Next steps:

[ ]
crosstab = pd.crosstab(Job_placement['age'], Job_placement['placement_status']) #A look at CGPAs and treatment to see if there is any correlation.
crosstab

Next steps:
randomly select 5 items from a list
0 / 1000

[ ]
crosstab = pd.crosstab(Job_placement['college_name'], Job_placement['placement_status']) #A look at CGPAs and treatment to see if there is any correlation.
crosstab

Next steps:

[ ]
stats.chi2_contingency(crosstab)
#Pvalue is slightly higher than the significance level of 0.05 which means that we cannot reject the null hypothesis that getting treatment affects CGPAs.
Chi2ContingencyResult(statistic=94.03923784724724, pvalue=3.798734668285352e-21, dof=2, expected_freq=array([[ 29.94277539, 131.05722461],
       [ 47.05293276, 205.94706724],
       [ 53.00429185, 231.99570815]]))
Numerical model: I used the statsmodel multiple Linear Regression (with Categorical Predictors) because it seemed to be the most appropriate for my project.

[ ]
import statsmodels.formula.api as smf

# Assuming your Job_placement DataFrame already exists
# Drop irrelevant columns ('id', 'name', 'salary') if you haven't already done so:
Job_placement_cleaned = Job_placement.drop(columns=['id', 'name', 'salary'])

# Ensure categorical variables are properly encoded (optional step if needed)
# Job_placement_cleaned = pd.get_dummies(Job_placement_cleaned, drop_first=True)

# Fit the regression model using the cleaned data
model = smf.ols("gpa ~ gender + age + stream + college_name + years_of_experience", 
                data=Job_placement_cleaned).fit()

# Print the summary of the regression model
print(model.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    gpa   R-squared:                       0.929
Model:                            OLS   Adj. R-squared:                  0.923
Method:                 Least Squares   F-statistic:                     168.9
Date:                Sat, 30 Nov 2024   Prob (F-statistic):               0.00
Time:                        04:23:29   Log-Likelihood:                 1406.5
No. Observations:                 699   AIC:                            -2711.
Df Residuals:                     648   BIC:                            -2479.
Df Model:                          50                                         
Covariance Type:            nonrobust                                         
=============================================================================================================================
                                                                coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------------------------------------
Intercept                                                     3.5056      0.048     73.206      0.000       3.412       3.600
gender[T.Male]                                                0.0048      0.004      1.353      0.177      -0.002       0.012
stream[T.Electrical Engineering]                             -0.0005      0.005     -0.101      0.919      -0.009       0.008
stream[T.Electronics and Communication]                      -0.0007      0.005     -0.145      0.884      -0.011       0.009
stream[T.Information Technology]                              0.0025      0.004      0.642      0.521      -0.005       0.010
stream[T.Mechanical Engineering]                             -0.0146      0.005     -2.938      0.003      -0.024      -0.005
college_name[T.California Institute of Technology]           -0.0075      0.048     -0.157      0.875      -0.101       0.086
college_name[T.Columbia University]                           0.0929      0.049      1.890      0.059      -0.004       0.189
college_name[T.Duke University]                               0.1916      0.048      3.990      0.000       0.097       0.286
college_name[T.Georgetown University]                        -0.0107      0.048     -0.221      0.825      -0.106       0.084
college_name[T.Harvard University]                           -0.0087      0.048     -0.181      0.857      -0.103       0.086
college_name[T.Johns Hopkins University]                      0.0007      0.048      0.015      0.988      -0.093       0.095
college_name[T.Massachusetts Institute of Technology]        -0.0071      0.049     -0.145      0.884      -0.103       0.089
college_name[T.Northwestern University]                      -0.0928      0.049     -1.888      0.059      -0.189       0.004
college_name[T.Princeton University]                          0.1916      0.048      3.990      0.000       0.097       0.286
college_name[T.Rice University]                              -0.3045      0.048     -6.345      0.000      -0.399      -0.210
college_name[T.Stanford University]                           0.0096      0.048      0.199      0.842      -0.085       0.104
college_name[T.University of California--Berkeley]            0.0888      0.034      2.597      0.010       0.022       0.156
college_name[T.University of California--Davis]              -0.0933      0.049     -1.900      0.058      -0.190       0.003
college_name[T.University of California--Irvine]             -0.0069      0.049     -0.140      0.889      -0.103       0.090
college_name[T.University of California--Los Angeles]         0.0750      0.035      2.169      0.030       0.007       0.143
college_name[T.University of California--Riverside]           0.1932      0.035      5.557      0.000       0.125       0.261
college_name[T.University of California--San Diego]          -0.0478      0.036     -1.332      0.183      -0.118       0.023
college_name[T.University of California--San Francisco]      -0.0011      0.034     -0.033      0.974      -0.069       0.066
college_name[T.University of California--Santa Barbara]      -0.0082      0.048     -0.170      0.865      -0.103       0.086
college_name[T.University of California--Santa Cruz]         -0.0423      0.035     -1.209      0.227      -0.111       0.026
college_name[T.University of Chicago]                        -0.1037      0.048     -2.167      0.031      -0.198      -0.010
college_name[T.University of Colorado--Boulder]              -0.0052      0.035     -0.151      0.880      -0.073       0.063
college_name[T.University of Connecticut]                    -0.1031      0.036     -2.853      0.004      -0.174      -0.032
college_name[T.University of Delaware]                        0.0928      0.036      2.565      0.011       0.022       0.164
college_name[T.University of Florida]                         0.1886      0.048      3.912      0.000       0.094       0.283
college_name[T.University of Georgia]                        -0.0078      0.049     -0.161      0.872      -0.104       0.088
college_name[T.University of Illinois--Urbana-Champaign]      0.0057      0.034      0.167      0.868      -0.061       0.073
college_name[T.University of Iowa]                           -0.0002      0.048     -0.005      0.996      -0.094       0.093
college_name[T.University of Maryland--College Park]         -0.0042      0.035     -0.122      0.903      -0.072       0.064
college_name[T.University of Michigan--Ann Arbor]             0.0859      0.034      2.516      0.012       0.019       0.153
college_name[T.University of Minnesota--Twin Cities]          0.0002      0.048      0.005      0.996      -0.093       0.094
college_name[T.University of North Carolina--Chapel Hill]    -0.0090      0.036     -0.252      0.801      -0.080       0.062
college_name[T.University of Notre Dame]                     -0.0109      0.048     -0.227      0.821      -0.106       0.084
college_name[T.University of Pennsylvania]                   -0.0049      0.035     -0.141      0.888      -0.073       0.063
college_name[T.University of Pittsburgh]                     -0.0084      0.048     -0.176      0.861      -0.103       0.086
college_name[T.University of Rochester]                       0.0045      0.034      0.131      0.896      -0.063       0.072
college_name[T.University of Southern California]            -0.0071      0.049     -0.145      0.885      -0.104       0.089
college_name[T.University of Texas--Austin]                   0.0452      0.035      1.300      0.194      -0.023       0.114
college_name[T.University of Texas--Dallas]                  -0.0082      0.036     -0.228      0.820      -0.079       0.063
college_name[T.University of Virginia]                        0.1665      0.035      4.820      0.000       0.099       0.234
college_name[T.University of Washington]                      0.0027      0.034      0.078      0.938      -0.065       0.070
college_name[T.University of Wisconsin--Madison]              0.1911      0.048      3.979      0.000       0.097       0.285
college_name[T.Yale University]                              -0.2059      0.048     -4.280      0.000      -0.300      -0.111
age                                                           0.0002      0.001      0.203      0.839      -0.002       0.003
years_of_experience                                           0.0961      0.006     17.330      0.000       0.085       0.107
==============================================================================
Omnibus:                      350.503   Durbin-Watson:                   2.265
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             6412.827
Skew:                          -1.799   Prob(JB):                         0.00
Kurtosis:                      17.396   Cond. No.                     4.42e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 4.42e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
Model validation techniques

[ ]


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline



[ ]
import warnings

warnings.filterwarnings('ignore')

[ ]
from google.colab import drive #code to mount Google Drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

[ ]
import pandas as pd # importing pandas
Job_placement=pd.read_csv('/content/drive/MyDrive/data/job_placement.csv') #uploading dataset

[ ]
Job_placement.head()

Next steps:

[ ]

Start coding or generate with AI.

[ ]
Job_placement.shape
(700, 8)

[ ]
Job_placement['placement_status'].value_counts()

create visual
13 / 1000
1 of 1
Use code with caution

[ ]
# prompt: create visual

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'Job_placement' DataFrame is already loaded as in your provided code

# Example 1: Distribution of placement status by gender
plt.figure(figsize=(8, 6))
sns.countplot(x='placement_status', hue='gender', data=Job_placement)
plt.title('Placement Status by Gender')
plt.show()

# Example 2:  Boxplot of GPA by placement status
plt.figure(figsize=(8, 6))
sns.boxplot(x='placement_status', y='gpa', data=Job_placement)
plt.title('GPA Distribution by Placement Status')
plt.show()

# Example 3: Scatter plot of years of experience vs. GPA, colored by placement status
plt.figure(figsize=(10, 6))
sns.scatterplot(x='years_of_experience', y='gpa', hue='placement_status', data=Job_placement)
plt.title('Years of Experience vs. GPA')
plt.show()


[ ]
Job_placement.isnull().sum()


[ ]
X = Job_placement.drop(['placement_status'], axis=1)

y = Job_placement['placement_status']

[ ]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

[ ]
# check the shape of X_train and X_test

X_train.shape, X_test.shape
((469, 7), (231, 7))

[ ]
X_train.dtypes


[ ]
X_train.head()

Next steps:

[ ]
pip install --upgrade category_encoders
Requirement already satisfied: category_encoders in /usr/local/lib/python3.10/dist-packages (2.6.4)
Requirement already satisfied: numpy>=1.14.0 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (1.26.4)
Requirement already satisfied: scikit-learn>=0.20.0 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (1.5.2)
Requirement already satisfied: scipy>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (1.13.1)
Requirement already satisfied: statsmodels>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (0.14.4)
Requirement already satisfied: pandas>=1.0.5 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (2.2.2)
Requirement already satisfied: patsy>=0.5.1 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (1.0.1)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.5->category_encoders) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.5->category_encoders) (2024.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.5->category_encoders) (2024.2)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=0.20.0->category_encoders) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=0.20.0->category_encoders) (3.5.0)
Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.10/dist-packages (from statsmodels>=0.9.0->category_encoders) (24.2)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas>=1.0.5->category_encoders) (1.16.0)

[ ]
import category_encoders as ce

[ ]
encoder = ce.OrdinalEncoder(cols=['age', 'stream', 'gender', 'gpa', 'degree', 'college_name','years_of_experience'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

[ ]
X_train.head()

Next steps:

[ ]
X_test.head()

Next steps:

[ ]
 from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(n_features=4, n_informative=2,
...                        random_state=0, shuffle=False)
>>> regr = RandomForestRegressor(max_depth=2, random_state=0)
>>> regr.fit(X, y)
RandomForestRegressor(...)
>>> print(regr.predict([[0, 0, 0, 0]]))
[-8.32987858]
[-8.32987858]
[-8.32987858]

[ ]
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Decision Tree Classifier
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_tree = tree_model.predict(X_test)

# Evaluate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred_tree)
print(f'Accuracy: {accuracy}')
Accuracy: 0.9826839826839827

[ ]
from sklearn.svm import SVC

# SVM Model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_svm = svm_model.predict(X_test)

# Evaluate accuracy
accuracy_svm = metrics.accuracy_score(y_test, y_pred_svm)
print(f'Accuracy: {accuracy_svm}')
Accuracy: 0.7705627705627706

# Final Conclusions:
Multiple Variables Influence Job Placement: The study suggests that a combination of factors, such as GPA, previous work experience, and major, play a crucial role in influencing job placement outcomes. While certain factors like age and gender may have some influence, they are likely secondary to skills and relevant experience.

The Importance of Work Experience: Previous work experience seems to be one of the most significant predictors of whether a candidate is placed. Regardless of the degree or college attended, having relevant internships or jobs during college is a key factor in securing a job post-graduation.

GPA and Academic Achievement: GPA remains an important criterion for employers, but its influence might be stronger in certain industries like consulting, finance, or research. However, for many candidates, experience and skills may outweigh GPA once a certain threshold has been met.

Industry-Specific Variability: The importance of each factor (e.g., GPA, degree, major) can vary by industry. For example, tech companies might place a higher emphasis on skills and previous work experience (such as coding projects, internships, etc.), while fields like consulting or finance might prioritize academic achievement more.

Networking and College Prestige Matter: Candidates from prestigious universities or those who have strong networks (e.g., alumni, internships, career fairs) may have better chances of securing jobs, but this is less of a factor in fields that rely more on demonstrable skills and work experience.

# Next Steps: For Career Coaches and Advisors: Tailor career support strategies based on the most influential factors in job placement for students. For example, focus on gaining relevant work experience, improving GPA (if needed), and preparing for job interviews.

For Employers: Consider creating more holistic hiring criteria that emphasize experience and skills, rather than solely relying on a candidate’s GPA or college name. For Students: Focus on building practical skills, securing internships, and networking to improve job placement chances.


