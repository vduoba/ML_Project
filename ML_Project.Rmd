
### Exercising the Correct Way

#### Executive Summary
It is possible to predict the "correctness" of exercise of an individual
with a very high degree of accuracy given 53 measurements on that individual.
This conclusion has been arrived at by investigating two prediction algorithms:

* Recursive Partitioning
* Random Forests

The Recursive Partioning algorithm was a poor performer, with an estimated accuracy
of only approximately 54.7%. On the other hand the Random Forests algorithm produced an
estimated accuracy of approximately 99.8%.
There is a steep computational cost associated with the more successful algorithm,
however. On a 3Ghz 2-CPU/4-Core processor the less successful algorithm executed in about 5 minutes, but the Randon Forests algorithm concluded after about 6 hours of execution. This is a 72-fold increase in computational time.

#### Methodology
The training dataset had 19622 observations and 154 columns of data.
Examination of this dataset revealed that some of the columns were metadata
(that is, administrative information dealing with survey collection details), and
many other columns were empty or had copious amounts of missing data. The metadata occupied the first 6 columns of the file and this data was cut out of the working file. (The same
was done to the test dataset). 

Also supplied was a test dataset with 20 observations. A pragmatic decision was made to restrict the working version of the training dataset to those columns in the TEST dataset that had no missing or NA data. This resulted in datasets that had 54 columns each, the last of which was either "classe" (the response variable) or "problem_id" (a sequential integer).

The original "training" dataset (after the preceding variable reductions) was then split 60/40 into:

* a training portion (df_train), and 
* a validation portion (df_valid)

The caret package was then used to provide the algorithms used to fit the two prediction models using just the df_train random subset of the observations. Each model was then validated using the df_valid subset. Finally, the model that was fitted to the df_train subsample was run against the test dataset to yield the test set predictions.

It is possible that fewer than 53 of the features used in the fitted model could have sufficed, but this possibility 
was not pursued owing to the lack of time.
#### Results
##### Recursive Partioning

Predicted(down) vs Actual(across) for Recursive Partioning (using the validation subset)

 
            A     B      C      D      E   
 
      A   1805   296    147    364    310
      
      B     88   721    119    293    253
      
      C    333   501   1102   629     219
      
      D     0      0     0      0       0
      
      E     6      0     0     0      660

Accuracy =  (1805+721+1102+0+660)/(7846) = 54.7%

##### Random Forests

Predicted(down) vs Actual(across) for Random Forests (using the validation subset)

            A       B         C       D      E 
      

      A    2232       3       0       0       0
      
      B       0    1511       1       0       0
      
      C       0       4    1367       3       0
      
      D       0       0       0     1283      4
      
      E       0       0       0        0   1438
      
Accuracy=(2232+1511+1367+1283+1438)/7846 =99.8%     


### Technical Appendix I: Code (only the substantive lines)
#### Read in the two data files:
pml_train<- read.table("C:\\Users\\vic\\SkyDrive\\Education\\Coursera\\Machine_Learning_June2014\\pml-training.csv", header=TRUE,sep = ",", quote = "\"") # Training

pml_train<-pml_train[,7:dim(pml_train)[2]] # Get rid of training set metadata in first 6 columns

pml_test<- read.table("C:\\Users\\vic\\SkyDrive\\Education\\Coursera\\Machine_Learning_June2014\\pml-testing.csv", header=TRUE,sep = ",", quote = "\"") # Test

pml_test<-pml_test[,7:dim(pml_test)[2]] # Get rid of test set metadata in first 6 columns

#### Are there any columns that are completely NA or empty in the test dataset (those vars will not be of predictive use!)
pml_test_useful  <- pml_test[,colSums(is.na(pml_test))<nrow(pml_test)] # Keep only cols that are not all NA or empty
#### So if the test dataset lacks those variables, omit them from the training set as well
pml_train_useful <- pml_train[,colSums(is.na(pml_test))<nrow(pml_test)] # Keep only cols that are not all NA or empty

#### Split the training data into two portions: Train and Validation
in_train =createDataPartition(y=pml_train_useful$classe, p=0.6,list=FALSE)

df_train =pml_train_useful[in_train, ] # Training subset

df_valid =pml_train_useful[-in_train, ] # Validation subset

### (1) Fit the Recursive Partitioning model:-
modFit=train(classe ~ .,method="rpart", data=df_train)
#### Do Recursive Partitioning predictions on the validation datset:-
pred=predict(modFit, newdata=df_valid[,1:(length(pml_train_useful)-1)])
#### Print the table of Recursive Partitioning predicted vs actual for the validation subset:-
table(pred,df_valid[,dim(df_valid)[2]])
### (2) Next fit a Random Forests model:-
modFit_RF=train(classe ~ .,method="rf", data=df_train, prox=TRUE)
#### Do Random Forests predictions on the validation datset:-
pred_RF=predict(modFit_RF, newdata=df_valid[,1:(length(pml_train_useful)-1)])
#### Print the table of Random Forests predicted vs actual for the validation subset:-
table(pred_RF,df_valid[,dim(df_valid)[2]])

### Technical Appendix II: Miscellaneous
```{r computing, fig.width=9, fig.height=8, echo=FALSE}
setwd("C:/Users/vic/SkyDrive/Education/Coursera/Machine_Learning_June2014")
library(caret);library(e1071); library(rattle); library(rpart.plot);
#
# The data being read in has lots of empty cells (""), lots of NA's and some ":DIV/0" cells.
# Have to decide whether or not to discard any columns and/or any observations
# Some columns might be names and date/time columns so might not be useful for prediction
# Read in the training data:
# Must get rid of any metadata of no use to prediction (1st 7 columns):
pml_train<- read.table("C:\\Users\\vic\\SkyDrive\\Education\\Coursera\\Machine_Learning_June2014\\pml-training.csv", header=TRUE,sep = ",", quote = "\"")
pml_train<-pml_train[,7:dim(pml_train)[2]] # Get rid of sequential var and person name
# Read in the test data in case there are fewer vars than in training (be paranoid!):
pml_test<- read.table("C:\\Users\\vic\\SkyDrive\\Education\\Coursera\\Machine_Learning_June2014\\pml-testing.csv", header=TRUE,sep = ",", quote = "\"")
pml_test<-pml_test[,7:dim(pml_test)[2]] # Get rid of sequential var and person name

# Get some metadata about the tables:
#dim(pml_train);
#dim(pml_test);
# Are there any columns that are completely NA or empty in the test dataset (those vars will not be of predictive use!)
pml_test_useful  <- pml_test[,colSums(is.na(pml_test))<nrow(pml_test)] # Keep only cols that are not all NA or empty
# So if the train dataset lacks those variable, omit them from the training set as well
pml_train_useful <- pml_train[,colSums(is.na(pml_test))<nrow(pml_test)] # Keep only cols that are not all NA or empty
#dim(pml_test_useful);  # Only 58 columns remain.  Last col is "problem_id".
#dim(pml_train_useful); # Only 58 columns remain, but 19622 rows. Last col is "classe".
# Get the column names of thiose variables:
useful_cols=colnames(pml_train_useful)[1:length(pml_train_useful)-1] # last column is classe (classification)
sprintf("%s","Remaining variables for model fitting and prediction:")
useful_cols
# Fancy tree diagram
sprintf("%s","Tree Diagram for Recursive Partitioning Model")
fancyRpartPlot(modFit$finalModel)
```

