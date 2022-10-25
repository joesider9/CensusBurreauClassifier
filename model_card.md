# Model Card


## Model Details
It is a Random forest classifier to estimate if the salary of an employ is greater than 50k or less
## Intended Use
This is an application to divide employees with respect to the salary in two categories
## Training Data
The dataset used come from UCI datasets and the 80% of data was used for training. 
Records with notation '?' considered nan and replaced with the most frequent value. 
White spaces are removed from the records.
All records rather than 'United States' for the native country feature was set to 'Other_value'
## Evaluation Data
The 20% of data was used for evaluation
Evaluation results
Precision: 0.806, Recall: 0.597, F1 0.686
## Metrics
Precision, Recall, F1 were used
## Ethical Considerations
The dataset contains sensitive information
## Caveats and Recommendations
The data is old, newer data should be found
Input data transformation could improve accuracy
More complex model could improve accuracy