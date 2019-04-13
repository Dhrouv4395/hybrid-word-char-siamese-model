## Add the Quora Question Pairs (QQP) data here

### The QQP data contains two files:
* __train.csv__ 
* __test.csv__

>  Use train.csv to train your model 
>> split into training and validation/dev set while training

> Use test.csv for out of sample testing/evaluation of final selected model 

- __NOTE__: The original '_train.csv_' file contains about 400,000 question pairs <br>
              whereas the '_test.csv_' contains about 2,300,000 pairs. <br>
  So make your train and test split as you want by taking some examples from <br> '_test.csv_' into your '_train.csv_'
  if your model needs a lot of training data. 