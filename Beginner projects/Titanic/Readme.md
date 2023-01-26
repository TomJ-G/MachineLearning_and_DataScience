# Titanic dataset

This is one of "classic" datasets used for machine learning.
Set contains of 627 rows of data in 10 columns.

First I'll download and explore the dataset:
```python
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic.head()
```
| survived |	sex |	age |	n_siblings_spouses |	parch |	fare |	class |	deck |	embark_town |	alone |
|----------|------|-----|--------------------|--------|------|--------|------|--------------|-------|
|	0	|male |	22.0 |	1 |	0	| 7.2500 | Third |	unknown |	Southampton |	n |
|	1	|female	|38.0	|1	|0	|71.2833	|First	|C	|Cherbourg	|n|
| 1	|female	|26.0	|0	|0	|7.9250	|Third	|unknown	|Southampton	|y|
|	1	|female	|35.0	|1	|0	|53.1000	|First	|C	|Southampton	|n|
|	0	|male	|28.0	0	|0	|0  |8.4583	|Third	|unknown	|Queenstown	|y|

We can see that columns are both numerical and categorical.
I'll remove the column "embark_town" as this should not influence survivability of the passenger. 
Also, I would like to change text (object) columns to categorical ones.

```python
#Copy the data to not overrite the original
data = titanic.copy()

data.drop(columns='embark_town',inplace=True)
#We will need change object-type columns to number-caregorical ones
categorical_columns = list(data.select_dtypes(include='object').columns)
#numeric columns will be all changed to float
numeric_columns = [i for i in data.columns if i not in categorical_columns]
data[numeric_columns] = data[numeric_columns].astype(float)
data.info()

#We change object columns to categorical ones:
for col in categorical_columns:
    unlist = {n:i for i,n in enumerate(data[col].unique())}
    data[col] = data[col].apply(lambda x: float(unlist[x]))
data.info()
```

Once this is done, we can try to create the predictive model. First I split labels from features. Because the prediction is 0 or 1 I'll use binary crossentropy loss function.
Also, I setup early callback function.

```python
#Split labels from features.
labels = data.pop('survived')

#build the model
def titanic_model():
    model = tf.keras.Sequential([
        layers.Dense(20),
        layers.Dense(1)
    ])

    model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)) 
    return(model)

#Callback function
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=0,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True)
```

Now I only need to fit the model to data:
```python
#fit the model
model = titanic_model()
hist = model.fit(x=data, y=labels, epochs=100,validation_split=0.2,verbose=0)
print("Loss:",hist.history['loss'][-1],"Validation loss:",hist.history['val_loss'][-1])
```
My loss functions scores are:  
Loss: 0.4575953185558319 Validation loss: 0.3655484914779663  
  
We can see how loss changed with each epoch in the below plot:  
![image](https://user-images.githubusercontent.com/59794882/214764190-4ea70b3b-0316-4eaf-bb51-a965a5569c51.png)


