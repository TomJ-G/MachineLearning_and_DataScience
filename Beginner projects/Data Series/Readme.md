# Interpolation of data series

In this project we want to interpolate function from incomplete data.  
First we import our data series csv. Let's see few last rows of data:

| index |	t |	x |	y |
|-------|---|---|----|
|1995|19.95|-|-0.008373420006295151|
|1996|19.96|-|	-|
|1997|19.97|-0.10338498289342064|	-0.016746906664168143|
|1998|19.98|-0.08008968816969166|	-|
|1999|19.99|-|	-|

As we can see some data is missing and is replaced by "-". The best aproach in this case would be to simply remove the rows which contain it. 
Here is the class to handle this task. We can also plot the data which is left.

```python
class ProcessData():
    
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file,sep=',')
        #We remove the "-" symbols to replace them with NaN
        df.replace('-',None,inplace=True)
        df = df.astype(float)
        df = df.dropna()
        self.t = df.values[:,:-2].astype('float32')
        self.y = df.values[:,1:].astype('float32')
        self.y = self.y.reshape((len(self.y),2))
        return 
        
    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return([self.t[idx],self.y[idx]])


X = ProcessData('dat.csv')
plt.scatter(b.t,b.y[:,0],s=3,c='k')
plt.scatter(b.t,b.y[:,1],s=3,c='r')
```
![image](https://user-images.githubusercontent.com/59794882/214746012-5d4b1cc6-101b-4e8c-be01-062291f55312.png)

Now we need to write a model.
```python
def Interpolation_model(n=2):
    """
    Model to interpolate the data.
    Parameter n indicates how many data series will be processed.
    """
    model = tf.keras.Sequential([
        layers.LSTM(40,input_shape=(None,1),return_sequences=True),
        layers.LSTM(40),
        layers.Dense(n)
    ])

    model.compile(loss = tf.keras.losses.MeanSquaredError(),
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))

    return(model)
```
And now we try to fit the data.

```python
model = Interpolation_model()
history = model.fit(X.t,X.y,epochs=600,verbose=0,batch_size=10)
history_dict = history.history
print(history_dict['loss'][-1])

>>0.0018835801165550947
```
The resulting fitting:  
![image](https://user-images.githubusercontent.com/59794882/214748642-c146bcf7-9da1-413e-aa70-54847e7f3428.png)
  
    
It should be noted that such model is good to interpolate data - not to predict the future course of the function.
We didn't do any validation, so we also don't know how much of overfitting we have.


