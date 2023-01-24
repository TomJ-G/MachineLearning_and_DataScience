### Please note, this is still under construction

# Volume estimation

In this small project, we would like to see, if it's possible to estimate [unit cell](https://en.wikipedia.org/wiki/Unit_cell). 
volume with different types of information.

## Why is this task not straightforward?
Arrangement of atoms in unit cell depends on multiple factors: number of atoms, atom types, bonds between atoms, symmetry ect.
It can even change for the same compound depending on temperature or pressure. I would like to know if it's enough to know chemical data (mass, formula ect), to derrive potential unit cell volume (and in the future projects, if I can predict unit cell geometry: lengths a, b, c and angles alpha, beta, gamma)


#### Here, we want to see to what extent volume can be predicted with increasing amount of information.

## 0) Setting up.
For the following models, I'll be using around 10000 CIF files from two Journals: Acta Crystallographica C and Organometallics. There are several ways you can download data from COD. The most preffered is to use subversion or rsync (for mode details please check [COD webpage](https://wiki.crystallography.net/howtoobtaincod/)).
First we need to explore our data. Let's load first file and check how to extract our data. I prepare the list of all CIF files with the following function:

```python
def list_files(root_folder,ftype,Silent=True,deep=True):
    """
    Creates list of files of selected format in the directory.
    Parameters:
        FPATH  - absolute path to the folder with diffraction frames files.
        ftype  - Extension of file (e.g. 'tif','h5'...)
        Silent - True by default. If set to False, prints information on number of found frames.
        deep   - True by default - check folder and ALL subfolders. If False will only check in FPATH 
    """
    from pathlib import Path
    if Silent == False:
        print("Reading",ftype,"files...")
    if deep == True:
        Read_folder = Path(root_folder).rglob('*.'+str(ftype))
    elif deep == False:
        Read_folder = Path(root_folder).glob('*.'+str(ftype))
    Files = [x for x in Read_folder]
    if Silent == False:
        print(len(Files),"files found!")
    return Files

```

Now I can view random file. It can be seen that file structure is not homogeneous. Some parts follow key: value pattern such as:

  *(The following data is cut and modified, just to show different parts of file structure)*
```
_cell_length_a                   1.000(1)
_cell_length_b                   5.0000(1)
_cell_length_c                   20.000(3)
_cell_measurement_temperature    295
_cell_volume                     100.0
```
And in the other parts, we can see atomic coordinates arranged like this:
```
loop_
_atom_site_label
_atom_site_occupancy
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_thermal_displace_type
_atom_site_B_iso_or_equiv
_atom_site_type_symbol
O1 1.0 0.0 0.0 0 Biso 1.000 O
C1 1.0 0.5 0.5 0.100(7) Biso 1.000 C
```

Different parts of file may require from us different data extraction approach. But this is problem for later. 
First, we should think, which parameters in CIF file can be used to derrive Unit Cell volume.
The most obvious answer would be: cell geometrc parameters: lengths a, b, c and angles alpha, beta, gamma. Using these would give us answer right away, just by using [formula](https://www.iucr.org/__data/iucr/cifdic_html/1/cif_core.dic/Icell_volume.html). As you may guess, the aim of this project is to use different values, not geometrical parameters.

Next, chemical formula, mass, number and type of atoms can be used to approximate the volume. Atom types should be very useful - each atom has different radius. By using only mass, we will have only very rough estimate - we won't be able to tell what combination of atoms is inside the cell.
Then there is pressure and temperature. It is known, that by decreasing temperature, or by increasing pressure, the volume of unit cell can be decreased (and in reverse: heating causes UC volume to expand). Such information should be included in the models.
Lastly, we should be able to tell UC volume very precisely by knowing atomic coordinates. But as I previously mentioned - I would rather estimate UC volume just by chemical information, not by structural data.

Let's try to make our first model.

## 1) Atomic mass + temperature + Z

As mentioned before, using just mass should give us very rough estimate of UC volume. To increase accuracy a little bit, we will also use temperature. In addition, we should extract the "Z" number, which tell us how many symmetrically equivalent parts are in the cell.
For extraction I used this code:
```python
#Load modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CIFs = list_files(root_path,'cif')

keys = ['_chemical_formula_weight ','_cell_measurement_temperature','_cell_formula_units_Z','_cell_volume']
optionals = {'_cell_measurement_temperature':297}

extracted = []
for cif in CIFs:
    data = {}.fromkeys(keys)
    with open(cif,'r+') as file:
        for i,line in enumerate(file):
            for k in keys:
                if k in line:
                    value = line.split(" ",maxsplit=1)[1].strip().strip("'").split('(')[0]
                    if data[k] == None:
                        data[k] = value
        for o in optionals:
            #Additional if, because some authors like to comment parameters later
            if data[o] == None: 
                data[o] = optionals[o] 
    #This is to prevent us from using incomplete data
    if None not in list(data.values()):
        extracted.append(list(data.values()))
```
Because in some cases authors used "," instead of "." in numbers, we must take care of it. Also, we should use floats instead of ints.
```python
for i,e in enumerate(extracted):
    #All purely numerical values should be changed to float
    if type(e[0]) == str:
        e[0] = e[0].replace(',','.')
    if type(e[1]) == str:
        e[1] = e[1].replace(',','.')
    if type(e[0]) == str:
        e[2] = e[2].replace(',','.')
    if type(e[3]) == str:
        e[3] = e[3].replace(',','.')
    e[0] = float(e[0])
    e[1] = float(e[1])
    e[2] = float(e[2])
    e[3] = float(e[3])
```
we can already split our data into train and test sets.
```python
DF = pd.DataFrame(extracted[:len(extracted)-200], columns=['weight','temperature','Z','volume'],dtype=float)
TST = pd.DataFrame(extracted[-200:], columns=['weight','temperature','Z','volume'],dtype=float)
```

In the next step I define TF model

```python
import tensorflow as tf
from tensorflow.keras import layers

features = DF.copy()
labels = features.pop('volume')
features = np.array(features)

test = TST.copy()
test_lab = test.pop('volume')
test = np.array(test)

model = tf.keras.Sequential([
    layers.LSTM(32,input_shape=(None,1),return_sequences=True),
    layers.LSTM(32),
    layers.Dense(32,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(1)
])

model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005))

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False)
```

Fit the model

```python
history = model.fit(features,labels,epochs=100,validation_split=0.2,verbose=0,callbacks=[callback],
                    batch_size=250)
history_dict = history.history
```
Let's check the loss functions over time.
```python
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'ko', label='Training loss',markersize=3)
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
```
As it can be seen there is an issue: validation loss is lower than training loss!
![image](https://user-images.githubusercontent.com/59794882/214294284-f8c8416d-09ed-45b6-ae62-96d6cffaa9d6.png)
#### This will be updated in the future, with better model, but in the meanwhile let's look how accurate the model is
I do this by plotting real volume vs predicted volume of the test set. For ideal fit, datapoints should follow red line (y=x)
```python
tested = model.predict(test)
check = []
for i,v in enumerate(tested):
    check.append([*v,test_lab[i]])
check = np.array(check)
plt.rcParams['figure.dpi'] = 100
plt.scatter(check[:,1],check[:,0],c='k',s=2)
plt.scatter(range(1000,3500),range(1000,3500),c='r',s=1)
plt.xlabel('Prediction')
plt.ylabel('Actual volume')
```
![image](https://user-images.githubusercontent.com/59794882/214295274-f9decba1-a60c-445e-b8ca-a2c1e65bc182.png)

So how accurate the model is? Not very accurate, which can be shown on few examples of well known compounds.

#### TBC
