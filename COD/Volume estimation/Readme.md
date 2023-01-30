# Volume estimation

In this small project, we would like to see, if it's possible to estimate [unit cell](https://en.wikipedia.org/wiki/Unit_cell). 
volume (UC volume) with different types of information.

## Why is this task not straightforward?
The arrangement of atoms in a unit cell depends on multiple factors: number of atoms, atom types, bonds between atoms, symmetry, etc.
It can even change for the same compound depending on temperature or pressure. I would like to know if it's enough to know chemical data (mass, formula etc), to derive potential unit cell volume (and in future projects, if I can predict unit cell geometry: lengths a, b, c and angles alpha, beta, gamma). I will use increasingly more information (i.e. additional features) to see how prediction accuracy changes.

## 0) Setting up.
For the following models, I'll be using around 10000 CIF files from two Journals: Acta Crystallographica C and Organometallics. There are several ways you can download data from COD. The most preferred is to use subversion or rsync (for more details please check [COD webpage](https://wiki.crystallography.net/howtoobtaincod/)).
The IDs of structures I used can be used in this folder in ID_list.txt file.  
Before we start building models, we need to explore our data. Let's load the first file and check how to extract our data. I prepare the list of all CIF files with the following function:

```python
#Function was taken out of one of my other projects
def list_files(root_folder,ftype,Silent=True,deep=True):
    """
    Creates list of files of selected format in the directory.
    Parameters:
        FPATH  - absolute path to the folder with diffraction frames files.
        ftype  - Extension of file (e.g. 'cif','h5' ect...)
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

Now I can view the random file. The data that would be interesting to us is in the form of key: value pairs such as:

*(The following data is cut and modified, just to show different parts of the file structure. The formula is not real one)*
```
_cell_length_a                   1.000(1)
_cell_length_b                   5.0000(1)
_cell_length_c                   20.000(3)
_cell_measurement_temperature    295
_cell_volume                     100.0
_chemical_formula_sum            'C4 H4 X...'
```

We should consider, which parameters in the CIF file can be used to derive Unit Cell volume.
The most obvious answer would be: cell geometrc parameters: lengths a, b, c and angles alpha, beta, gamma. Using these would give us answer right away, just by using [formula](https://www.iucr.org/__data/iucr/cifdic_html/1/cif_core.dic/Icell_volume.html). But the purpose of this project is to *NOT* use structural data. Structural information can be obtained after compound is studied with XRD of simillar technique. We would like to use only chemical and physical properties, known before XRD experiment.

We will use chemical formula, formula mass, number and type of atoms. Also information such as pressure and temperature should be helpful. It is known, that by decreasing temperature, or by increasing pressure, the volume of unit cell can be decreased (and in reverse: heating causes UC volume to expand).

Let's try to make our first model, with the simplest approach possible.

## 1) Atomic mass + temperature + Z

Using formula mass should give us very rough estimate of UC volume. Mass is connected with types and number of atoms in the structure.
To increase accuracy a little bit, we will also use temperature. In addition, we should extract the "Z" number, which tell us how many symmetrically equivalent parts are in the cell.
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
We can shuffle the data and then split it into train and test sets. Shuffling can be simply done with "sample" method.
```python
DF = pd.DataFrame(extracted, columns=['weight','temperature','Z','volume'],dtype=float)
DF = DF.sample(frac=1).reset_index().drop(columns='index')

test = DF.iloc[-1000:]
features = DF.iloc[:-1000]
labels = features.pop('volume')
```

In the next step I define TF model

```python
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Dense(256,activation='relu'),
    layers.Dense(128,),
    layers.Dense(1)
])

model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=1,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False)
```

Fit the model

```python
history = model.fit(features,labels,epochs=500,validation_split=0.2,verbose=0,callbacks=[callback])
history_dict = history.history
```
Let's check the loss functions over time. We define plotting function
```python
def plot_loss(history_dict):
    
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'ko', markersize=3, label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

plot_loss(history_dict)
```
As it can be seen there is an issue: validation loss is lower than training loss!  
![image](https://user-images.githubusercontent.com/59794882/215477219-d4d64aec-f7a6-4e1b-ac48-dc9812c3ac8d.png)
  
#### This will be updated in the future, with better model, but in the meanwhile let's look how accurate the model is
I do this by plotting real volume vs predicted volume of the test set. For ideal fit, datapoints should follow red line (y=x)
```python
test_labels = test.pop("volume")

tested = model.predict(test)
check = []
for i,v in enumerate(tested):
    check.append([*v,test_labels.iloc[i]])
check = np.array(check)
plt.rcParams['figure.dpi'] = 100
plt.scatter(check[:,1],check[:,0],c='k',s=2)
plt.scatter(range(1000,3500),range(1000,3500),c='r',s=1)
plt.xlabel('Prediction')
plt.ylabel('Actual volume')
```
  
![image](https://user-images.githubusercontent.com/59794882/215477320-1c1afd94-94f0-4f20-83f5-6d5bdacd528b.png)
  
As it can see, the initial model is not very good. Before we optimize this model, lets try another one.
  
  
## 2) Atom types + temperature
This time we will use chemical composition of each compound.
Because we cannot just pass a string with formule e.g. C9H8O2, we need to use encoding.
We start again with data extraction. Instead od weight we use "_chemical_formula_sum" key.
```python
keys = ['_chemical_formula_sum','_cell_formula_units_Z',
        '_cell_measurement_temperature','_cell_volume']
optionals = {'_cell_measurement_temperature':297}

#Because we will be working with atom names, we should make sure that ONLY correct atom names are used.
#To do this, we will check atom names against periodic table.
periodic_table = ['H','D','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',
                  'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br',
                  'Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te',
                  'I','Xe','Cs','Ba','La','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi',
                  'Po','At','Rn','Fr','Ra','Ac','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl',
                  'Mc','Lv','Ts','Og','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm',
                  'Yb','Lu','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']
```

For data extraction we use exactly the same function as previously. There is a change in data processing:

```python
atom_types = []
atd = {}
#Now we want to correctly porcess extracted data
for i,e in enumerate(extracted):
    #All purely numerical values should be changed to float
    e[1] = float(e[1])
    e[2] = float(e[2])
    e[3] = float(e[3])
    #Atom list need additional extraction
    atoms = e[0].split(' ')
    datom = {}
    for a in atoms:
        M = re.match('([A-Za-z]+)(\d*)',a)
        atom = M[1]
        if atom not in atom_types and atom in periodic_table:
            atom_types.append(atom)
        if M[2] == '':
            count = 1
        else:
            count = float(M[2])
        datom[atom] = count
    e[0] = datom
    if all([True for d in e[0] if d in atom_types]) == False:
        extracted.pop(i)

atom_types.sort()
    
#Prepare encoding
dt = []
for e in extracted:
    row = []
    for a in atom_types:
        if a in e[0].keys():
            val = e[0][a]
            row.append(val*e[1])
        else:
            row.append(0)
    row.append(e[2])
    row.append(e[3])
    dt.append(row)
    
DF = pd.DataFrame(dt, columns=[*atom_types,'Temp','Cell Volume'])
DF.head()
```
Which results in:  

![image](https://user-images.githubusercontent.com/59794882/214562792-4d68f2fa-ce6b-4a49-a4be-589d5438c2c5.png)

As you can see, this is **not optimal** way to encode our data. Encoding is very sparse, but we will deal with this later.  
We again, shuffle, split the data into train/test, and separate features from labels.

```python
DF = DF.sample(frac=1).reset_index().drop(columns='index')
test = DF.iloc[-1000:]
features = DF.iloc[:-1000]

#features = DF.copy()
labels = features.pop('Cell Volume')
features = np.array(features)

model = tf.keras.Sequential([
    layers.Dense(256,activation='relu'),
    layers.Dense(128,),
    layers.Dense(1)
])

model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=1,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False)
    
history = model.fit(features,labels,epochs=500,validation_split=0.2,verbose=None)
history_dict = history.history
plot_loss(history_dict)
```
  
![image](https://user-images.githubusercontent.com/59794882/215462672-531ee65e-a869-42c6-b98a-d636a5993b3b.png)
  
Loss seems to be much lower. Let's see how plot of real/predicted data will perform.

![image](https://user-images.githubusercontent.com/59794882/215463900-404f9451-ab99-4158-b306-100b61d422e7.png)
  
It seems that this approach is giving really good results.  
  
At this point we might not want to optimize the first model, because this one is performing so much better, and the only modification was to use chemical formula data instead of weight. But, this approach might not seem so good becasue of sparse input. So how do we deal with this?  
If we insist on using only chemical data, we might try to calculate atomic volumes, and use this information in the model. Atomic radii are well known, we can easily get this data from the web.  
It should be noted that volume calculated from chemical formula almost always is *different* from unit cell volume due to "void" space in between atoms. Also, we might want to combine this model with the initial one - if we use formula weight and atomic radii.

## 3) Atomic volume + temperature
First we load atomic data table. The original data contains 109 rows. In this one I added 110th row with data for Deuterium. To simplyfy things, all data for Deuterium is the same as for hydrogen (which is not accurate, but should be enough for us).  
  
```python
table = pd.read_csv("Atom_table.csv",index_col=0)
 
#We don't need other data, only atom symbol and radius
WdV_table = table[['Symbol','vdW_Radius']].set_index('Symbol')

#We define function for atomic volume
def atom_vol(atom_type):
    return(4/3*np.pi*(float(WdV_table.loc[atom_type]))**3) #Volume is in Angstrom^3

#Create DataFrame as in previous cases. Atomic volume is multiplied by Z number.
dt_2 = []
for e in extracted:
    row = []
    vol = 0
    for a in atom_types:
        if a in e[0].keys():
            val += atom_vol(a)*e[0][a]
    row.append(val*e[1])
    row.append(e[2])
    row.append(e[3])
    dt_2.append(row)
DF2 = pd.DataFrame(data=dt_2,columns=['Mol_V','Temp','Cell Volume'])
```
Once again, data is shuffled and separated, then we build model, fit data and plot loss
```python
DF2 = DF2.sample(frac=1).reset_index().drop(columns='index')

test = DF.iloc[-1000:]
features = DF.iloc[:-1000]

labels = features.pop('Cell Volume')
test_labels = test.pop('Cell Volume')
features = np.array(features)

model = tf.keras.Sequential([
    layers.Dense(256,activation='relu'),
    layers.Dense(128,),
    layers.Dense(1)
])

model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=1,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False)
    
history = model.fit(features,labels,epochs=500,validation_split=0.2,verbose=None)
history_dict = history.history
plot_loss(history_dict)
```
  
![image](https://user-images.githubusercontent.com/59794882/215473454-be1eb71f-68b0-4a07-8cda-3569fe3e5247.png)
  
Predicted vs real data:  
![image](https://user-images.githubusercontent.com/59794882/215473520-ba95f489-bd6d-4bc7-aab8-8db78364764a.png)
  
As you can see we managed to get similarly accurate model, with simplified input!


