# Setting up Experiments

The typical use case of the package is to document and run experiments with different set of parameters. 
I will describe typical usage with help of an example that is typical in training deep networks, but 
the same paradigm can be easily extended to different use cases. 

## Simple Experiments

We want to run multiple experiments with different values of ```NetParams```. Essentially what we want is 
a path name for each set of parameters, so that we can store the experiment related stuff in paths that are
related to the parameters. We would do this in the following way: 
```python
from exp_params_mng import params_object

""" 
We create the base class from which all other parameter classes in the
project will inherit from. The projectName property creates a database in the 
server by the name projectName. All other parameter objects create collections
in this database. 
""" 
class MyExampleObject(params_object.ParamsObject):
  @property
  def projectName(self):
     return 'example'
  
 
class NetParams(MyExampleObject):  
  def default_params(self):
    """ 
    Define the default set of parameters 
    """
    dParams = {}
    dParams['netName'] = 'alexnet'
    dParams['learningRate'] = 0.01
    return dParams
    
    
class ExperimentParams(MyExampleObject):
  def __init__(self, netObj, usertag=None):
    #netObj: object of type NetParams
    self._netObj = netObj
    super(ExperimentParams, self).__init__(usertag=usertag)

  def default_params(self):
    dParams = {}
    dParams['netHash'] = self._netObj.hash_name()
    return dParams
    
  def get_paths(self):
    paths = {}
    paths['exp'] = self.hash_name()
    return paths
```

Now suppose, we want to run two experiments with different learning rates, we would do the following

```python
netParams1 = NetParams(learningRate=0.1)
expParams1 = ExperimentParams(netParams1)
path_to_store = expParams1.get_paths()

#Second experiment with a different learning rate. 
netParams2 = NetParams(learningRate=0.2)
expParams2 = ExperimentParams(netParams2)
path_to_store = expParams2.get_paths()
```

This was easy! As you run these experiments you realize that you want to vary one parameter -- oh no! what to do now?
With `exp_params_mng` this is super simple. Suppose you were using the learning algorithm as `SGD` until now, and now 
you want to experiment with `RMSProp`. To do this we will simply update the default parameters in the `NetParams` object. 
This would add a new field in the `NetParams` table (collection in MongoDB lingo), and populate all the previous entries with
the value of learning algoithm as `SGD`. 

```python
class NetParams(MyExampleObject):  
  def default_params(self):
    dParams = {}
    dParams['netName'] = 'alexnet'
    dParams['learningRate'] = 0.01
    dParams['learningAlgo'] = 'SGD'
    return dParams
```

These changes will not effect the path of previous experiments! We can now go and launch new experiments:
```python
netParams3 = NetParams(learningRate=0.1, learningAlgo='RMSProp')
expParams3 = ExperimentParams(netParams3)
path_to_store = expParams3.get_paths()
```

## Slightly more complex experiments
You have now happily run many experiments and might have significantly changed `NetParams` to have more values etc. 
But now you realize that there is a different set of parameters that you care about -- the manner in which you 
are creating the dataset to run your experiments. Until now you were using 60\% of data for training, and 20\% each
for validation and testing. But you want to vary this now. 

How would you do this? Its again very simple! See below:

```python
class DatasetParams(MyExampleObject):  
  def default_params(self):
    """ 
    Define the default set of parameters 
    """
    dParams = {}
    dParams['trainPct'] = 60
    dParams['valPct'] = 20
    dParams['testPct'] = 20
    return dParams

#We will also update the ExperimentParams
class ExperimentParams(MyExampleObject):
  def __init__(self, netObj, dataObj=None, usertag=None):
    #netObj: object of type NetParams
    self._netObj  = netObj
    self._dataObj = dataObj
    super(ExperimentParams, self).__init__(usertag=usertag)

  def default_params(self):
    dParams = {}
    dParams['netHash'] = self._netObj.hash_name()
    dParams['dataHash'] = self._dataObj.hash_name()
    return dParams
    
  def get_paths(self):
    paths = {}
    paths['exp'] = self.hash_name()
    return paths
```

Now again, none of the previous experiment paths will be affected. We can create new experiments as following:

```python
netParams  = NetParams(learningRate=0.05, learningAlgo='RMSProp')
dataParams = DatasetParams(trainPct=80, valPct=20, testPct=20) 
expParams  = ExperimentParams(netParams, dataParams)
path_to_store = expParams.get_paths()
```

This is great! We can easily introduce new parameters as we go along to support experiments that we didn't think
we needed initially. 


## Deleting Irrelevant Parameters
Maybe at the beginning you were not sure of what parameters would be required later and you start with following

```python
class NetParams(MyExampleObject):  
  def default_params(self):
    dParams = {}
    dParams['netName'] = 'alexnet'
    dParams['learningRate'] = 0.01
    dParams['learningAlgo'] = 'SGD'
    dParams['deepLearningPackage'] = 'caffe'
    return dParams
```
You thought that you might need to experiment with different deep learning packages, but you realized at some point
that the awesome *caffe* package is sufficient and you don't need anything else. You can simply drop it and modify
`NetParams` as following:


```python
class NetParams(MyExampleObject):  
  def default_params(self):
    dParams = {}
    dParams['netName'] = 'alexnet'
    dParams['learningRate'] = 0.01
    dParams['learningAlgo'] = 'SGD'
    return dParams
```

That simple! 

However, try dropping the *learningRate* parameter:

```python
class NetParams(MyExampleObject):  
  def default_params(self):
    dParams = {}
    dParams['netName'] = 'alexnet'
    dParams['learningAlgo'] = 'SGD'
    return dParams
```

Because you ran experiments with different learning rates, this would not be possible and you will be prompted
with an error/warning. 


## Quick Access to Favorite Set of Parameters
While running many experiments you might realize that there is a set of parameters you love! For e.g. you really like:
```python
netParams  = NetParams(netName='resnet',learningRate=0.05, learningAlgo='RMSProp')
```
Instead of accessing your favorite set of parameters wouldn't it be awesome if one could just call these set of parameters
by a nick-name? This is exactly what `usertag` does!

Define the parameters as: 
```python
netParams  = NetParams(netName='resnet',learningRate=0.05, learningAlgo='RMSProp', usertag='myFavNet')
```
Now, next time, one can simply retrieve these parameters as:
```python
netParams  = NetParams(usertag='myFavNet')
```









