import abc
import copy
import numpy as np
import datetime
import pymongo
import pickle
import os
from os import path as osp
import bson
import yaml

##3rd party
from utilities import path_utils
from utilities.config_utils import ConfigHelper


class CustomError(Exception):
  """
  Custom Exception
  """
  pass


class ExpMngConfigHelper(ConfigHelper):
  @property
  def allOptKeys(self):
    return ['EXP_PARAMS_MONGO_SERVER', 
            'EXP_PARAMS_MONGO_PORT',
            'EXP_PARAMS_AUTH_DB',
            'EXP_PARAMS_USER',
            'EXP_PARAMS_PWD'
            ]


def get_config():
  """ Return configuration parameters."""
  opts = {'EXP_PARAMS_MONGO_SERVER': 'db',
           'EXP_PARAMS_MONGO_PORT': '27017',
           'EXP_PARAMS_AUTH_DB': 'params',
           'EXP_PARAMS_USER': 'default',
           'EXP_PARAMS_PWD': 'default'
          }
  config = ExpMngConfigHelper(opts)
  #check if the conf file exists
  confDir = os.path.expanduser('~/.exp_params_mng')
  if not os.path.exists(confDir):
    os.makedirs(confDir)
  confFile = os.path.join(confDir, 'config.yml')
  #Update config from a config file if it exists
  config.update_from_yaml(confFile)
  #Update config from shell 
  config.update_from_shell()
  return config.opts


def get_mongo_client():
  """ 
  Return the mongo client for storing experiment parameters.
  
  :param opts: options for constructing the client. 
  """
  opts = get_config()
  port, server = opts['EXP_PARAMS_MONGO_PORT'],opts['EXP_PARAMS_MONGO_SERVER']
  if port is None:
    addr = server
  else:
    addr = '{0}:{1}'.format(server, port)
  expClient = pymongo.MongoClient(addr)
  dbName = opts['EXP_PARAMS_AUTH_DB']
  expClient[dbName].authenticate(opts['EXP_PARAMS_USER'],
                                 opts['EXP_PARAMS_PWD'])
  return expClient


class ParamsObject(object):
  __metaclass__ = abc.ABCMeta
  _RESERVED_PARAM_NAMES_ = ['name', 'usertag']
  def __init__(self, params, usertag=None, userConfirm=True):
    """
      usertag: a user specified tag to name the object
      userConfirm: whether a user needs to confirm some consistency check operations 
      params : a set of parameters defining the abstract object
    """
    self._params = self.default_params()
    #Assert that user specified params is a subset
    #of default parameters 
    userKeys    = params.keys()
    defaultKeys = self._params.keys()
    if not set(userKeys).issubset(set(defaultKeys)):
      print('ERROR: arguments are not a subset of default parameters')
      raise
    #Update the parameters by user parameters 
    self._params.update(params)
    assert len(set(self._RESERVED_PARAM_NAMES_).intersection(set(defaultKeys))) == 0
    self._params['usertag'] = usertag
    #Initial the database collection
    expClient = get_mongo_client()
    self._expDb  = expClient[self.projectName]
    self._dbColl = self._expDb[self.name]
    #Check consistency
    self.userConfirm = userConfirm
    self.check_params_consistency() 

  @property
  def dbColl(self):
    """
    The collection storing the parameters on the server
    """
    return self._dbColl

  @property
  def name(self):
    """
    name of the object
    the database file name depends on this
    """
    return type(self).__name__

  @abc.abstractproperty
  def projectName(self):
    """
    Returns the name of the project
    """

  @abc.abstractproperty
  def ignoreHashKeys(self):
    """
    Returns:
      list of keys that should be ignored when
      generating or loading the hash
    """ 
 
  @abc.abstractmethod
  def default_params(self):
    """
    Returns:
      a dict containing default params
    """

  @property
  def params(self):
    """
    If new parameters are added, they are set to 
    the default value
    """
    params = copy.deepcopy(self._params)
    params['name'] = self.name
    return params 

  @property
  def hashableParams(self):
    """
    Remove keys in params that have been specified in 
    """
    params = copy.deepcopy(self.params)
    for k in self.ignoreHashKeys:
      if k in params:
        del params[k]
    return params

  @property
  def defaultHashableParams(self):
    """
    default parameter values after the removing the
    keys specifed in ignoreHashKeys
    """
    params = copy.deepcopy(self.default_params())
    params['name'] = self.name
    params['usertag'] = self._params['usertag']
    for k in self.ignoreHashKeys:
      if k in params:
        del params[k]
    return params

  def params_no_reserved(self):
    """Returns the params after stripping the reserved names"""
    params = self.params
    for k in self._RESERVED_PARAM_NAMES_:
      del params[k] 
    return params

  def is_safe_to_delete_key(self, k):
    """
    A key k is safe to delete only if all 
    the entries have the same value for key k
    otherwise it should not be deleted
    """
    cur = self.dbColl.find()
    keyVal = None
    for i, dat in enumerate(cur):
      if i == 0:
        keyVal = dat[k]
      if not keyVal == dat[k]:
        #print (keyVal, dat[k])
        return False
    return True
    
  def check_params_consistency(self):
    """
    mantain consistency of field names beween DB and default params
    it is possible a user can add a new param name
    we mantain the fact all entries have the same set
    of parameters for all entries
    """
    defParams = self.defaultHashableParams
    cursor    = self.dbColl.find()
    if cursor.count() == 0:
      return
    doc = cursor.next()
    #keys in the database
    dbKeys = set([str(k) for k in doc.keys()])
    #keys in the parameters
    prKeys = set(defParams.keys())
    prKeys.add('_id')
    isOld  =  dbKeys.issubset(prKeys) and prKeys.issubset(dbKeys) 
    if isOld:
      return
    if dbKeys.issubset(prKeys):
      print ('#### New default parameters found ... ####')
      #If new parameters are added
      newKeys = prKeys.difference(dbKeys)
      for k in newKeys:
        print ('Setting all entries in {0} DB with value {1} for key {2}'.\
               format(self.name, defParams[k], k))
        if self.userConfirm:
          try:
            #python2
            ip = raw_input('Proceed (Y/N): ')
          except:
            #python3
            ip = input('Proceed (Y/N): ')
        else: 
          ip = 'Y'
        assert ip == 'Y' or ip == 'N', 'Invalid Input, Enter Y/N'
        if ip == 'Y':
          docs = self.dbColl.find()
          for d in docs:
            newd = copy.deepcopy(d)
            newd.update({k: defParams[k]})
            self.dbColl.update(d, newd) 
          print ('Updated')
        else:
          print ('Not proceeding with update')
    else:
      print ('#### Some default parameters are deleted ... ####')
      newKeys = dbKeys.difference(prKeys)
      for k in newKeys:
        isSafe = self.is_safe_to_delete_key(k)
        if not isSafe:
          raise CustomError('Key %s cannot be deleted/ignored' % k)
        print ('For all entries in {0} DB, deleting key {1}'.\
               format(self.name, k))
        if self.userConfirm:
          try:
            #python2
            ip = raw_input('Proceed (Y/N): ')
          except:
            #python3
            ip = input('Proceed (Y/N): ')
        else: 
          ip = 'Y'
        assert ip == 'Y' or ip == 'N', 'Invalid Input, Enter Y/N'
        if ip == 'Y':
          docs = self.dbColl.find()
          for d in docs:
            newd = copy.deepcopy(d)
            del newd[k]
            self.dbColl.update(d, newd) 
          print ('Updated')
        else:
          print ('Not proceeding with update')

 
  def find_by_id(self, _id):
    """
    _id: the id by which the entry is to be found
    """
    if type(_id) is str:
      _id = bson.objectid.ObjectId(_id)
    assert type(_id) is bson.objectid.ObjectId 
    print(self.dbColl) 
    return self.dbColl.find({'_id': _id})

  def find_by_usertag(self, usertag):
    """
    usertag: the usertag for finding the entry
    """
    return self.dbColl.find({'usertag': usertag})

  def from_id(self, _id):
    """ 
    Return the entry by the _id
    """
    cur = self.find_by_id(_id)
    assert cur.count() == 1
    return cur.next()

  def delete_by_id(self, _id, userConfirm=True):
    cur = self.find_by_id(_id)
    for dat in cur:
      if userConfirm:
        print ('Deleting\n {0}'.format(dat))
        try:
          #python2
          ip = raw_input('Proceed (Y/N): ')
        except:
          ip = input('Proceed (Y/N): ')
        assert ip in ['Y', 'N']
        if ip == 'Y':
          self.dbColl.delete_one(dat)
          print ('Deleted')
        else:
          print ('Not Deleted') 
      else:
        self.dbColl.delete_one(dat)

  def delete_by_usertag(self, usertag, userConfirm=True):
    ent = self.find_by_usertag(usertag)
    assert ent.count() == 1
    ent = ent.next()
    self.delete_by_id(ent['_id'], userConfirm=userConfirm)

  def get_all_ids(self):
    """
    Returns the ids of all the entries
    """
    cur = self.dbColl.find()
    ids = []
    for dat in cur:
      ids.append(str(dat['_id']))
    return ids
    
  def update_key_name(self):
    print ('Setting all entries in {0} DB with value {1} for key {2}'.\
               format(self.name, defParams[k], k))
    if self.userConfirm:
      try:
        #python2
        ip = raw_input('Proceed (Y/N): ')
      except:
        #python3
        ip = input('Proceed (Y/N): ')
      assert ip == 'Y' or ip == 'N', 'Invalid Input, Enter Y/N'
    else:
      ip = 'Y'
    if ip == 'Y':
      docs = self.dbColl.find()
      for d in docs:
        newd = copy.deepcopy(d)
        del newd[k]
        self.dbColl.update(d, newd) 
      print ('Updated')
    else:
      print ('Not proceeding with update')
      
 
  def hash_name(self):
    cursor = self.dbColl.find(self.hashableParams)
    usertag = self.hashableParams['usertag']
    #mantain uniqueness of usertag
    if usertag is not None:
      utCursor = self.find_by_usertag(usertag)
      if cursor.count() == 0 and utCursor.count() >0:
        raise Exception('Usertag {0} already used'.format(usertag))
    if cursor.count() == 0:
      self.dbColl.insert_one(self.hashableParams)
      cursor = self.dbColl.find(self.hashableParams)
    assert cursor.count() == 1, 'Duplicates found'
    hashName = str(cursor.next()['_id'])
    return hashName

  def _is_equal(self, e1, e2):
    """
    Determines if entry e1 is equal to e2 or not
    """
    e1 = copy.deepcopy(e1)
    e2 = copy.deepcopy(e2)
    del e1['_id']
    del e2['_id']
    k1 = set(e1.keys())
    k2 = set(e2.keys())
    isEq = k1.issubset(k2) and k2.issubset(k1)
    if not isEq:
      return False
    for k in k1:
      isEq = isEq and (e1[k] == e2[k])
    return isEq

  #Debugging tool
  def _find_duplicates(self):
    """
    Determines if two entries have the exact same value. 
    In the usual functioning of the code this will never
    happen, but a user may introduce inconsistencies in the
    code which need to be resolved
    """
    allIds = self.get_all_ids()
    data   = [self.from_id(_id) for _id in allIds]
    #Iterate over the data to find if duplicate or not
    for i in range(len(data)):
      for j in range(i+1, len(data)):
        isEq = self._is_equal(data[i], data[j])
        if isEq:
          print ('Duplicates Found', allIds[i], allIds[j])
    print ('End of finding duplicates')

  #Debugging tool
  def _find_duplicate_id(self):
    """
    Returns id of the entries which are 
    the exact same entries
    """
    allIds = self.get_all_ids()
    data   = [self.from_id(_id) for _id in allIds]
    #Iterate over the data to find if duplicate or not
    for i in range(len(data)):
      for j in range(i+1, len(data)):
        isEq = self._is_equal(data[i], data[j])
        if isEq:
          return allIds[i]
    return None
    
  #Debugging tool
  def _remove_duplicates(self, userConfirm=True):
    while True:
      delId = self._find_duplicate_id()
      if delId is None:
        print ('All duplicates removed')
        return
      self.delete_by_id(delId, userConfirm=userConfirm)
        

