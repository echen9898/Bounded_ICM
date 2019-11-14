# Installation


```
sudo apt-get install mongodb-server
```
If the server is configured with a password, one can put the credentials in `db_config.py`. Note that `db_config.py` is
not version controlled for security reasons. A sample script, `sample_db_config.py` is provided.
Use `cp sample_db_config.py db_config.py` and set approrpriate authentication mechanism. 

# Running

First start a mongo server, 
```
mongod --port=27900 --dbpath=/path/to/db/store
```

Alternatively, options can be specified in `/etc/mongod.conf` and the
mongod server can be started as

```
mongod --config /etc/mongod.conf
```

In case, mongodb is run as a different user, use

```
sudo -H -u otheruser bash -c 'mongod --config /etc/mongod.conf' 
```

For enabling authorization, in the `.conf` file specify:
```
security
  authorization: enabled
```

In case you are using the conf file and cannot connect from 
other machines (other than the localhost), comment out `bindIp` in the conf file. 

Then connect with the server,
```python
import pymongo
client = pymongo.MongoClient('locahost:27900')
```

Relevant links
```
http://stackoverflow.com/questions/37014186/running-mongodb-on-ubuntu-16-04-lts
```

# Automatic Testing
On the `ci` branch. 

To install code for running the tests automatically before a commit run
```
./scripts/install-hooks.bash
```

# Giving user permissions

Login into the mongo shell. 
```
mongo --port 27900 -u "myUserAdmin" -p "abc123" --authenticationDatabase "admin"
#See roles of a user
use db db_permissions
db.getUser("username")
```
Look at this [link](https://docs.mongodb.com/manual/tutorial/manage-users-and-roles/#grant-a-role) to grant
permissions to a user. 


