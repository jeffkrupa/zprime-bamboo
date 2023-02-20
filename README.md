# bamboo
bamboo for zprime run2

First install bamboo: https://bamboo-hep.readthedocs.io/en/latest/install.html#fresh-install

Then install plotIt: ```pip install git+https://gitlab.cern.ch/cp3-cms/pyplotit.git```

```cd ../bamboo/examples/
git clone git@github.com:jeffkrupa/zprime-bamboo.git
cd zprime-bamboo
```

Each time, do 
```
cd zprime-bamboo/
. setup.sh
```

Running locally:
```
bambooRun  -m  zprlegacy.py:zprlegacy samples_2017_jet.yml  --envConfig=../cern.ini -o test --mvaSkim --maxFile 1
```

Running distributed: 
```
bambooRun  -m  zprlegacy.py:zprlegacy samples_2017_jet.yml  --envConfig=../cern.ini -o test --mvaSkim --maxFile -1 --distributed=driver
```
To check jobs/run plots:
```
bambooRun  -m  zprlegacy.py:zprlegacy samples_2017_jet.yml  --envConfig=../cern.ini -o test --mvaSkim --maxFile -1 --distributed=finalize
```




