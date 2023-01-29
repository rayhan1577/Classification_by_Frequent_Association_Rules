# Classification by Frequent Association rules
This repo contains the implementation of paper: ”Classification by Frequent Association Rules”, accepted in The 38th ACM/SIGAPP Symposium on Applied Computing (SAC ’23), March 27 - March 31, 2023, Tallinn, Estonia.

# Set Up
```
# Setup python virtual environment
$ virtualenv venv --python=python3
$ source venv/bin/activate


# Install python dependencies
$ pip3 install  -r requirements.txt 

```
# Execution
To run on terminal: 
```
python3 CFAR.py
```
for different dataset change at Line number 56.
To print the rules of base learners, at line 129, change the last parameter as "True" 
Rules produced by the base learners are saved in savedrules.txt.
To print the rules of CFAR, at line 153, change the last parameter as "True"

# Data
Given dataset was used for the project

# Contributor
---
- Rayhan Kabir (rayhan.kabir@ualberta.ca)
---
