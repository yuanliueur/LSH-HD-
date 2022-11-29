# LSH-HD+ and ModelID Matching Method
This repository contains an enhanced method for duplicate detection in a product category. 

## What is included?
- TVs-all-merged.json : Used data set containing information on 1624 TVs in .json format
- data.py       : Includes all required functions for loading and cleaning the data
- Functions.py: : Includes all required functions for duplicate detection
- main.py       : Loop for generating results based on $n$ boostraps with replacement
- imports.py.   : Imports all required packages
- brandlist.txt : A brandlist of popular TV manufacturers taken from https://en.wikipedia.org/wiki/List_of_television_manufacturers

## How to run?
In main.py
- Set number of desired bootstraps
- Configure to desired results output
- Results will be stored in csv form (results.csv)

Note that the running time increases for a larger number of bootstraps.
