## A9 Group Repository

# Set up 
`conda env create -f environment.yml`

# Create Predictions
```powershell
#invoke the test script to get predictions
python test.py dir/with/focal/stack 

# optionally specify output directory, otherwise its the same as input directory
python test.py dir/with/focal/stack --output_dir dir/to/save/result
```

The focal planes must be provided in a folder. The folder must contain 6 files, named:
```
'000.png'  
'040.png'
'080.png'
'120.png'
'160.png'
'200.png'
```
Other files are ignored. If some files are duplicated, behaviour is undefined.