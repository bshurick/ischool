# DS241_VideoTextMath

## Data ##

The following files in the data folder.

1. **CombinatoricsPilot.csv** - First Survey results (101)
2. **CombinatoricsPilot_08122015.csv** - Second Survey Results (50)
3. **CombinatoricsPilot-Responses in Progress.csv**  - incomplete responses (113)

## Other ##

1. **ColumnMapping.xlsx** - Excel file used to track column mappings of the data files.


## Rcode ##

1. **exp_analysis.R**
    - Covariate Analysis

2. **vid_text_math.R** 
    - Simple ATE Calculations. 
    - Depends on the vidmathDatamapper.R file.
    
3. **vid_text_math_Combined.R** 
    - Regression. 
    - Heterogeneous Treatment Effects
   
4. **vidmathDatamapper.R**  
    - Maps the raw datasets to user friendly column names
    - Extracts key analysis columns into functions.
    
    - **Renamecols** : 
	    - Take as input a dataframe from the raw survey 1 file **"CombinatoricsPilot.csv"**
	    - Function to rename dataframe to userfriendly columns 
	     
    - **RenamecolsNoplacebo** :
    
	    - Function to rename dataframe to userfriendly columns 
	    - Take as input a dataframe from the raw survey 1 file **"CombinatoricsPilot_08122015.csv"**
	          
    - **GetAnalysisSubset1** : 
    
	    - Extract keys columns for analysis
	    - Adds result column for each question (pre & Post).
	    - Also adds columns for prescore and postscore.
       

## Results ##

