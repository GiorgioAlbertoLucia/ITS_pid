'''
    Functions to load data from a file based on the file extension.
'''


import os
import polars as pl
import uproot

import sys 
sys.path.append('..')
sys.path.append('../..')
from framework.utils.terminal_colors import TerminalColors as tc

def LoadData(inFiles:list) -> pl.DataFrame:
    '''
    Load data from multiple files

    Parameters
    ----------
    inFiles (list): list of input files  
    '''

    df = pl.DataFrame()
    for inFile in inFiles:
        df_tmp = LoadDataFile(inFile)
        if df_tmp is not None:  df = pl.concat([df, df_tmp])

    return df

def LoadDataFile(inFile:str):
    '''
    Load data from a single file
    
    Parameters
    ----------
    inFile (str): input file  
    '''

    # check if the file exists
    if not os.path.exists(inFile):  
        print("File not found: "+tc.UNDERLINE+tc.RED+f'{inFile}'+tc.RESET)
        return None
    
    print("Loading data from: "+tc.UNDERLINE+tc.BLUE+f'{inFile}'+tc.RESET)
    if inFile.endswith(".root"):        df = LoadRoot(inFile)   
    elif inFile.endswith(".parquet"):   df = LoadParquet(inFile)
    else:
        print("Unknown file type: "+tc.UNDERLINE+tc.RED+f'{inFile}'+tc.RESET)
        return None

    return df

def LoadRoot(inFile:str) -> pl.DataFrame:
    '''
    Load data from a ROOT file

    Parameters
    ----------
    inFile (str): input file
    '''
    
    tree = uproot.open(inFile)["outTree"]
    df = tree.arrays(library="pd", how="zip")
    df = pl.from_pandas(df)

    return df

def LoadParquet(inFile:str) -> pl.DataFrame:
    '''
    Load data from a parquet file

    Parameters
    ----------
    inFile (str): input file
    '''
    
    df = pl.read_parquet(inFile)
    
    return df