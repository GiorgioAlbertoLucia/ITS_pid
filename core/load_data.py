'''
    Functions to load data from a file based on the file extension.
'''


import os
import polars as pl
import uproot
from hipe4ml.tree_handler import TreeHandler

import sys 
sys.path.append('..')
sys.path.append('../..')
from framework.utils.terminal_colors import TerminalColors as tc

def LoadData(inFiles:list, **kwargs) -> pl.DataFrame:
    '''
    Load data from multiple files

    Parameters
    ----------
    inFiles (list): list of input files  
    '''

    df = pl.DataFrame()
    for inFile in inFiles:
        df_tmp = LoadDataFile(inFile, **kwargs)
        if df_tmp is not None:  df = pl.concat([df, df_tmp])

    return df

def LoadDataFile(inFile:str, **kwargs):
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
    if inFile.endswith(".root") and "AO2D" in inFile:   df = LoadAO2D(inFile, **kwargs)
    elif inFile.endswith(".root"):                        df = LoadRoot(inFile)   
    elif inFile.endswith(".parquet"):                   df = LoadParquet(inFile)
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

def LoadAO2D(inFile:str, **kwargs) -> pl.DataFrame:
    '''
    Load data from an AO2D file

    Parameters
    ----------
    inFile (str): input file
    '''

    tree_name = kwargs.get('tree_name', "O2clsttable")
    folder_name = kwargs.get('folder_name', "DF_*")
    th = TreeHandler(inFile, tree_name, folder_name=folder_name)
    df = th.get_data_frame()
    return pl.from_pandas(df)
    

def LoadParquet(inFile:str) -> pl.DataFrame:
    '''
    Load data from a parquet file

    Parameters
    ----------
    inFile (str): input file
    '''
    
    df = pl.read_parquet(inFile)
    
    return df