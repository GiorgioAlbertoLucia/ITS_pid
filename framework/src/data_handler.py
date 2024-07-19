'''
    Classes to manage input data
'''

from abc import ABC, abstractmethod

import uproot
from hipe4ml.tree_handler import TreeHandler

from ..utils.terminal_colors import TerminalColors as tc

class DataHandler:

    def __init__(self):
        return NotImplemented
    
    @abstractmethod
    def __open(self):
        return NotImplemented


class TableHandler(DataHandler): 
    '''
        Class to open data from AO2D.root files generated with a O2Physics table producer
    '''
    def __init__(self, inFilePath: str, treeName: str, dirPrefix: str):

        self.inFilePath = inFilePath
        self.inData = self.__open(inFilePath, treeName, dirPrefix)

    def __open(self, inFilePath: str, treeName: str, dirPrefix: str):

        if inFilePath.endswith('.root'):
            
            print('Opening '+tc.UNDERLINE+tc.CYAN+f'{inFilePath}'+tc.RESET+'...')
            print('Using tree '+tc.GREEN+f'{treeName}'+tc.RESET+' and directory prefix '+tc.GREEN+f'{dirPrefix}'+tc.RESET+'\n')
            th = TreeHandler(inFilePath, treeName, folder_name=dirPrefix)
            return th.get_data_frame()

        else:   raise ValueError('File extension not supported')






class TaskHandler(DataHandler):
    '''
        Class to open data from AO2D.root files generated with a O2Physics task
    '''

    def __init__(self, inFilePath: str, mainDir: str):

        self.inFilePath = inFilePath
        self.mainDir = mainDir
        self.inData = self.__open(inFilePath)[mainDir]

    def __open(self, inFilePath):

        if inFilePath.endswith('.root'):
            
            print('Opening '+tc.UNDERLINE+tc.CYAN+f'{inFilePath}'+tc.RESET+'...')
            print('Using main directory '+tc.GREEN+f'{self.mainDir}'+tc.RESET+'\n')
            return uproot.open(inFilePath)

        else:   raise ValueError('File extension not supported')


