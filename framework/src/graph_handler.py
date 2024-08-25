'''
    Class to create a TGraph or TGraphErrors from a polars DataFrame
'''

import numpy as np
import ctypes
import polars as pl
from ROOT import TGraph, TGraphErrors


class GraphHandler:

    def __init__(self, df: pl.DataFrame):
        '''
            Initialize the class

            Parameters
            ----------
            df (pl.DataFrame): input DataFrame
        '''
        self.df = df

    def createTGraph(self, x: str, y: str) -> TGraph:
        '''
            Create a TGraph from the input DataFrame

            Parameters
            ----------
            x (str): x-axis variable
            y (str): y-axis variable
        '''
        # eliminate None values on x and y
        self.df = self.df.filter(self.df[x].is_not_null())
        self.df = self.df.filter(self.df[y].is_not_null())

        if len(self.df) == 0:
            return TGraph()
        graph = TGraph(len(self.df[x]))
        for irow, row in enumerate(self.df.iter_rows(named=True)):
            if irow == 0:
                graph = TGraph()
            graph.SetPoint(irow, row[x], row[y])

        #x_values = self.df[x].to_numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #y_values = self.df[y].to_numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #graph = TGraph(len(self.df[x]), x_values, y_values)

        return graph
    
    def createTGraphErrors(self, x: str, y: str, ex, ey) -> TGraphErrors:
        '''
            Create a TGraphErrors from the input DataFrame

            Parameters
            ----------
            x (str): x-axis variable
            y (str): y-axis variable
            ex (str): x-axis error
            ey (str): y-axis error
        '''

        # eliminate None values on x, y
        self.df = self.df.filter(self.df[x].is_not_null())
        self.df = self.df.filter(self.df[y].is_not_null())

        #x_values = self.df[x].to_numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #y_values = self.df[y].to_numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #if ex == 0:
        #    ex_values = np.zeros(len(self.df[x])).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #else:
        #    ex_values = self.df[ex].to_numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #if ey == 0:
        #    ey_values = np.zeros(len(self.df[y])).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #else:
        #    ey_values = self.df[ey].to_numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #graph = TGraphErrors(len(self.df[x]), x_values, y_values, ex_values, ey_values)
        
        if len(self.df) == 0:
            return TGraphErrors()
        graph = TGraphErrors(len(self.df[x]))
        for irow, row in enumerate(self.df.iter_rows(named=True)):
            graph.SetPoint(irow, row[x], row[y])
            xerr = row[ex] if ex != 0 else 0.
            yerr = row[ey] if ey != 0 else 0.
            graph.SetPointError(irow, xerr, yerr)

        return graph
