'''
    Class to produce plots from given THn
'''

from ROOT import TCanvas, TFile, TH1F, TLine, TBox, TLegend, TMultiGraph, TGraph
from ROOT import gStyle, gROOT

from .axis_spec import AxisSpec

class Plotter:

    def __init__(self, outPath):
        
        self.outFile = TFile(outPath, 'RECREATE')
        self.canvas = None
        self.hframe = None
        self.legend = None
        self.multigraph = None
        
        self.histDict = {}
        self.graphDict = {}
        self.lineDict = {}
        self.funcDict = {}
        self.boxDict = {}

        gStyle.SetOptStat(0)

    def createCanvas(self, axisSpecs: list, **kwargs):
        
        canvas_width = kwargs.get('canvas_width', 800)
        canvas_height = kwargs.get('canvas_height', 600)
        self.canvas = TCanvas(f'{axisSpecs[0]["name"]}_canvas', 'canvas', canvas_width, canvas_height)
        if kwargs.get('logy', False):   self.canvas.SetLogy()
        if kwargs.get('logz', False):   self.canvas.SetLogz()
        if 'right_margin' in kwargs:    self.canvas.SetRightMargin(kwargs['right_margin'])
        if 'left_margin' in kwargs:     self.canvas.SetLeftMargin(kwargs['left_margin'])
        if 'top_margin' in kwargs:      self.canvas.SetTopMargin(kwargs['top_margin'])
        if 'bottom_margin' in kwargs:   self.canvas.SetBottomMargin(kwargs['bottom_margin'])
        self.hframe = self.canvas.DrawFrame(axisSpecs[0]['xmin'], axisSpecs[1]['xmin'], axisSpecs[0]['xmax'], axisSpecs[1]['xmax'], axisSpecs[0]['title'])

    def createMultiGraph(self, axisSpecs: list, **kwargs):

        self.multigraph = TMultiGraph(f'{axisSpecs[0]["name"]}_mg', axisSpecs[0]["title"])

    def drawMultiGraph(self, **kwargs):

        self.canvas.cd()
        self.multigraph.Draw(kwargs.get('draw_option', 'SAME'))
    
    def addHist(self, inPath:str, histName:str, histLabel:str, **kwargs):

        inFile = TFile(inPath, 'READ')
        hist = inFile.Get(histName)

        hist.SetDirectory(0)
        hist.SetLineColor(kwargs.get('line_color', 1))
        hist.SetMarkerColor(kwargs.get('marker_color', 1))
        hist.SetMarkerStyle(kwargs.get('marker_style', 20))
        hist.SetMarkerSize(kwargs.get('marker_size', 1))
        hist.SetLineWidth(kwargs.get('line_width', 1))
        hist.SetLineStyle(kwargs.get('line_style', 1))
        hist.SetFillColorAlpha(kwargs.get('fill_color', 0), kwargs.get('fill_alpha', 1))
        hist.SetFillStyle(kwargs.get('fill_style', 0))
        gStyle.SetPalette(kwargs.get('palette', 1))

        self.histDict[histLabel] = hist  
        if kwargs.get('leg_add', True) and self.legend is not None: self.legend.AddEntry(self.histDict[histLabel], histLabel, kwargs.get('leg_option', 'fl'))
        self.canvas.cd()
        self.histDict[histLabel].Draw(kwargs.get('draw_option', 'SAME'))
    
        inFile.Close()

    def addGraph(self, inPath:str, graphName:str, graphLabel:str, **kwargs):

        inFile = TFile(inPath, 'READ')
        graph = inFile.Get(graphName)

        graph.SetFillColorAlpha(kwargs.get('fill_color', 0), kwargs.get('fill_alpha', 1))
        graph.SetFillStyle(kwargs.get('fill_style', 0))
        graph.SetLineColor(kwargs.get('line_color', 1))
        graph.SetMarkerColor(kwargs.get('marker_color', 1))
        graph.SetMarkerStyle(kwargs.get('marker_style', 20))
        graph.SetMarkerSize(kwargs.get('marker_size', 1))
        graph.SetLineWidth(kwargs.get('line_width', 1))
        graph.SetLineStyle(kwargs.get('line_style', 1))

        self.graphDict[graphLabel] = graph  
        if kwargs.get('leg_add', True) and self.legend is not None: self.legend.AddEntry(self.graphDict[graphLabel], graphLabel, kwargs.get('leg_option', 'p'))
        self.multigraph.Add(self.graphDict[graphLabel], kwargs.get('draw_option', 'SAME'))

        inFile.Close()

    def addFunc(self, inPath:str, funcName:str, funcLabel:str, **kwargs):
        '''
            Add a TF1 function to the plot
            
            func: TF1
            funcName: str
            funcLabel: str
        '''

        inFile = TFile(inPath, 'READ')
        func = inFile.Get(funcName)
        
        func.SetLineColor(kwargs.get('line_color', 1))
        func.SetLineWidth(kwargs.get('line_width', 1))
        func.SetLineStyle(kwargs.get('line_style', 1))
        self.funcDict[funcName] = func
        if kwargs.get('leg_add', True) and self.legend is not None: self.legend.AddEntry(self.funcDict[funcName], funcLabel, kwargs.get('leg_option', 'l'))
        
        self.canvas.cd()
        self.funcDict[funcName].Draw(kwargs.get('draw_option', 'SAME'))

        inFile.Close()

    def addROI(self, lineSpecs: dict, boxSpecs: dict, **kwargs):
        '''
            Draw a line between point 1 and 2 and a color band around it
            
            lineSpecs: dict 
                    x1, y1, x2, y2: float
                    name: str  
            boxSpecs: dict
                x1, y1, x2, y2: float
                    coordinates of the color band
        '''
        
        line = TLine(lineSpecs['x1'], lineSpecs['y1'], lineSpecs['x2'], lineSpecs['y2'])
        line.SetLineColor(kwargs.get('line_color', 1))
        line.SetLineWidth(kwargs.get('line_width', 1))
        line.SetLineStyle(kwargs.get('line_style', 1))
        self.lineDict[lineSpecs['name']] = line
        if kwargs.get('leg_add', True) and self.legend is not None: self.legend.AddEntry(line, lineSpecs['name'], kwargs.get('leg_option', 'l'))
        
        band = TBox(boxSpecs['x1'], boxSpecs['y1'], boxSpecs['x2'], boxSpecs['y2'])
        band.SetFillColorAlpha(kwargs.get('fill_color', 0), kwargs.get('fill_alpha', 1))
        band.SetFillStyle(kwargs.get('fill_style', 0))
        self.boxDict[lineSpecs['name']] = band
        
        self.canvas.cd()
        self.lineDict[lineSpecs['name']].Draw(kwargs.get('draw_option', 'SAME'))
        self.boxDict[lineSpecs['name']].Draw(kwargs.get('draw_option', 'SAME'))

    def createLegend(self, position, **kwargs):
        ''' 
            position: list
                x1, y1, x2, y2: float
            kwargs: dict
                header: str
                border_size: int
                fill_color: int
                fill_style: int
        '''
        
        self.legend = TLegend(position[0], position[1], position[2], position[3])
        self.legend.SetHeader(kwargs.get('header', ''))
        self.legend.SetBorderSize(kwargs.get('border_size', 0))
        #legend.SetFillColor(kwargs.get('fill_color', 0))
        #legend.SetFillStyle(kwargs.get('fill_style', 0))
        #legend.SetTextSize(kwargs.get('text_size', 0.03))

        nColumns = kwargs.get('nColumns', 0)
        if nColumns != 0:
            self.legend.SetNColumns(nColumns)
    
    def drawLegend(self, **kwargs):
        self.canvas.cd()
        self.legend.Draw(kwargs.get('draw_option', 'SAME'))

    def _reset(self):
        self.histDict = {}
        self.graphDict = {}
        self.lineDict = {}
        self.funcDict = {}
        self.boxDict = {}
        self.canvas.Clear()
        self.hframe = None
        if self.legend is not None: self.legend.Clear()
        self.legend = None
        self.multigraph = None 

    def save(self, outPath:str):
        self.canvas.SaveAs(outPath)
        self.outFile.cd()
        self.canvas.Write()
        self._reset()
        
    def close(self):
        self.outFile.Close()
        
        



