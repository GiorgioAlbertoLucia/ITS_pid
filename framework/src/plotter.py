'''
    Class to produce plots from given THn
'''

from ROOT import TCanvas, TFile, TH1F, TLine, TBox, TLegend
from ROOT import gStyle, gROOT

from .axis_spec import AxisSpec

class Plotter:

    def __init__(self, outPath):
        
        self.outFile = TFile(outPath, 'RECREATE')
        self.canvas = None
        self.hframe = None
        self.legend = None
        
        self.histDict = {}
        self.lineDict = {}
        self.boxDict = {}

    def createCanvas(self, axisSpecs: list, **kwargs):
        
        canvas_width = kwargs.get('canvas_width', 800)
        canvas_height = kwargs.get('canvas_height', 600)
        self.canvas = TCanvas(f'{axisSpecs[0]["name"]}_canvas', 'canvas', canvas_width, canvas_height)
        self.hframe = self.canvas.DrawFrame(axisSpecs[0]['xmin'], axisSpecs[1]['xmin'], axisSpecs[0]['xmax'], axisSpecs[1]['xmax'], axisSpecs[0]['title'])
    
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

        self.histDict[histLabel] = hist  
        self.canvas.cd()
        self.histDict[histLabel].Draw(kwargs.get('draw_option', 'SAME'))
    
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
        
        band = TBox(boxSpecs['x1'], boxSpecs['y1'], boxSpecs['x2'], boxSpecs['y2'])
        band.SetFillColorAlpha(kwargs.get('fill_color', 0), kwargs.get('fill_alpha', 1))
        band.SetFillStyle(kwargs.get('fill_style', 0))
        self.boxDict[lineSpecs['name']] = band
        
        self.canvas.cd()
        self.lineDict[lineSpecs['name']].Draw(kwargs.get('draw_option', 'SAME'))
        self.boxDict[lineSpecs['name']].Draw(kwargs.get('draw_option', 'SAME'))

    def addLegend(self, position, **kwargs):
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

        for histName, hist in self.histDict.items():
            self.legend.AddEntry(hist, histName, 'fl')
        for lineName, line in self.lineDict.items():
            self.legend.AddEntry(line, lineName, 'l')
        
        self.canvas.cd()
        self.legend.Draw(kwargs.get('draw_option', 'SAME'))

    def _reset(self):
        self.histDict = {}
        self.lineDict = {}
        self.boxDict = {}
        self.canvas.Clear()
        self.hframe = None
        self.legend = None

    def save(self, outPath:str):
        self.canvas.SaveAs(outPath)
        self.outFile.cd()
        self.canvas.Write()
        self._reset()
        
    def close(self):
        self.outFile.Close()
        
        



