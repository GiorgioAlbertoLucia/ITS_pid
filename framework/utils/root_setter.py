'''
    Single functions to set multiple features of ROOT objects
'''

from ROOT import gStyle, gROOT, TGraph


def obj_setter(obj, **kwargs):
    '''
        Set multiple properties of a ROOT object (the arguments might require that 
        the object has the corresponding methods)

        Parameters
        ----------
        obj: ROOT object
            ROOT object
        kwargs: dict
            properties to be set
    '''

    if 'name' in kwargs:
        obj.SetName(kwargs['name'])
    if 'title' in kwargs:
        obj.SetTitle(kwargs['title'])
    if 'line_color' in kwargs:
        obj.SetLineColor(kwargs['line_color'])
    if 'line_width' in kwargs:
        obj.SetLineWidth(kwargs['line_width'])
    if 'line_style' in kwargs:
        obj.SetLineStyle(kwargs['line_style'])
    if 'marker_color' in kwargs:
        obj.SetMarkerColor(kwargs['marker_color'])
    if 'marker_style' in kwargs:
        obj.SetMarkerStyle(kwargs['marker_style'])
    if 'marker_size' in kwargs:
        obj.SetMarkerSize(kwargs['marker_size'])
    if 'fill_color' in kwargs:
        obj.SetFillColor(kwargs['fill_color'])
    if 'fill_style' in kwargs:
        obj.SetFillStyle(kwargs['fill_style'])
    if 'fill_alpha' in kwargs and 'fill_color' in kwargs:
        obj.SetFillColorAlpha(kwargs['fill_color'], kwargs['fill_alpha'])

    return obj
