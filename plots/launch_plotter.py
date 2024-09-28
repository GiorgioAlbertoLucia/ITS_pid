import yaml

import sys
sys.path.append('..')
from framework.src.plotter import Plotter

if __name__ == '__main__':

    with open('../plot_config/cfg_efficiency_purity.yml', 'r') as f:
    #with open('../plot_config/cfg_clsize_vs_betagamma.yml', 'r') as f:
    #with open('../plot_config/cfg_clsize_vs_p.yml', 'r') as f:
    #with open('../plot_config/cfg_rocs.yml', 'r') as f:
        config = yaml.safe_load(f)

    plotter = Plotter(config['outPath'])

    for plot in config['plots']:
        
        plotter.createCanvas(plot['axisSpecs'], **plot['canvas'])
        plotter.createMultiGraph(plot['axisSpecs'])
        if plot['legend']['bool']:  
            position = [plot['legend']['xmin'], plot['legend']['ymin'], plot['legend']['xmax'], plot['legend']['ymax']]
            plotter.createLegend(position, **plot['legend']['kwargs'])

        if 'graphs' in plot:
            for graph in plot['graphs']:
                plotter.addGraph(graph['inPath'], graph['graphName'], graph['graphLabel'], **graph['kwargs'])
        if 'hists' in plot:
            for hist in plot['hists']:
                plotter.addHist(hist['inPath'], hist['histName'], hist['histLabel'], **hist['kwargs'])
        if 'funcs' in plot:
            for func in plot['funcs']:
                plotter.addFunc(func['inPath'], func['funcName'], func['funcLabel'], **func['kwargs'])

        if 'multigraph' in plot:
            plotter.drawMultiGraph(**plot['multigraph']['kwargs'])
        if plot['legend']['bool']: 
            plotter.drawLegend(**plot['legend']['kwargs'])

        plotter.save(plot['outPDF'])
    
    plotter.outFile.Close()