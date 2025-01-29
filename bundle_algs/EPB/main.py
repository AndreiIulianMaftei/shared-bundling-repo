from collections import defaultdict
import os
import networkx as nx
import numpy as np
from epb import EPB, EPB_Biconn
from experiments import Experiment
from abstractBundling import GWIDTH, GraphLoader
from sepb import SpannerBundling, SpannerBundlingAStar, SpannerBundlingFG, SpannerBundlingFGSSP, SpannerBundlingNoSP, SpannerBundlingNoSPWithWF
from straight import StraightLine
from reader import Reader
import sys
import gc

def effectivenessExperiments(base, datasets, systems, directed):
    expData = {}
    expDataRev = defaultdict(dict)

    for input, invertY in datasets:

        G = Reader.readGraphML(f'{base}/datasets/{input}.graphml', G_width=GWIDTH, invertY=invertY, directed=directed)

        if G == None:
            continue

        if directed:
            outPath = f'{base}/output/{input}/directed/'
            if not os.path.exists(outPath):
                os.makedirs(outPath)           
        else:
            outPath = f'{base}/output/{input}/undirected/'
            if not os.path.exists(outPath):
                os.makedirs(outPath)

        if not os.path.exists(outPath + 'pickle/'):
            os.makedirs(outPath + 'pickle/')

        expData[input] = {}

        #First create straight line drawing
        straight = StraightLine(G)
        straight.bundle()
        straight.draw(outPath)

        exp = Experiment(straight, straight)
        data = exp.run(outPath)
        expData[input][exp.name] = data
        expDataRev[exp.name][input] = data

        for system, settings in systems:
            bundling = system(G)

            if settings:
                for key, value in settings.items():
                    setattr(bundling, key, value)
                
            if isinstance(bundling, GraphLoader):
                if directed:
                    path = f'{base}/datasets/{input}/directed/'
                else:
                    path = f'{base}/datasets/{input}/undirected/'
                setattr(bundling, 'filepath', path)
                
            bundling.bundle()
            bundling.draw(outPath)

            exp = Experiment(bundling, straight)
            data = exp.run(outPath)
            expData[input][exp.name] = data
            expDataRev[exp.name][input] = data


    for instance, experiments in expData.items():
        print(instance)

        for system, values in experiments.items():
            print(f'{system}: {values}')
            
        formater = "{:.10f}".format   
        for system, values in experiments.items():
            print("\n", system, end=" ; ")
            print(formater(values[0]), end =" ; ")
            print(formater(values[1][1]), end =" ; ")
            print(formater(values[1][2]), end =" ; ")
            print(formater(values[1][3]), end =" ; ")
            print(formater(values[1][4]), end =" ; ")
            print(formater(values[2][0]), end =" ; ")
            print(formater(values[2][1]), end =" ; ")
            print(formater(values[2][2]), end =" ; ")
            print(formater(values[2][3]), end =" ; ")
            print(formater(values[2][4]), end ="")
        
        print("\n")
        formater = "{:.2f}".format   
        for system, values in experiments.items():
            print("\n", system, end=" & ")
            print(formater(values[0]), end =" & ")
            print(formater(values[1][1]), end=" & ")
            #print(formater(values[1][2]), end=" & ")
            #print(formater(values[1][3]), end=" & ")
            print(formater(values[1][4]), end=" & ")
            print(formater(values[2][0]), end=" & ")
            print(formater(values[2][1]), end=" & ")
            print(formater(values[2][2]), end=" & ")
            print(formater(values[2][3]), end=" & ")
            print(formater(values[2][4]), end ="")
            
        print("\n")
             


def plotting(base, datasets, systems, directed, printTrl=False):
   for input, invertY in datasets:

        G = Reader.readGraphML(f'{base}/datasets/{input}.graphml', G_width=GWIDTH, invertY=invertY, directed=directed)

        if G == None:
            continue

        if directed:
            outPath = f'{base}/output/{input}/directed/'
            if not os.path.exists(outPath):
                os.makedirs(outPath)           
        else:
            outPath = f'{base}/output/{input}/undirected/'
            if not os.path.exists(outPath):
                os.makedirs(outPath)

        #First create straight line drawing
        straight = StraightLine(G)
        
        if not printTrl:
            straight.bundle()
            straight.draw(outPath)

        for system, settings in systems:
            bundling = system(G)

            if settings:
                for key, value in settings.items():
                    setattr(bundling, key, value)
                
            if isinstance(bundling, GraphLoader):
                if directed:
                    path = f'{base}/datasets/{input}/directed/'
                else:
                    path = f'{base}/datasets/{input}/undirected/'
                setattr(bundling, 'filepath', path)
                
            time = bundling.bundle()
            print(f"{bundling.name}: {time}", flush=True)
            if not printTrl:
                bundling.draw(outPath)
            else:
                print(f"{bundling.name}: {time}", flush=True)
                bundling.drawTrl(outPath)
            


def runtimeExperiments(base, datasets, systems, iterations, directed):
    runtime = {}

    #print(f"running experiment on graph: {datasets} {systems} {iterations}", flush=True)
    for input, invertY in datasets:
        #print(f"running experiment on graph: {input}", flush=True)
        G = Reader.readGraphML(f'{base}/datasets/{input}.graphml', G_width=1600, invertY=invertY, directed=directed)

        if G == None:
            continue

        if directed:
            outPath = f'{base}/output/{input}/directed/'
            if not os.path.exists(outPath):
                os.makedirs(outPath)           
        else:
            outPath = f'{base}/output/{input}/undirected/'
            if not os.path.exists(outPath):
                os.makedirs(outPath)

        writePath = outPath + 'experiments/'
        if not os.path.exists(writePath):
            os.makedirs(writePath)

        #First create straight line drawing
        straight = StraightLine(G)
        straight.bundle()
        #straight.draw(outPath)

        runtime[input] = defaultdict(list)

        #now run the experiments on the bunlding itself. track time.
        for system, settings in systems:
            bundling = system(G)

            if settings:
                for key, value in settings.items():
                    setattr(bundling, key, value)
                
            for i in range(iterations):
                bTime = bundling.bundle()
                
                runtime[input][bundling.name].append(bTime)

            #print(f'{bundling.name}: {np.mean(runtime[input][bundling.name]):.4f}, {np.median(runtime[input][bundling.name]):.4f}' , flush=True)
            del bundling
            gc.collect()

            #bundling.draw(outPath)
                
    for instance, runtimes in runtime.items():
        print(instance, flush=True)

        for system, values in runtimes.items():
            print(f'{system}: {np.mean(values):.4f}, {np.median(values):.4f}', flush=True)

            if directed:
                with open(f"{base}/output/{instance}/directed/experiments/{system}.txt", 'a+') as f:
                    f.write(" ".join(str(item) for item in values) + " ")
            if not directed:
                with open(f"{base}/output/{instance}/undirected/experiments/{system}.txt", 'a+') as f:
                    f.write(" ".join(str(item) for item in values) + " ")

# def checkDensity():
#     G = nx.complete_graph(1)

#     print(len(G.edges()))

#     for u,v,data in G.edges(data=True):
#         data['dist'] = np.random.randint(1, 2000)

#     T = nx.spanner(G, 4.0)

#     print(len(T.edges()))

def plotLatexTable(base, datasets, systems, directed):

    exp = defaultdict(dict)
    #print(f"running experiment on graph: {datasets} {systems} {iterations}", flush=True)
    for input, invertY in datasets:
        
        d = "directed" if directed else 'undirected'
        path = f'{base}/output/{input}/{d}/experiments/'

        straight = StraightLine(nx.Graph())

        for system, settings in systems:
            bundling = system(nx.Graph())
            
            if settings:
                for key, value in settings.items():
                    setattr(bundling, key, value)
            
            with open(f'{path}{bundling.name}.txt', 'r') as f:
                lines = f.readlines()
                values = []
                
                for line in lines:
                    vals = line.split(" ")
                    
                    for val in vals:
                        try:
                            val = float(val)
                            values.append(val)
                        except:
                            continue
                        
                exp[bundling.name][input] = values
            
    formater = "{:.3f}".format
    print("\n\nmean")
    for key, entry in exp.items():
        print("\n", key, end=" & ")
        for ds, values in entry.items():
             print(formater(np.mean(values)), end =" & ")
             
    print("\n\nmedian")
    for key, entry in exp.items():
        print("\n", key, end=" & ")
        for ds, values in entry.items():
             print(formater(np.median(values)), end =" & ")
             
    print("\n\nmean and median")
    for key, entry in exp.items():
        print("\n", key, end=" & ")
        for ds, values in entry.items():
             print(formater(np.mean(values)), " & ", formater(np.median(values)), end =" & ")

def experiments(base, numExp):
    
    print(f"running experiment: {numExp}", flush=True)
    numRuns = 25
    
    #Runtime experiment between EPB and EPB with Biconnected component decomposition
    if numExp == "1":
        directed = False
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(EPB, None), (EPB_Biconn, {'numWorkers' : 1, 'distortion':2})]
        
        runtimeExperiments(base, datasets, systems, numRuns, directed)
        
    #Runtime experiment with different S-EPB variants.
    if numExp == "2":
        directed = False
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(SpannerBundling, None),(SpannerBundlingFG, None), (SpannerBundlingAStar, None), (SpannerBundlingNoSPWithWF, None), (SpannerBundlingNoSP, None)]
        
        runtimeExperiments(base, datasets, systems, numRuns, directed)
        
    #Runtime experiment with different distortion values for EPB
    if numExp == "3":
        directed = False
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(EPB_Biconn, {'numWorkers' : 2, 'distortion':1.5}), (EPB_Biconn, {'numWorkers' : 2, 'distortion':2.5}), (EPB_Biconn, {'numWorkers' : 2, 'distortion':3}), (EPB_Biconn, {'numWorkers' : 2, 'distortion':4}), (EPB_Biconn, {'numWorkers' : 2, 'distortion':5})]
        
        runtimeExperiments(base, datasets, systems, numRuns, directed)

    #Runtime experiment with different distortion values for S-EPB
    if numExp == "4":
        directed = False
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(SpannerBundling, {'distortion':1.5}), (SpannerBundling, {'distortion':2.5}), (SpannerBundling, {'distortion':3}), (SpannerBundling, {'distortion':4}), (SpannerBundling, {'distortion':5})]
        
        runtimeExperiments(base, datasets, systems, numRuns, directed)
               
    #Runtime experiment with different number of worker threads for EPB
    if numExp == "5":
        directed = False
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(EPB_Biconn, {'numWorkers' : 2, 'distortion':1}), (EPB_Biconn, {'numWorkers' : 2, 'distortion':2}), (EPB_Biconn, {'numWorkers' : 4, 'distortion':2}), (EPB_Biconn, {'numWorkers' : 8, 'distortion':2})]
        
        runtimeExperiments(base, datasets, systems, numRuns, directed)
        
    #Runtime experiment with different number of worker threads for S-EPB
    if numExp == "5":
        directed = False
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(SpannerBundling, {'numWorkers' : 1, 'distortion':1}), (SpannerBundling, {'numWorkers' : 2, 'distortion':2}), (SpannerBundling, {'numWorkers' : 4, 'distortion':2}), (SpannerBundling, {'numWorkers' : 8, 'distortion':2})]
        
        runtimeExperiments(base, datasets, systems, numRuns, directed)
        
    #Directed graph experiments
    #Runtime experiment between EPB and EPB with Biconnected component decomposition
    if numExp == "6":
        directed = True
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(EPB, None), (EPB_Biconn, {'numWorkers' : 1, 'distortion':2})]
        
        runtimeExperiments(base, datasets, systems, numRuns, directed)
        
    #Runtime experiment with different S-EPB variants.
    if numExp == "7":
        directed = True
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(SpannerBundling, None), (SpannerBundlingFG, None), (SpannerBundlingAStar, None), (SpannerBundlingNoSPWithWF, None), (SpannerBundlingNoSP, None)]
        
        runtimeExperiments(base, datasets, systems, numRuns, directed)
        
    #Runtime experiment with different distortion values for EPB
    if numExp == "8":
        directed = True
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(EPB_Biconn, {'numWorkers' : 2, 'distortion':1.5}), (EPB_Biconn, {'numWorkers' : 2, 'distortion':2.5}), (EPB_Biconn, {'numWorkers' : 2, 'distortion':3}), (EPB_Biconn, {'numWorkers' : 2, 'distortion':4}), (EPB_Biconn, {'numWorkers' : 2, 'distortion':5})]
        
        runtimeExperiments(base, datasets, systems, numRuns, directed)

    #Runtime experiment with different distortion values for S-EPB
    if numExp == "9":
        directed = True
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(SpannerBundling, {'distortion':1.5}), (SpannerBundling, {'distortion':2.5}), (SpannerBundling, {'distortion':3}), (SpannerBundling, {'distortion':4}), (SpannerBundling, {'distortion':5})]
        
        runtimeExperiments(base, datasets, systems, numRuns, directed)
               
    #Runtime experiment with different number of worker threads for EPB
    if numExp == "10":
        directed = True
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(EPB_Biconn, {'numWorkers' : 2, 'distortion':1}), (EPB_Biconn, {'numWorkers' : 2, 'distortion':2}), (EPB_Biconn, {'numWorkers' : 4, 'distortion':2}), (EPB_Biconn, {'numWorkers' : 8, 'distortion':2})]
        
        runtimeExperiments(base, datasets, systems, numRuns, directed)
        
    #Runtime experiment with different number of worker threads for S-EPB
    if numExp == "11":
        directed = True
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(SpannerBundling, {'numWorkers' : 1, 'distortion':1}), (SpannerBundling, {'numWorkers' : 2, 'distortion':2}), (SpannerBundling, {'numWorkers' : 4, 'distortion':2}), (SpannerBundling, {'numWorkers' : 8, 'distortion':2})]
        
        runtimeExperiments(base, datasets, systems, numRuns, directed)
              
    #Runtime experiments for the large datasets. Only one run per experiment. Multiple starts required
    if numExp == "12":
        directed = False
        datasets = [('amazon200k', False)]
        systems = [(SpannerBundling, {'distortion':2})]
        
        runtimeExperiments(base, datasets, systems, 1, directed)
        
    if numExp == "13":
        directed = False
        datasets = [('amazon200k', False)]
        systems = [(SpannerBundlingFG, {'distortion':2}), (EPB_Biconn, {'distortion':2})]
        
        runtimeExperiments(base, datasets, systems, 1, directed)
        
    if numExp == "14":
        directed = False
        datasets = [('amazon200k', False)]
        systems = [(EPB_Biconn, {'distortion':2})]
        
        runtimeExperiments(base, datasets, systems, 1, directed)
        
    if numExp == "15":
        directed = False
        datasets = [('amazon200k', False)]
        systems = [(EPB, {'distortion':2})]
        
        runtimeExperiments(base, datasets, systems, 1, directed)
        
    #Runtime experiments for the large datasets. Only one run per experiment. Multiple starts required
    if numExp == "16":
        directed = False
        datasets = [('panama', False)]
        systems = [(SpannerBundling, {'distortion':2})]
        
        runtimeExperiments(base, datasets, systems, 1, directed)
        
    if numExp == "17":
        directed = False
        datasets = [('panama', False)]
        systems = [(SpannerBundlingFG, {'distortion':2}), (EPB_Biconn, {'distortion':2})]
        
        runtimeExperiments(base, datasets, systems, 1, directed)
        
    if numExp == "18":
        directed = False
        datasets = [('panama', False)]
        systems = [(EPB_Biconn, {'distortion':2})]
        
        runtimeExperiments(base, datasets, systems, 1, directed)
        
    if numExp == "19":
        directed = False
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(SpannerBundlingFG, {"weightFactor": 1, 'distortion': 1.5}), (SpannerBundlingFG, {"weightFactor": 2, 'distortion': 1.5}), (SpannerBundlingFG, {"weightFactor": 3, 'distortion': 1.5}), 
                   (SpannerBundlingFG, {"weightFactor": 1, 'distortion': 2}), (SpannerBundlingFG, {"weightFactor": 2, 'distortion': 2}), (SpannerBundlingFG, {"weightFactor": 3, 'distortion': 2}),
                   (SpannerBundlingFG, {"weightFactor": 1, 'distortion': 2.5}), (SpannerBundlingFG, {"weightFactor": 2, 'distortion': 2.5}), (SpannerBundlingFG, {"weightFactor": 3, 'distortion': 2.5})]
        
        plotting(base, datasets, systems, directed)
        effectivenessExperiments(base, datasets, systems, directed) 
        
    #Plot all drawings of weight factor and distortion variation and calculate ambiguity afterwards.
    if numExp == "20":
        directed = True
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(SpannerBundlingFG, {"weightFactor": 1, 'distortion': 1.5}), (SpannerBundlingFG, {"weightFactor": 2, 'distortion': 1.5}), (SpannerBundlingFG, {"weightFactor": 3, 'distortion': 1.5}), 
                   (SpannerBundlingFG, {"weightFactor": 1, 'distortion': 2}), (SpannerBundlingFG, {"weightFactor": 2, 'distortion': 2}), (SpannerBundlingFG, {"weightFactor": 3, 'distortion': 2}),
                   (SpannerBundlingFG, {"weightFactor": 1, 'distortion': 2.5}), (SpannerBundlingFG, {"weightFactor": 2, 'distortion': 2.5}), (SpannerBundlingFG, {"weightFactor": 3, 'distortion': 2.5})]
        systems = [(EPB_Biconn, {'numWorkers' : 2, 'distortion':2})]
        
        plotting(base, datasets, systems, directed)
        effectivenessExperiments(base, datasets, systems, directed) 

    #calculate ambiguity values
    if numExp == "21":
        directed = False
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(SpannerBundlingFG, None), (SpannerBundlingNoSPWithWF, None), (SpannerBundlingNoSP, None), (EPB_Biconn, {'numWorkers' : 2, 'distortion':2})]
        
        plotting(base, datasets, systems, directed)
        effectivenessExperiments(base, datasets, systems, directed) 

    #calculate ambiguity values for directed graphs
    if numExp == "22":
        directed = True
        datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        systems = [(SpannerBundlingFG, None), (SpannerBundlingNoSPWithWF, None), (SpannerBundlingNoSP, None), (EPB_Biconn, {'numWorkers' : 2, 'distortion':2})]
        
        plotting(base, datasets, systems, directed)
        effectivenessExperiments(base, datasets, systems, directed) 

    if numExp == "23":
        directed = False
        datasets = [('panama', False)]
        systems = [(SpannerBundlingFG, {'distortion':2})]
        
        plotting(base, datasets, systems, directed, printTrl=True)
        
    if numExp == "24":
        directed = False
        datasets = [('panama', False)]
        systems = [(SpannerBundlingFGSSP, {'distortion':2})]
        
        plotting(base, datasets, systems, directed, printTrl=True)

def main():
    if len(sys.argv) > 1:
        base = sys.argv[1]
    else:
        base = "."
        
    if len(sys.argv) > 2:
        experiments(base, sys.argv[2])
    else:
        print('No arguments given. Plot all variations of algorithms, then do a 1 run runtime experiment and afterwards calculate ambiguity values. Everything will be stored in folder output. Comment out lines to get individual results. Arguments should be [python3 main.py [basepath] [NumExperiment]]')
        directed = False
        # datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        # datasets = [('airlines', True), ('migration', False), ('airtraffic', False)]
        # systems = [(EPB, None), (EPB_Biconn, {'numWorkers' : 1}), (SpannerBundling, None), (SpannerBundlingFG, None), (SpannerBundlingNoSPWithWF, None), (SpannerBundlingNoSP, None)]
        # systems = [(EPB, None), (SpannerBundlingFG, None)]
        datasets = [('migration', False)]
        #systems = [(EPB_Biconn, {'numWorkers' : 1, 'distortion':2}), (EPB_Biconn, {'numWorkers' : 1, 'distortion':2.5}), (EPB_Biconn, {'numWorkers' : 1, 'distortion':3}), (EPB_Biconn, {'numWorkers' : 1, 'distortion':2, 's_tighten': 2}), (EPB_Biconn, {'numWorkers' : 1, 'distortion':2.5, 's_tighten': 2}), (EPB_Biconn, {'numWorkers' : 1, 'distortion':3, 's_tighten': 2})]

        systems = [(SpannerBundlingFG, None)]
        plotting(base, datasets, systems, directed, printTrl=False)
        
        # datasets = [('migration', False)]
        # systems = [(EPB_Biconn, {'numWorkers' : 1, 'distortion':2}), (EPB_Biconn, {'numWorkers' : 1, 'distortion':2.5}), (EPB_Biconn, {'numWorkers' : 1, 'distortion':3}), (EPB_Biconn, {'numWorkers' : 1, 'distortion':2, 's_tighten': 2}), (EPB_Biconn, {'numWorkers' : 1, 'distortion':2.5, 's_tighten': 2}), (EPB_Biconn, {'numWorkers' : 1, 'distortion':3, 's_tighten': 2})]

        # directed = True
        # #plotting(base, datasets, systems, directed, printTrl=False)
        # # plotting(base, datasets, systems, directed)
        # # #runtimeExperiments(base, datasets, systems, 1, directed)
        # # #plotLatexTable(base, datasets, systems, directed)
        # effectivenessExperiments(base, datasets, systems, directed) 
        
        # directed = False
        # datasets = [('airlines', True)]
        # systems = [(EPB_Biconn, {'numWorkers' : 1}), (SpannerBundlingFG, None), (GraphLoader, {'filename': 'cubu', 'invertX': False, 'invertY': True}), (GraphLoader, {'filename': 'wr', 'invertX': False, 'invertY': True}), (GraphLoader, {'filename': 'forcebased', 'invertX': True, 'invertY': False})]
        # #plotting(base, datasets, systems, directed)
        # #effectivenessExperiments(base, datasets, systems, directed) 
        
        # systems = [(EPB_Biconn, {'numWorkers' : 1}), (SpannerBundlingFG, None), (GraphLoader, {'filename': 'cubu', 'invertX': False, 'invertY': False}), (GraphLoader, {'filename': 'wr', 'invertX': False, 'invertY': False}), (GraphLoader, {'filename': 'forcebased', 'invertX': False, 'invertY': False})]
        # datasets = [('migration', False), ('airtraffic', False)]
        #plotting(base, datasets, systems, directed)
        #effectivenessExperiments(base, datasets, systems, directed) 
        
        
        # directed = True

        # datasets = [('airlines', True)]
        # systems = [ (GraphLoader, {'filename': 'fbd', 'invertX': False, 'invertY': True, 'reverse': False})]
        # #plotting(base, datasets, systems, directed)
        # effectivenessExperiments(base, datasets, systems, directed) 
        
        # datasets = [('migration', False),('airtraffic', False)]
        # systems = [(EPB_Biconn, {'numWorkers' : 1}), (SpannerBundlingFG, None), (GraphLoader, {'filename': 'cubu', 'invertX': False, 'invertY': False, 'reverse': False}), (GraphLoader, {'filename': 'fbd', 'invertX': False, 'invertY': False, 'reverse': False})]
        # #plotting(base, datasets, systems, directed)
        #effectivenessExperiments(base, datasets, systems, directed) 
        
        # datasets = []
        # systems = [(GraphLoader, {'filename': 'cubu', 'invertX': False, 'invertY': False, 'reverse': False}), (GraphLoader, {'filename': 'fbd', 'invertX': False, 'invertY': False, 'reverse': False})]
        # plotting(base, datasets, systems, directed)
        # runtimeExperiments(base, datasets, systems, 1, directed)
        # plotLatexTable(base, datasets, systems, directed)
        # effectivenessExperiments(base, datasets, systems, directed) 

if __name__ == "__main__":
    print('starting experiments')
    main()    


