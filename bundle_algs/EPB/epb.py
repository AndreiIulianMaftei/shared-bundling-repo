
from concurrent.futures import ThreadPoolExecutor
from bundle_algs.EPB.abstractBundling import AbstractBundling
import networkx as nx
import numpy as np
from nx2ipe.nx2ipe import SplineC
import time

class EPB(AbstractBundling):
    '''
    EPB. Implementation
    
    weightFactor: kappa value that sets the bundling strength
    distortion: t value that sets the maximum allowed stretch/distortion
    '''

    def __init__(self, G: nx.Graph, weightFactor = 2, distortion = 2, s_tighten = 1):
        super().__init__(G)
        self.distortion = distortion
        self.weightFactor = weightFactor
        self.s_tighten = s_tighten
        self.name = 'epb'
        self.numWorkers = 'no_biconn'

    @property
    def name(self):
        return f'ebp_d_{self.distortion}_w_{self.weightFactor}_s_{self.s_tighten}_nw_{self.numWorkers}'

    @name.setter
    def name(self, value):
        self._name = value

    def bundle(self):

        
        t = time.process_time()
        G = self.G.copy()
        
        nx.set_edge_attributes(G, False, 'backbone')
        nx.set_edge_attributes(self.G, 'Unbundled', name='Layer')
        edges = sorted(G.edges(data=True), key=lambda t: t[2].get('dist', 1), reverse=True)

        for u,v,data in edges:
            data['weight'] = np.power(data['dist'], self.weightFactor)

        for u,v,data in edges:
            
            
            if data['backbone']:
                continue

            G.remove_edge(u,v)

            Path = None
            try:
                Path = nx.single_source_dijkstra(G, u, target=v, weight='weight')      
                Path = Path[1]
            except nx.NetworkXNoPath:
                G.add_edge(u, v)
                nx.set_edge_attributes(G, {(u, v): data})
                continue

            length = 0
            current = Path[0]

            flag = False
            for next in Path[1:]:

                length += G[current][next]['dist']
                
                if length > self.distortion * data['dist']:
                    flag = True
                    break

                current = next

            if flag:
                G.add_edge(u,v)
                nx.set_edge_attributes(G, {(u, v): data})
                continue

            spline = []

            current = Path[0]
            for node in Path[1:-1]:
                x = G.nodes[node]['X']
                y = G.nodes[node]['Y']

                spline.append((x,y))

                self.G[current][node]['Stroke'] = 'blue'  
                self.G[current][node]['Layer'] = 'Backbone'  
                G[current][node]['backbone'] = True
                current = node

            self.G[current][Path[-1]]['backbone'] = True
            G[current][Path[-1]]['Stroke'] = 'blue'

            self.G[u][v]['Spline'] = SplineC(spline)
            self.G[u][v]['Stroke'] = 'purple'
            self.G[u][v]['Layer'] = 'Bundled'

        if self.s_tighten >= 2:
            self.tighten_bundles()

        t2 = time.process_time() - t

        return t2       

    def tighten_bundles(self):
        
        for u,v,data in self.G.edges(data=True):
            if 'Spline' not in data:
                continue
            
            for _ in range(1, self.s_tighten):
                new_spline = []
                old_spline = data['Spline'].points    
                
                last = (self.G.nodes[u]['X'], self.G.nodes[u]['Y'])
                
                for p in old_spline:
                    pX = (last[0] + p[0]) / 2
                    pY = (last[1] + p[1]) / 2
                    new_spline.append((pX,pY))
                    new_spline.append(p)
                    last = p
                    
                data['Spline'] = SplineC(new_spline)
                
                    

class EPB_Biconn(EPB):
    '''
    EPB Implementation with biconnected component decomposition.
    
    weightFactor: kappa value that sets the bundling strength
    distortion: t value that sets the maximum allowed stretch/distortion
    numWorkers: number of workers that process biconnected components 
    '''

    def __init__(self, G: nx.Graph, weightFactor=2, distortion=2, numWorkers=8, s_tighten=1):
        super().__init__(G, weightFactor=weightFactor, distortion=distortion, s_tighten=s_tighten)
        self.numWorkers = numWorkers
        
    @property
    def name(self):
        return f'ebp_d_{self.distortion}_w_{self.weightFactor}_s_{self.s_tighten}_nw_{self.numWorkers}'

    @name.setter
    def name(self, value):
        self._name = value

    def bundle(self):
        
        t = time.process_time()
        
        if nx.is_directed(self.G):
            GG = self.G.to_undirected(as_view=True)
            components = nx.biconnected_components(GG)
        else:
            components = nx.biconnected_components(self.G)

        toProcess = []
        for nodes in components:
            if len(nodes) > 2:
                G = self.G.subgraph(nodes).copy()
                toProcess.append(G)
          

        with ThreadPoolExecutor(max_workers=self.numWorkers) as executor:
            for g in toProcess:
                self.process(g)
                #executor.submit(self.process, (g))

        if self.s_tighten >= 2:
            self.tighten_bundles()

        t2 = time.process_time() - t

        return t2

    def process(self, component):

        nx.set_edge_attributes(component, False, 'backbone')
        edges = sorted(component.edges(data=True), key=lambda t: t[2].get('dist', 1), reverse=True)

        for u,v,data in edges:
            data['weight'] = np.power(data['dist'], self.weightFactor)

        for u,v,data in edges:
            if data['backbone']:
                continue

            component.remove_edge(u,v)

            Path = None
            try:
                Path = nx.single_source_dijkstra(component, u, target=v, weight='weight')      
                #print(Path)   
                Path = Path[1]
            except nx.NetworkXNoPath:
                component.add_edge(u, v)
                nx.set_edge_attributes(component, {(u, v): data})
                continue

            length = 0
            current = Path[0]

            flag = False
            for next in Path[1:]:

                length += component[current][next]['dist']
                
                if length > self.distortion * data['dist']:
                    flag = True
                    break

                current = next

            if flag:
                component.add_edge(u,v)
                nx.set_edge_attributes(component, {(u, v): data})
                continue

            spline = []
            #print(Path, Path[-1])

            current = Path[0]

            for node in Path[1:-1]:
                x = component.nodes[node]['X']
                y = component.nodes[node]['Y']

                spline.append((x,y))

                self.G[current][node]['Stroke'] = 'blue'  
                component[current][node]['backbone'] = True
                current = node

            self.G[current][Path[-1]]['backbone'] = True
            component[current][Path[-1]]['Stroke'] = 'blue'

            self.G[u][v]['Spline'] = SplineC(spline)
            self.G[u][v]['Stroke'] = 'purple'

        return
