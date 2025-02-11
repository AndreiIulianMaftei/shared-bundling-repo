import networkx as nx
import numpy as np

def writeOutput(G, filename):
    '''
    Write output as .graphml and the custom .edges .nodes format

    G - input graph with x and y coordinates as attributes
    filename - output file name 
    '''

    with open(f"datasets/{filename}.nodes", "w") as f:
        for n in G.nodes():
            f.write(f'{n} {G.nodes[n]["x"]} {G.nodes[n]["y"]}\n')
        
        f.close()

    with open(f"datasets/{filename}.edges", "w") as f:
        for e in G.edges():
            f.write(f'{e[0]} {e[1]}\n')
        
        f.close()

    nx.write_graphml(G, f'datasets/{filename}.graphml', named_key_ids=True)

def parseGraphml(pathIn, pathOut):
    G = nx.read_graphml(pathIn)

    writeOutput(G, pathOut)

def parseGml(pathIn, pathOut):
    G = nx.read_gml(pathIn, label='id')
    GG = nx.DiGraph()

    i = 0
    d = dict()

    for n, data in G.nodes(data=True):
        d[n] = i

        GG.add_node(i, x=data["graphics"]["x"], y=data["graphics"]["y"])
        i += 1

    for s,t in G.edges():
        ss = d[s]
        tt = d[t]

        if ss == tt:
            continue

        GG.add_edge(ss,tt)

    miin = 123312
    maax = -1010230

    for n, data in GG.nodes(data=True):
        data['x'] = data['x'] / 10
        data['y'] = data['y'] / 10


    GG.remove_edges_from(nx.selfloop_edges(GG))
    writeOutput(GG, pathOut + '_D')

    GGG = GG.to_undirected()
    GGG.remove_edges_from(nx.selfloop_edges(GGG))

    writeOutput(GGG, pathOut)

    with open(f"{pathOut}.edges", "w") as f:
        for e in G.edges():
            f.write(f'{e[0]} {e[1]}\n')
        
        f.close()

def parseTrailset(pathIn, fileName, pathOut, delta = 0.0001, skip = 1):
    '''
    Read files and parse into the formats we will use (.graphml and .nodes .edges). 
    As the files provide a trail set, we detect if two points of different trails are the same node by
    comparing if their difference in positions is below a threshold.

    pathIn - Input path of the file
    fileName - Name of the input file
    pathOut - Output path of the files
    delta - Threshold when two points are considered a node
    '''
    G = nx.DiGraph()

    with open(f'{pathIn}{fileName}', 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.split(' ')

            x1 = float(line[skip])
            y1 = float(line[skip + 1])
            x2 = float(line[-3])
            y2 = float(line[-2])

            n1 = -1
            n2 = -1

            for n, data in G.nodes(data=True):
                x = data['x']
                y = data['y']

                if np.abs(x1 - x) < delta and np.abs(y1 - y) < delta:
                    n1 = n
                
                if np.abs(x2 - x) < delta and np.abs(y2 - y) < delta:
                    n2 = n

            if n1 == -1:
                n1 = len(G.nodes())
                G.add_node(n1, x=x1, y=y1)
            if n2 == -1:
                n2 = len(G.nodes())
                G.add_node(n2, x=x2, y=y2)

            G.add_edge(n1,n2)

    print(len(G.nodes()), len(G.edges()))

    for n,data in G.nodes(data=True):
        data['x'] = data['x'] * 100
        data['y'] = data['y'] * 100

    writeOutput(G, pathOut + fileName)

def trailToOurFormat(pathIn, fileName, pathOut, delta = 0.0001, skip = 1):
    G = nx.DiGraph()

    with open(f'{pathIn}{fileName}', 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.rstrip()
            line = line.split(' ')

            x1 = float(line[skip])
            y1 = float(line[skip + 1])
            x2 = float(line[-2])
            y2 = float(line[-1])

            n1 = -1
            n2 = -1

            for n, data in G.nodes(data=True):
                x = data['x']
                y = data['y']

                if np.abs(x1 - x) < delta and np.abs(y1 - y) < delta:
                    n1 = n
                
                if np.abs(x2 - x) < delta and np.abs(y2 - y) < delta:
                    n2 = n

            if n1 == -1:
                n1 = len(G.nodes())
                G.add_node(n1, x=x1, y=y1)
            if n2 == -1:
                n2 = len(G.nodes())
                G.add_node(n2, x=x2, y=y2)
            
            s = " ".join(line[1:])
            #print(s)
            G.add_edge(n1,n2, trail=s)

    print(len(G.nodes()), len(G.edges()))

    with open(f"{pathOut}.nodes", "w") as f:
        for n in G.nodes():
            f.write(f'{n} {G.nodes[n]["x"]} {G.nodes[n]["y"]}\n')
        
        f.close()

    with open(f"{pathOut}.edges", "w") as f:
        for u,v,data in G.edges(data=True):
            f.write(f'{u} {v} {data["trail"]}\n')
        
        f.close()

def ebpToTrail(filenameIn, filenameOut):
    with open(f'{filenameIn}', 'r') as f:
        lines = f.readlines()
        lineOut = []

        for i, line in enumerate(lines):
            ind = line.find(" ")
            ind = line.find(" ", ind + 1)

            l = f'{i}: ' + line[(ind+1):]
            lineOut.append(l)

    with open(f'{filenameOut}', 'w') as f:
        f.writelines(lineOut)


def writeOutput(G, filename):
    '''
    Write output as .graphml and the custom .edges .nodes format

    G - input graph with x and y coordinates as attributes
    filename - output file name 
    '''
    with open(f"{filename}.trl", 'w') as f:
        i = 0
        for u,v in G.edges():
            f.write(f'{i}: {G.nodes[u]["x"]} {G.nodes[u]["y"]} {G.nodes[v]["x"]} {G.nodes[v]["y"]}\n')
            i += 1

    with open(f"{filename}.nodes", "w") as f:
        for n in G.nodes():
            f.write(f'{n} {G.nodes[n]["x"]} {G.nodes[n]["y"]}\n')
        
        f.close()

    with open(f"{filename}.edges", "w") as f:
        for e in G.edges():
            f.write(f'{e[0]} {e[1]}\n')
        
        f.close()

    nx.write_graphml(G, f'{filename}.graphml', named_key_ids=True)
    
    G = G.to_undirected()
    
    with open(f"{filename}_undirected.trl", 'w') as f:
        i = 0
        for u,v in G.edges():
            f.write(f'{i}: {G.nodes[u]["x"]} {G.nodes[u]["y"]} {G.nodes[v]["x"]} {G.nodes[v]["y"]}\n')
            i += 1

    with open(f"{filename}_undirected.nodes", "w") as f:
        for n in G.nodes():
            f.write(f'{n} {G.nodes[n]["x"]} {G.nodes[n]["y"]}\n')
        
        f.close()

    with open(f"{filename}_undirected.edges", "w") as f:
        for e in G.edges():
            f.write(f'{e[0]} {e[1]}\n')
        
        f.close()

    nx.write_graphml(G, f'{filename}_undirected.graphml', named_key_ids=True)

def parseEdgeNodeList(edges, nodes, filenameOut):
    G = nx.DiGraph()
    
    with open(f'{nodes}', 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            info = line.split(' ')
            info = [x.strip() for x in info]
                        
            G.add_node(info[0], x=info[1], y=info[2])
            
    with open(f'{edges}', 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            info = line.split(' ')
            info = [x.strip() for x in info]
            
            if info[0] not in G.nodes or info[1] not in G.nodes:
                print("error")
                continue
                
            G.add_edge(info[0], info[1])
            
    writeOutput(G, filenameOut)
        
def main():
    #parseGraphml('datasets/migrations.xml', "datasets/airlines")
    #parseGraphml('datasets/airlines.xml', "datasets/airlines")
    #parseTrailset('datasets/migrations/', '4.US-M', 'datasets/migrations/')
    #parseTrailset('datasets/migrations/', '5.Poker', 'datasets/migrations/')
    #parseTrailset('datasets/migrations/', '6.FrAir', 'datasets/migrations/', skip=2)
    #parseGml('datasets/airtraffic.gml', "airtraffic")
    #parseGml('datasets/planar.gml', "planar")
    #parseGml('datasets/amazon19k.gml', 'datasets/amazon19k')
    #parseGml('datasets/amazon200k.gml', 'datasets/amazon200k')
    trailToOurFormat("", "migration_cubu.trl", "cubu")
    #ebpToTrail('input/amazon/epb.edges', 'amazon.trl')
    #ebpToTrail('input/amazon/wr.edges', 'wr.trl')
    #parseEdgeNodeList("panama_papers.edges", "panama_papers.nodes", "panama")
    #parseGraphml('datasets/bibIV.gml', "bibIV")
    # parseGraphml('migration.graphml', "migration")
    # parseGraphml('airlines.graphml', "airlines")
    # parseGraphml('airtraffic.graphml', "airtraffic")


if __name__ == "__main__":
    main()