import networkx as nx
from collections import deque


class Spanner:

    @staticmethod
    def compute(G, t, weight=None):

        if nx.is_directed(G):
            spanner = nx.DiGraph()
        else:
            spanner = nx.Graph()

        if weight == None:
            i = 0
            for u, v, data in G.edges(data=True):
                data['ID'] = i
                data['weight'] = 1
                i += 1

            weight = 'weight'

        edges = sorted(G.edges(data=True), key=lambda t: (
            t[2].get(weight, 1), t[2].get('ID', 1)))

        for u, v, data in edges:
            if u not in spanner.nodes:
                spanner.add_edge(u, v, weight=data[weight])
                continue
            if v not in spanner.nodes:
                spanner.add_edge(u, v, weight=data[weight])
                continue

            pred, pathLength = nx.dijkstra_predecessor_and_distance(
                spanner, u, weight=weight, cutoff=t * data[weight])

            if not (v in pathLength):
                spanner.add_edge(u, v, weight=data[weight])

        return spanner


def inverse_betweenness(G):
    betweenness = nx.edge_betweenness_centrality(G, normalized=False)
    eps = 10 ** (-8)
    inverse_betweenness = {edge: 1 / (value + eps)
                           for edge, value in betweenness.items()}
    return inverse_betweenness


class InverseBetweennessSpanner():
    @staticmethod
    def compute(G, t, weight=None):

        if nx.is_directed(G):
            spanner = nx.DiGraph()
        else:
            spanner = nx.Graph()

        inverse_betweenness_edges = inverse_betweenness(G)
        if weight == None:
            i = 0
            # modified version with distances
            for u, v, data in G.edges(data=True):
                data['ID'] = i

                data['weight'] = inverse_betweenness_edges[(u, v)]

                i += 1

            weight = 'weight'

        edges = sorted(G.edges(data=True), key=lambda t: (
            t[2].get(weight, 1), t[2].get('ID', 1)))

        for u,v,data in G.edges(data=True):
            data['weight'] = 1

        for u, v, data in edges:
            if u not in spanner.nodes:
                spanner.add_edge(u, v, weight=data[weight])
                continue
            if v not in spanner.nodes:
                spanner.add_edge(u, v, weight=data[weight])
                continue

            pred, pathLength = nx.dijkstra_predecessor_and_distance(spanner, u, weight=weight,
                                                                    cutoff=t * data[weight])

            if not (v in pathLength):
                spanner.add_edge(u, v, weight=data[weight])

        print(f'> Computed t-spanner with t={t}: {spanner}')
        return spanner


def cycle_weight(G):
    cycles = nx.simple_cycles(G, length_bound=5)
    nx.set_edge_attributes(G, 0, 'cycle')

    for cycle in cycles:

        n1 = cycle[-1]
        for n2 in cycle:
            G[n1][n2]['cycle'] += 1
            n1 = n2

    cycle_weight = {}
    for u, v, data in G.edges(data=True):
        cycle_weight[(u, v)] = 1 / (data['cycle'] + 0.00001)

    return cycle_weight


class CycleSpanner():
    @staticmethod
    def compute(G, t, weight=None):

        if nx.is_directed(G):
            spanner = nx.DiGraph()
        else:
            spanner = nx.Graph()

        inverse_betweenness_edges = cycle_weight(G)
        if weight == None:
            i = 0
            # modified version with distances
            for u, v, data in G.edges(data=True):
                data['ID'] = i

                data['weight'] = inverse_betweenness_edges[(u, v)]

                i += 1

            weight = 'weight'

        edges = sorted(G.edges(data=True), key=lambda t: (
            t[2].get(weight, 1), t[2].get('ID', 1)))

        for u, v, data in edges:
            if u not in spanner.nodes:
                spanner.add_edge(u, v, weight=data[weight])
                continue
            if v not in spanner.nodes:
                spanner.add_edge(u, v, weight=data[weight])
                continue

            pred, pathLength = nx.dijkstra_predecessor_and_distance(spanner, u, weight=weight,
                                                                    cutoff=t * data[weight])

            if not (v in pathLength):
                spanner.add_edge(u, v, weight=data[weight])

        print(f'> Computed t-spanner with t={t}: {spanner}')
        return spanner


class MST:

    @staticmethod
    def compute(G, t, weight=None):

        if nx.is_directed(G):
            mst = nx.DiGraph()
        else:
            mst = nx.Graph()

        if weight == None:
            i = 0
            for u, v, data in G.edges(data=True):
                data['ID'] = i
                data['weight'] = 1
                i += 1

            weight = 'weight'

        edges = sorted(G.edges(data=True), key=lambda t: (
            t[2].get(weight, 1), t[2].get('ID', 1)))

        for u, v, data in edges:
            if u not in mst.nodes:
                mst.add_edge(u, v, weight=data[weight])
                continue
            if v not in mst.nodes:
                mst.add_edge(u, v, weight=data[weight])
                continue

            if not nx.has_path(mst, u, v):
                mst.add_edge(u, v, weight=data[weight])

        return mst


# --- Inverse Betweenness Spanner VARIANT ---
## ----------------- Betweenness centrality ----------------- ##
# The following code is from the networkx private library.
def _single_source_shortest_path_basic(G, s):
    S = []
    P = {}
    for v in range(len(G)):  # G:
        P[v] = []
    # dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
    sigma = dict.fromkeys(range(len(G)), 0.0)
    D = {}
    sigma[s] = 1.0
    D[s] = 0
    Q = deque([s])
    while Q:  # use BFS to find shortest paths
        v = Q.popleft()
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:  # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)  # predecessors
    return S, P, sigma, D


def _accumulate_edges(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            if (v, w) not in betweenness:
                betweenness[(w, v)] += c
            else:
                betweenness[(v, w)] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


def _rescale_e(betweenness, n, normalized, directed=False, k=None):
    if normalized:
        if n <= 1:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / (n * (n - 1))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness


def reconstruct_path(source, target, predecessors):
    if source == target:
        return [[source]]
    paths = []
    for predecessor in predecessors[target]:
        for path in reconstruct_path(source, predecessor, predecessors):
            paths.append(path + [target])
    return paths


def compute_sp_dict(G, edge_to_exclude, sp_dict):
    # Removing the edge
    G_modified = G.copy()
    G_modified.remove_edges_from([edge_to_exclude])

    # Compute all the shortest paths
    try:
        _, predecessors, _, _ = _single_source_shortest_path_basic(
            G_modified, edge_to_exclude[0])
        shortest_paths = reconstruct_path(
            edge_to_exclude[0], edge_to_exclude[1], predecessors)
    except nx.NetworkXNoPath:
        # print('No path between the two nodes' + str(edge_to_exclude))
        return

    sp_dict[edge_to_exclude] = shortest_paths


def edge_in_path(edge, path):
    return any((edge == (path[i], path[i+1]) or edge == (path[i+1], path[i])) for i in range(len(path) - 1))


def count_sp(sp_dict, edge_to_compute, betweenness):
    num = 0
    den = 0
    sum = 0
    # print('>Edge to compute:', edge_to_compute)
    for edge, paths in sp_dict.items():
        if edge != edge_to_compute:
            num = 0
            for path in paths:
                if edge_in_path(edge_to_compute, path):
                    num += 1

            if len(paths) == 0:
                sum += 0
            else:
                #sum += num              #without normalization
                sum += num/len(paths)


    betweenness[edge_to_compute] = sum


def compute_edge_betweenness_variant(G):
    betweenness = {e: 0 for e in G.edges()}
    sp_dict = {}
    for edge in G.edges():
        compute_sp_dict(G, edge, sp_dict)

    for edge in G.edges():
        count_sp(sp_dict, edge, betweenness)

    return betweenness


def inverse_betweenness_variant(G):
    betweenness = compute_edge_betweenness_variant(G)
    eps = 10 ** (-8)
    inverse_betweenness = {edge: 1 / (value + eps)
                           for edge, value in betweenness.items()}
    return inverse_betweenness


class InverseBetweennessSpanner_variant():
    @staticmethod
    def compute(G, t, weight=None):

        if nx.is_directed(G):
            spanner = nx.DiGraph()
        else:
            spanner = nx.Graph()

        inverse_betweenness_edges = inverse_betweenness_variant(G)
        if weight == None:
            i = 0
            # modified version with distances
            for u, v, data in G.edges(data=True):
                data['ID'] = i

                data['weight'] = inverse_betweenness_edges[(u, v)]

                i += 1

            weight = 'weight'

        edges = sorted(G.edges(data=True), key=lambda t: (
            t[2].get(weight, 1), t[2].get('ID', 1)))

        for u,v,data in G.edges(data=True):
            data['weight'] = 1

        for u, v, data in edges:
            if u not in spanner.nodes:
                spanner.add_edge(u, v, weight=data[weight])
                continue
            if v not in spanner.nodes:
                spanner.add_edge(u, v, weight=data[weight])
                continue

            pred, pathLength = nx.dijkstra_predecessor_and_distance(spanner, u, weight=weight,
                                                                    cutoff=t * data[weight])

            if not (v in pathLength):
                spanner.add_edge(u, v, weight=data[weight])

        print(f'> Computed t-spanner with t={t}: {spanner}')
        return spanner
