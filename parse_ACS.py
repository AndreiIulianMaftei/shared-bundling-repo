import pandas as pd
import networkx as nx


from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from pyproj import Transformer

def add_projected_coordinates(G):
    """
    Adds projected 'x', 'y' coordinates to each node in graph G,
    using an Albers Equal Area projection for the contiguous US.

    Requires node attributes:
        - 'lat' (latitude in degrees)
        - 'lon' (longitude in degrees)
    """

    # Transformer: WGS84 (lat/lon) -> NAD83 / Conus Albers (EPSG:5070)
    transformer = Transformer.from_crs(
        "EPSG:4326",   # source: WGS84 lat/lon
        "EPSG:5070",   # target: US Albers Equal Area
        always_xy=True # expects (lon, lat) order
    )

    for node, data in G.nodes(data=True):
        lon = data.get("lon")
        lat = data.get("lat")

        # skip if we don't have coords
        if lon is None or lat is None:
            continue

        try:
            x, y = transformer.transform(lon, lat)
        except Exception:
            # if transformation fails for some odd reason, skip
            continue

        # store projected coords
        data["x"] = x
        data["y"] = y



# -------------------------------------------------------------------
# 1. CONFIG: column names in your Excel file
#    (Use print(df.columns.tolist()) once to confirm & edit if needed)
# -------------------------------------------------------------------
COLUMN_MAP = {
    "state_a_fips": "State Code of Geography A",
    "county_a_fips": "FIPS County Code of Geography A",
    "state_b_fips": "State/U.S. Island Area/Foreign Region Code of Geography B",
    "county_b_fips": "FIPS County Code of Geography B",

    "state_a_name": "State Name of Geography A",
    "county_a_name": "County Name of Geography A",
    "state_b_name": "State/U.S. Island Area/Foreign Region of Geography B",
    "county_b_name": "County Name of Geography B",

    # A ← B (flow from Geography B to Geography A)
    "flow_b_to_a": "Flow from Geography B to Geography A Estimate",
    # A → B (counterflow from Geography A to Geography B)
    "flow_a_to_b": "Counterflow from Geography A to Geography B2 Estimate",
}

import geopandas as gpd

def load_county_centroids(shapefile_path):
    gdf = gpd.read_file(shapefile_path)

    # Centroid in EPSG:4326 (lat/lon)
    gdf = gdf.to_crs("EPSG:4326")
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["lon"] = gdf.geometry.centroid.x

    # Build lookup dict keyed by "SS-CCC" (e.g. "01-001")
    lookup = {}
    for _, row in gdf.iterrows():
        state_fips = row["STATEFP"]
        county_fips = row["COUNTYFP"]
        key = f"{state_fips}-{county_fips}"
        lookup[key] = (row["lat"], row["lon"])

    return lookup

# -------------------------------------------------------------------
# 2. Helper: geocode a county to (lat, lon) with caching
# -------------------------------------------------------------------
def make_geocoder():
    geolocator = Nominatim(user_agent="us_migration_graph")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    cache = {}

    def get_coords(state_name, county_name):
        """
        Return (lat, lon) for 'County Name, State Name, USA'.
        Results are cached so each county is only geocoded once.
        """
        key = (state_name, county_name)
        if key in cache:
            return cache[key]

        if pd.isna(state_name) or pd.isna(county_name):
            cache[key] = (None, None)
            return cache[key]

        query = f"{county_name}, {state_name}, USA"
        location = geocode(query)
        if location:
            cache[key] = (location.latitude, location.longitude)
        else:
            cache[key] = (None, None)
        return cache[key]

    return get_coords

US_STATE_NAMES = { # , "Alaska", "Hawaii", 
    "Alabama", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
    "District of Columbia",  # keep or remove as you like
}

def get_coords_from_fips(centroid_lookup, state_fips, county_fips):
    key = f"{state_fips}-{county_fips}"
    return centroid_lookup.get(key, (None, None))

# -------------------------------------------------------------------
# 3. Core function: build the migration graph from an Excel workbook
# -------------------------------------------------------------------
def build_migration_graph(excel_path, exception=False):
    """
    excel_path: path to your .xls file where each sheet is a state.

    Returns: networkx.DiGraph
    Nodes = counties (identified by state_fips-county_fips)
    Edges = directed migration flows with attribute 'weight'
    """
    # Read all sheets into a dict of DataFrames
    # If your file has extra header rows, use skiprows=... here.
    sheets = pd.read_excel(excel_path, sheet_name=None, dtype=str, header=[0, 1])
    #sheets.head()

    centroids = load_county_centroids("migration/tl_2024_us_county.shp")

    G = nx.DiGraph()

    for sheet_name, df in sheets.items():
        # Optional: inspect the first few rows to make sure it looks right
        df = df.drop(['FIPS Minor Civil Division (MCD) Code of Geography B1', 'Minor Civil Division of Geography B'], axis=1, errors='ignore')
        #print(sheet_name, df.head())

        # If your file has multi-line headers (e.g. Estimate / MOE),
        # you might need to drop MOE columns or rename columns here.
        # Example of keeping only the Estimate columns if they are multi-indexed:
        # if isinstance(df.columns, pd.MultiIndex):
        #     df.columns = [' '.join([c for c in col if isinstance(c, str)])
        #                   for col in df.columns]

        for _, row in df.iterrows():
            # --- Build node IDs (zero-padded FIPS codes) ---
            state_a_fips = str(row.iloc[0]).zfill(2)
            county_a_fips = str(row.iloc[1]).zfill(3)
            state_b_fips = str(row.iloc[2]).zfill(2)
            county_b_fips = str(row.iloc[3]).zfill(3)

            if len(state_a_fips) > 2:
                state_a_fips = state_a_fips[-2:]

            if len(state_b_fips) > 2:
                state_b_fips = state_b_fips[-2:]

            node_a = f"{state_a_fips}-{county_a_fips}"
            node_b = f"{state_b_fips}-{county_b_fips}"

            if exception:
                state_a_name = row.iloc[5]
                county_a_name = row.iloc[6]
                state_b_name = row.iloc[7]
                county_b_name = row.iloc[8]
            else:
                state_a_name = row.iloc[4]
                county_a_name = row.iloc[5]
                state_b_name = row.iloc[6]
                county_b_name = row.iloc[7]

            if pd.isna(county_a_name) or pd.isna(county_b_name):
                # Skip if county names are NaN
                continue

            if county_a_name is None or county_b_name is None:
                # Skip if county names are missing
                continue

            if state_b_name not in US_STATE_NAMES or state_a_name not in US_STATE_NAMES:
                # Skip non-US locations
                continue

            lat_a, lon_a = get_coords_from_fips(centroids, state_a_fips, county_a_fips)
            lat_b, lon_b = get_coords_from_fips(centroids, state_b_fips, county_b_fips)

            if not (lat_a and lon_a and lat_b and lon_b):
                continue

            # --- Add / update nodes with attributes & coordinates ---
            if node_a not in G:
                G.add_node(
                    node_a,
                    county_name=county_a_name,
                    lat=lat_a,
                    lon=lon_a,
                )


            if node_b not in G:
                G.add_node(
                    node_b,
                    county_name=county_b_name,
                    lat=lat_b,
                    lon=lon_b,
                )


            # --- Add directed edges using A→B and B→A flows ---
            if exception:
                flow_b_to_a = row[10]
                flow_a_to_b = row[12]
            else:
                flow_b_to_a = row[8]
                flow_a_to_b = row[10]

            # Convert to numeric (errors='coerce' turns non-numeric into NaN)
            try:
                flow_b_to_a = float(flow_b_to_a)
            except (TypeError, ValueError):
                flow_b_to_a = 0.0

            try:
                flow_a_to_b = float(flow_a_to_b)
            except (TypeError, ValueError):
                flow_a_to_b = 0.0

            # Edge: B -> A
            if flow_b_to_a and flow_b_to_a > 0:
                if G.has_edge(node_b, node_a):
                    G[node_b][node_a]["weight"] += flow_b_to_a
                else:
                    G.add_edge(node_b, node_a, weight=flow_b_to_a)

            # Edge: A -> B
            if flow_a_to_b and flow_a_to_b > 0:
                if G.has_edge(node_a, node_b):
                    G[node_a][node_b]["weight"] += flow_a_to_b
                else:
                    G.add_edge(node_a, node_b, weight=flow_a_to_b)

    return G

def toGrossGraph(G):
    G2 = nx.DiGraph()

    # Copy node attributes
    for node, data in G.nodes(data=True):
        if node not in G2:
            G2.add_node(node)
        for key, value in data.items():
            G2.nodes[node][key] = value

    nodes = list(G.nodes())

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u = nodes[i]
            v = nodes[j]

            weight_uv = G[u][v]["weight"] if G.has_edge(u, v) else 0
            weight_vu = G[v][u]["weight"] if G.has_edge(v, u) else 0

            total_weight = weight_uv + weight_vu

            if total_weight > 0:
                G2.add_edge(u, v, weight=total_weight)
            elif total_weight < 0:
                G2.add_edge(v, u, weight=-total_weight)     
    return G2


def filtered_graph(G, min_weight):
    """
    Returns a *new* DiGraph containing only edges with weight >= min_weight.
    Preserves all node attributes.
    """
    H = nx.DiGraph()

    # copy all nodes with attributes
    H.add_nodes_from(G.nodes(data=True))

    # copy only edges meeting criteria
    for u, v, d in G.edges(data=True):
        if d.get("weight", 0) >= min_weight:
            H.add_edge(u, v, **d)

    return H


# -------------------------------------------------------------------
# 4. Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":

    for name in ['cc2020.xlsx']:
        excel_file = f"migration/{name}"  # <-- your file here
        # excel_file = f"migration/test.xlsx"  # <-- your file here

        if name in ['cc2020.xlsx']:
            G1 = build_migration_graph(excel_file, exception=True)
        else:
            G1 = build_migration_graph(excel_file)

        G2 = toGrossGraph(G1)
        G3 = nx.to_undirected(G2)
        GC = sorted(nx.connected_components(G3), key=len, reverse=True)[0]
        G3 = G3.subgraph(GC).copy()

        add_projected_coordinates(G1)
        add_projected_coordinates(G2)
        add_projected_coordinates(G3)

        print("Number of nodes:", G3.number_of_nodes())
        print("Number of edges:", G3.number_of_edges())

        min_weight = 400
        G4 = filtered_graph(G1, min_weight=min_weight)

        G5 = toGrossGraph(G4)
        G6 = nx.to_undirected(G5)
        GC = sorted(nx.connected_components(G6), key=len, reverse=True)[0]
        G6 = G6.subgraph(GC).copy()


        print("After filtering edges with weight >= 300:")
        print("Number of nodes:", G6.number_of_nodes())
        print("Number of edges:", G6.number_of_edges())

        nx.write_graphml(G1, f"migration/graphs/migration_full_directed_{name.split('.')[0]}.graphml")
        nx.write_graphml(G2, f"migration/graphs/migration_full_directed_gross_{name.split('.')[0]}.graphml")
        nx.write_graphml(G3, f"migration/graphs/migration_full_undirected_{name.split('.')[0]}.graphml")

        nx.write_graphml(G4, f"migration/graphs/migration_>{min_weight}_directed_{name.split('.')[0]}.graphml")
        nx.write_graphml(G5, f"migration/graphs/migration_>{min_weight}_directed_gross_{name.split('.')[0]}.graphml")
        nx.write_graphml(G6, f"migration/graphs/migration_>{min_weight}_undirected_{name.split('.')[0]}.graphml")
    # Save for later use
