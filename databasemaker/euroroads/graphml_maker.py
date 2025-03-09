import sys

# If you want to geocode via geopy (online approach):
# pip install geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time

def geocode_city(city_name, geolocator, max_retries=3, delay=2): 
    """
    Use geopy to fetch geographic coordinates (latitude, longitude) for a given city name.
    Returns (lat, lng).
    """
    # You may want to add retries, error handling, etc.
    # Sleep to respect Nominatim usage policy
    time.sleep(1)
    for attempt in range(1, max_retries + 1):
        try:
            location = geolocator.geocode(city_name)
            if location:
                return (location.latitude, location.longitude)
            else:
                # If geocode returns None, city wasn't found
                print(f"Warning: City '{city_name}' not found by geocoder.")
                return (None, None)
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            print(f"Warning: Geocoding '{city_name}' failed (attempt {attempt}/{max_retries}). Error: {e}")
            if attempt < max_retries:
                # Wait a bit before retrying
                time.sleep(delay)
            else:
                # After final attempt, give up
                print(f"Giving up on '{city_name}' after {max_retries} attempts.")
                return (None, None)

def main():
    # 1. Read city names from file
    city_file = "databasemaker/euroroads/cities.txt"
    with open(city_file, "r", encoding="utf-8") as f:
        cities = [line.strip() for line in f if line.strip()]
    
    # We will map city IDs (1-based) to (city_name, lat, lng).
    # For example, ID=1 -> ('Greenock', 55.948..., -3.196...), etc.
    # The simplest approach is to assume the line order matches
    # the IDs in the edges file. I.e., line 1 => ID=1, line 2 => ID=2, ...
    city_data = {}
    
    # 2. Geocode each city to get lat/lng
    geolocator = Nominatim(user_agent="city_geocoder_example", timeout=10)
    
    for i, city_name in enumerate(cities, start=1):
        print(f"Geocoding city {i}/{len(cities)}: {city_name}")
        lat, lng = geocode_city(city_name, geolocator)
        city_data[i] = (city_name, lat, lng)
    
    # 3. Read edges from second file
    # Each line is "source target"
    edges_file = "databasemaker/euroroads/edges.txt"
    with open(edges_file, "r", encoding="utf-8") as f:
        raw_edges = [line.strip().split() for line in f if line.strip()]
    
    # Convert edges to integer pairs and skip those for which we have no city data
    edges = []
    for e in raw_edges:
        try:
            sid = int(e[0])
            tid = int(e[1])
            # Only keep this edge if both endpoints exist in our city_data
            if sid in city_data and tid in city_data:
                edges.append((sid, tid))
            else:
                # For example, 855 doesn't exist in the sample city list
                print(f"Skipping edge referencing unknown city ID(s): {sid}, {tid}", file=sys.stderr)
        except ValueError:
            print(f"Skipping invalid edge entry: {e}", file=sys.stderr)
    
    # 4. Generate GraphML output
    graphml_header = """<?xml version="1.0" encoding="UTF-8"?>
<xml xmlns="http://graphml.graphdrawing.org/xmlns"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
  http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">

  <key id="x" for="node" attr.name="x" attr.type="double"/>
  <key id="tooltip" for="node" attr.name="tooltip" attr.type="string"/>
  <key id="y" for="node" attr.name="y" attr.type="double"/>

  <graph edgedefault="undirected">
"""
    
    # Create nodes
    node_entries = []
    for city_id, (city_name, lat, lng) in city_data.items():
        if lat is None or lng is None:
            # If we never got coordinates, skip or handle differently
            continue
        # Example coordinate transform for x, y in GraphML
        # (Negating them so they resemble your sample's negative coordinates)
        x = -lng * 10
        y = -lat * 10
        
        tooltip = f"{city_name}(lngx={lng},laty={lat})"
        
        node_xml = f"""    <node id="{city_id}">
      <data key="x">{x}</data>
      <data key="tooltip">{tooltip}</data>
      <data key="y">{y}</data>
    </node>"""
        node_entries.append(node_xml)
    
    # Create edges
    edge_entries = []
    for idx, (source, target) in enumerate(edges, start=1):
        edge_xml = f"""    <edge id="{idx}" source="{source}" target="{target}"></edge>"""
        edge_entries.append(edge_xml)
    
    graphml_footer = """  </graph>
</xml>
"""
    
    # Combine everything
    graphml_content = (
        graphml_header
        + "\n".join(node_entries)
        + "\n"
        + "\n".join(edge_entries)
        + "\n"
        + graphml_footer
    )
    
    # 5. Write the GraphML content to a file
    output_file = "output.graphml"
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(graphml_content)
    
    print(f"GraphML file written to: {output_file}")

if __name__ == "__main__":
    main()
