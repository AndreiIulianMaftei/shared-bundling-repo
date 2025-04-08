import csv

def escape_xml(text):
    """
    Escape special characters in text for XML.
    """
    return (text.replace("&", "&amp;")
                .replace('"', "&quot;")
                .replace("'", "&apos;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

def main():
    """
    This script processes airport and route data to generate a GraphML file representing
    airports as nodes and routes as edges for a specified country. The generated GraphML
    file can be used for graph-based visualizations and analyses.
    Functions:
        main():
            Reads airport and route data from CSV files, filters the data for a specified
            country, and generates a GraphML file containing nodes (airports) and edges
            (routes). The nodes include attributes such as coordinates and tooltips, while
            the edges represent connections between airports.
    File Structure:
        - Input files:
            - `./databasemaker/airport_data.csv`: Contains airport data with headers such as
              "Airport ID", "Name", "City", "Country", "IATA", "Longitude", "Latitude", etc.
            - `./databasemaker/routes_data.csv`: Contains route data with headers such as
              "Source airport ID", "Destination airport ID", etc.
        - Output file:
            - `{definedCountry}_airports.xml`: A GraphML file containing the graph representation
              of airports and routes for the specified country.
    Key Features:
        - Filters airports and routes based on the specified country (`definedCountry`).
        - Converts longitude and latitude to x and y coordinates for graph visualization.
        - Avoids duplicate edges in the graph.
        - Escapes special characters in tooltips to ensure valid XML.
    Dependencies:
        - Requires the `csv` module for reading CSV files.
        - Assumes the existence of input files in the specified paths.
    Usage:
        Run the script to generate a GraphML file for the specified country. Update the
        `definedCountry` variable to change the target country.
    Note:
        Ensure that the input CSV files are properly formatted and contain the required
        headers for successful execution.
    """
    airport_headers = ["Airport ID", "Name", "City", "Country", "IATA", "ICAO",
                       "Latitude", "Longitude", "Altitude", "Timezone", "DST",
                       "Tz database timezone", "Type", "Source"]
    
    route_headers = ["Airline", "Airline ID", "Source airport", "Source airport ID",
                     "Destination airport", "Destination airport ID", "Codeshare",
                     "Stops", "Equipment"]
    
    airport_id_to_node_id = {}
    node_id = 0  

    definedCountry = 'Indonesia'

    country_airport_ids = set()
    
    with open(f'{definedCountry}_airports.xml', 'w', encoding='UTF-8') as xmlfile:
        xml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
  http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">

  <key id="x" for="node" attr.name="x" attr.type="double"/>
  <key id="tooltip" for="node" attr.name="tooltip" attr.type="string"/>
  <key id="y" for="node" attr.name="y" attr.type="double"/>

  <graph edgedefault="undirected">
    <!-- nodes -->
'''
        xmlfile.write(xml_header)

        # Step 1: Process airports and generate nodes
        with open('./databasemaker/airport_data.csv', 'r', encoding='utf-8') as csvfile_airport:
            reader = csv.reader(csvfile_airport, delimiter=',')
            
            for row in reader:
                if not row or len(row) < len(airport_headers):
                    continue
                
                airport_data = dict(zip(airport_headers, row))
                
                airport_id = airport_data['Airport ID'].strip().strip('"')
                longitude = airport_data['Longitude'].strip().strip('"')
                latitude = airport_data['Latitude'].strip().strip('"')
                iata = airport_data['IATA'].strip().strip('"')
                country = airport_data['Country'].strip().strip('"')
                
                # Filter: Only include airports in the specified country
                if country != definedCountry:
                    continue

                if not airport_id or not longitude or not latitude or not iata or iata == '\\N':
                    continue
                
                try:
                    longitude = float(longitude)
                    latitude = float(latitude)
                except ValueError:
                    continue  # Skip rows with invalid data

                # Compute x and y coordinates
                x = -longitude * 10
                y = latitude * 10

                # Prepare tooltip
                tooltip = f"{iata}(lngx={longitude},laty={latitude})"
                tooltip = escape_xml(tooltip)

                # Create node element
                node_xml = f'''    <node id="{node_id}">
      <data key="x">{x}</data>
      <data key="tooltip">{tooltip}</data>
      <data key="y">{y}</data>
    </node>
'''
                xmlfile.write(node_xml)

                # Map airport ID to node ID
                airport_id_to_node_id[airport_id] = node_id
                country_airport_ids.add(airport_id)
                node_id += 1  

        # Step 2: Process routes and generate edges
        xmlfile.write('\n    <!-- edges -->\n')
        edge_id = 0 

        # Set to keep track of added edges to avoid duplicates
        added_edges = set()

        with open('./databasemaker/routes_data.csv', 'r', encoding='utf-8') as csvfile_routes:
            reader = csv.reader(csvfile_routes, delimiter=',')
            
            for row in reader:
                if not row or len(row) < len(route_headers):
                    continue

                route_data = dict(zip(route_headers, row))
                
                source_airport_id = route_data['Source airport ID'].strip().strip('"')
                dest_airport_id = route_data['Destination airport ID'].strip().strip('"')

                if not source_airport_id or not dest_airport_id:
                    continue

                # Include edge only if both airports are in the specified country
                if (source_airport_id in country_airport_ids and 
                    dest_airport_id in country_airport_ids):
                    
                    source_node_id = airport_id_to_node_id[source_airport_id]
                    dest_node_id = airport_id_to_node_id[dest_airport_id]

                    # Create a frozenset of node IDs to represent the edge
                    edge_nodes = frozenset([source_node_id, dest_node_id])

                    # Check if the edge has already been added
                    if edge_nodes in added_edges:
                        continue  # Skip adding this edge again

                    # Add the edge to the set of added edges
                    added_edges.add(edge_nodes)

                    # Create edge element
                    edge_xml = f'''    <edge id="{edge_id}" source="{source_node_id}" target="{dest_node_id}">
    </edge>
'''
                    xmlfile.write(edge_xml)
                    edge_id += 1  

        # Write footer
        xml_footer = '''  </graph>
</graphml>
'''
        xmlfile.write(xml_footer)

if __name__ == '__main__':
    main()
