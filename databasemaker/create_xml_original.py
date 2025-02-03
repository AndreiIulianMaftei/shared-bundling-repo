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
    airport_headers = ["Airport ID", "Name", "City", "Country", "IATA", "ICAO",
                       "Latitude", "Longitude", "Altitude", "Timezone", "DST",
                       "Tz database timezone", "Type", "Source"]
    
    route_headers = ["Airline", "Airline ID", "Source airport", "Source airport ID",
                     "Destination airport", "Destination airport ID", "Codeshare",
                     "Stops", "Equipment"]
    
    # ------------------------------------------------------
    # 1) Read routes first, gather which airports are used.
    # ------------------------------------------------------
    used_airports = set()   # Will store the Airport IDs (strings) actually used by routes
    all_routes = []         # Store valid routes for later use (avoid duplicates in memory)
    
    with open('./databasemaker/routes_data.csv', 'r', encoding='utf-8') as csvfile_routes:
        reader = csv.reader(csvfile_routes, delimiter=',')
        for row in reader:
            if not row or len(row) < len(route_headers):
                continue
            route_data = dict(zip(route_headers, row))
            
            source_id = route_data['Source airport ID'].strip().strip('"')
            dest_id   = route_data['Destination airport ID'].strip().strip('"')

            # Skip if missing IDs
            if not source_id or not dest_id:
                continue
            
            # We'll store this route; but final check for "only US airports" will be done later
            all_routes.append((source_id, dest_id))
            
            # At least track that these two IDs occur in routes
            used_airports.add(source_id)
            used_airports.add(dest_id)
    
    # ------------------------------------------------------
    # 2) Read airports; only keep those in the "United States"
    #    AND that appear in used_airports. Also build node XML.
    # ------------------------------------------------------
    airport_id_to_node_id = {}
    node_lines = []         # Collect XML lines for nodes
    node_id = 0

    # We'll keep a separate set of "actually used airports" (both in the country and in used_airports).
    actually_used_airports = set()

    with open('./databasemaker/airport_data.csv', 'r', encoding='utf-8') as csvfile_airport:
        reader = csv.reader(csvfile_airport, delimiter=',')
        for row in reader:
            if not row or len(row) < len(airport_headers):
                continue
            
            airport_data = dict(zip(airport_headers, row))
            
            airport_id = airport_data['Airport ID'].strip().strip('"')
            country    = airport_data['Country'].strip().strip('"')
            
            # Only consider the country we want (e.g., "United States")
            # and only if it appeared in routes
            if country != 'Russia':
                continue
            if airport_id not in used_airports:
                continue

            # Extract coords & IATA
            longitude = airport_data['Longitude'].strip().strip('"')
            latitude  = airport_data['Latitude'].strip().strip('"')
            iata      = airport_data['IATA'].strip().strip('"')
            
            if not airport_id or not longitude or not latitude or not iata or iata == '\\N':
                continue
            
            try:
                longitude = float(longitude)
                latitude  = float(latitude)
            except ValueError:
                continue  # Invalid numeric data

            # Compute x,y (just as in your original script)
            x = -longitude * 10
            y = latitude * 10

            # Tooltip
            tooltip = f"{iata}(lngx={longitude},laty={latitude})"
            tooltip = escape_xml(tooltip)

            # Compose the <node> XML
            node_xml = f'''    <node id="{node_id}">
      <data key="x">{x}</data>
      <data key="tooltip">{tooltip}</data>
      <data key="y">{y}</data>
    </node>
'''
            node_lines.append(node_xml)
            
            # Map from airport ID -> new node ID
            airport_id_to_node_id[airport_id] = node_id
            
            # Mark this airport as actually used
            actually_used_airports.add(airport_id)
            
            node_id += 1
    
    # ------------------------------------------------------
    # 3) Generate edges from the routes we stored earlier,
    #    but only if both airports are in actually_used_airports.
    # ------------------------------------------------------
    added_edges = set()   # to de-duplicate
    edge_lines  = []
    edge_id     = 0

    for (source_id, dest_id) in all_routes:
        # Only generate edge if both are "actually used" (i.e., valid US airports we read)
        if source_id in actually_used_airports and dest_id in actually_used_airports:
            source_node_id = airport_id_to_node_id[source_id]
            dest_node_id   = airport_id_to_node_id[dest_id]

            # Use a frozenset to avoid duplicates in an undirected graph
            edge_key = frozenset([source_node_id, dest_node_id])
            if edge_key in added_edges:
                continue

            added_edges.add(edge_key)

            edge_xml = f'''    <edge id="{edge_id}" source="{source_node_id}" target="{dest_node_id}">
    </edge>
'''
            edge_lines.append(edge_xml)
            edge_id += 1

    # ------------------------------------------------------
    # 4) Write everything to 'airports2.xml'
    # ------------------------------------------------------
    with open('airports2.xml', 'w', encoding='UTF-8') as xmlfile:
        xml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<xml xmlns="http://graphml.graphdrawing.org/xmlns"
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

        # Write the node elements we collected
        for node_xml in node_lines:
            xmlfile.write(node_xml)

        # Add a comment for edges
        xmlfile.write('\n    <!-- edges -->\n')
        for edge_xml in edge_lines:
            xmlfile.write(edge_xml)

        xml_footer = '''  </graph>
</xml>
'''
        xmlfile.write(xml_footer)

if __name__ == '__main__':
    main()
