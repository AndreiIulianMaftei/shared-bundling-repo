import csv

def escape_xml(text):
    """
    Escape special characters in text for XML.
    """
    return (text.replace("&", "&amp;")
                .replace("\"", "&quot;")
                .replace("'", "&apos;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

def main():
    # Define the headers since the CSV files don't include them
    airport_headers = ["Airport ID", "Name", "City", "Country", "IATA", "ICAO", 
                       "Latitude", "Longitude", "Altitude", "Timezone", "DST", 
                       "Tz database timezone", "Type", "Source"]
    
    route_headers = ["Airline", "Airline ID", "Source airport", "Source airport ID", 
                     "Destination airport", "Destination airport ID", "Codeshare", 
                     "Stops", "Equipment"]
    
    # Mapping from Airport ID to node ID
    airport_id_to_node_id = {}
    node_id = 0  # Node ID counter

    # Set to store IDs of Romanian airports
    romanian_airport_ids = set()
    
    # Open the output XML file
    with open('airports.xml', 'w', encoding='UTF-8') as xmlfile:
        # Prepare XML header
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

        # Step 1: Process airports and generate nodes
        with open('airport_data.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            
            for row in reader:
                # Skip empty rows or incomplete rows
                if not row or len(row) < len(airport_headers):
                    continue
                
                # Create a mapping from header name to value
                airport_data = dict(zip(airport_headers, row))
                
                airport_id = airport_data['Airport ID'].strip().strip('"')
                longitude = airport_data['Longitude'].strip().strip('"')
                latitude = airport_data['Latitude'].strip().strip('"')
                iata = airport_data['IATA'].strip().strip('"')
                country = airport_data['Country'].strip().strip('"')
                
                # Keep track of Romanian airports
                if country == 'Romania':
                    romanian_airport_ids.add(airport_id)

                # Skip if longitude, latitude, or IATA code is missing
                if not airport_id or not longitude or not latitude or not iata or iata == '\\N':
                    continue
                
                # Convert to float
                try:
                    longitude = float(longitude)
                    latitude = float(latitude)
                except ValueError:
                    continue  # Skip rows with invalid data

                # Compute x and y
                x = longitude * 10
                y = -latitude * 10

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
                node_id += 1  # Increment node ID

        # Step 2: Process routes and generate edges
        xmlfile.write('\n    <!-- edges -->\n')
        edge_id = 0  # Edge ID counter

        with open('routes_data.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            
            for row in reader:
                # Skip empty rows or incomplete rows
                if not row or len(row) < len(route_headers):
                    continue

                # Create a mapping from header name to value
                route_data = dict(zip(route_headers, row))
                
                source_airport_id = route_data['Source airport ID'].strip().strip('"')
                dest_airport_id = route_data['Destination airport ID'].strip().strip('"')

                # Skip if source or destination airport ID is missing
                if not source_airport_id or not dest_airport_id:
                    continue

                # Include edge only if source airport is in Romania
                if source_airport_id in romanian_airport_ids:
                    # Check if both source and destination airports exist in the mapping
                    if (source_airport_id in airport_id_to_node_id and 
                        dest_airport_id in airport_id_to_node_id):
                        
                        source_node_id = airport_id_to_node_id[source_airport_id]
                        dest_node_id = airport_id_to_node_id[dest_airport_id]

                        # Create edge element
                        edge_xml = f'''    <edge id="{edge_id}" source="{source_node_id}" target="{dest_node_id}">
    </edge>
'''
                        xmlfile.write(edge_xml)
                        edge_id += 1  # Increment edge ID

        # Write footer
        xml_footer = '''  </graph>
</xml>
'''
        xmlfile.write(xml_footer)

if __name__ == '__main__':
    main()
