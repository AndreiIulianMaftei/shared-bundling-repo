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
    airport_headers = ["Airport ID", "Name", "City", "Country", "IATA", "ICAO", 
                       "Latitude", "Longitude", "Altitude", "Timezone", "DST", 
                       "Tz database timezone", "Type", "Source"]
    
    route_headers = ["Airline", "Airline ID", "Source airport", "Source airport ID", 
                     "Destination airport", "Destination airport ID", "Codeshare", 
                     "Stops", "Equipment"]
    
    airport_id_to_node_id = {}
    node_id = 0  

    russian_airport_ids = set()
    
    with open('airports.xml', 'w', encoding='UTF-8') as xmlfile:
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

        with open('airport_data.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            
            for row in reader:
                if not row or len(row) < len(airport_headers):
                    continue
                
                airport_data = dict(zip(airport_headers, row))
                
                airport_id = airport_data['Airport ID'].strip().strip('"')
                longitude = airport_data['Longitude'].strip().strip('"')
                latitude = airport_data['Latitude'].strip().strip('"')
                iata = airport_data['IATA'].strip().strip('"')
                country = airport_data['Country'].strip().strip('"')
                
                # Filter: Only include airports in Country
                if country != 'Japan':
                    continue

                if not airport_id or not longitude or not latitude or not iata or iata == '\\N':
                    continue
                
                try:
                    longitude = float(longitude)
                    latitude = float(latitude)
                except ValueError:
                    continue  # Skip rows with invalid data

                
                x = -longitude * 10
                y = latitude * 10

                tooltip = f"{iata}(lngx={longitude},laty={latitude})"
                tooltip = escape_xml(tooltip)

                node_xml = f'''    <node id="{node_id}">
      <data key="x">{x}</data>
      <data key="tooltip">{tooltip}</data>
      <data key="y">{y}</data>
    </node>
'''
                xmlfile.write(node_xml)

                airport_id_to_node_id[airport_id] = node_id
                russian_airport_ids.add(airport_id)
                node_id += 1  

        xmlfile.write('\n    <!-- edges -->\n')
        edge_id = 0 

        with open('routes_data.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            
            for row in reader:
                if not row or len(row) < len(route_headers):
                    continue

                route_data = dict(zip(route_headers, row))
                
                source_airport_id = route_data['Source airport ID'].strip().strip('"')
                dest_airport_id = route_data['Destination airport ID'].strip().strip('"')

                if not source_airport_id or not dest_airport_id:
                    continue

                if (source_airport_id in russian_airport_ids and 
                    dest_airport_id in russian_airport_ids):
                    
                    source_node_id = airport_id_to_node_id[source_airport_id]
                    dest_node_id = airport_id_to_node_id[dest_airport_id]

                    edge_xml = f'''    <edge id="{edge_id}" source="{source_node_id}" target="{dest_node_id}">
    </edge>
'''
                    xmlfile.write(edge_xml)
                    edge_id += 1  

        xml_footer = '''  </graph>
</xml>
'''
        xmlfile.write(xml_footer)

if __name__ == '__main__':
    main()

