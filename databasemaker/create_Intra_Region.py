import csv
import sys

def escape_xml(text):
    """
    Escape special characters in text for XML.
    """
    return (text.replace("&", "&amp;")
                .replace('"', "&quot;")
                .replace("'", "&apos;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

def get_continent_from_timezone(tz_value):
    """
    Given a tz database string like "Pacific/Port_Moresby",
    return a broad 'continent' classification.

    You can expand or modify the dictionary below as needed.
    """
    # Extract the top-level part of the timezone (split at the '/')
    # e.g., "Pacific" from "Pacific/Port_Moresby"
    if "/" not in tz_value:
        return None  # Malformed or unexpected tz
    top_level = tz_value.split("/")[0]

    # Simple mapping from tz top-level to a "continent" name
    TZ_TO_CONTINENT = {
        "Africa": "Africa",
        "America": "America",      # Covers both North & South America in tz database
        "Antarctica": "Antarctica",
        "Arctic": "Arctic",        # Rare; you can lump with "Europe" if desired
        "Asia": "Asia",
        "Atlantic": "Atlantic",    # Some islands, can be mapped to "Europe" or "America" if you prefer
        "Australia": "Oceania",    # We often consider "Australia" as part of Oceania
        "Europe": "Europe",
        "Indian": "Asia",          # Or "Oceania", depending on your preference
        "Pacific": "Oceania"
    }

    return TZ_TO_CONTINENT.get(top_level, None)

def main():
    # We expect a single argument with the name of the “continent” we want to filter by
    # E.g. python script.py Oceania
    if len(sys.argv) < 2:
        print("Usage: python script.py <Continent>")
        print("Example: python script.py Oceania")
        sys.exit(1)

    target_continent = "Africa"

    # CSV headers based on your data
    airport_headers = [
        "Airport ID", "Name", "City", "Country", "IATA", "ICAO",
        "Latitude", "Longitude", "Altitude", "Timezone", "DST",
        "Tz database timezone", "Type", "Source"
    ]

    route_headers = [
        "Airline", "Airline ID", "Source airport", "Source airport ID",
        "Destination airport", "Destination airport ID", "Codeshare",
        "Stops", "Equipment"
    ]

    # Mapping: Airport ID -> node ID in GraphML
    airport_id_to_node_id = {}
    node_id = 0

    # Keep track of which airport IDs belong to the user-specified continent
    selected_airport_ids = set()

    # Output file named by continent
    output_filename = f"airports_{target_continent}.xml"

    with open(output_filename, 'w', encoding='UTF-8') as xmlfile:
        # Write GraphML header
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

        #
        # Step 1: Process airports and generate nodes
        #
        with open('./databasemaker/airport_data.csv', 'r', encoding='utf-8') as csvfile_airport:
            reader = csv.reader(csvfile_airport, delimiter=',')
            
            for row in reader:
                # Skip rows that are too short or empty
                if not row or len(row) < len(airport_headers):
                    continue

                airport_data = dict(zip(airport_headers, row))

                airport_id = airport_data['Airport ID'].strip().strip('"')
                longitude = airport_data['Longitude'].strip().strip('"')
                latitude = airport_data['Latitude'].strip().strip('"')
                iata = airport_data['IATA'].strip().strip('"')
                tz_db = airport_data['Tz database timezone'].strip().strip('"')

                # Determine continent from timezone
                inferred_continent = get_continent_from_timezone(tz_db)
                if not inferred_continent:
                    # If we can’t parse or map, skip
                    continue

                # Filter: only include this airport if it matches the user’s requested continent
                if inferred_continent != target_continent:
                    continue

                # Skip if critical fields are missing
                if not airport_id or not longitude or not latitude or not iata or iata == '\\N':
                    continue

                try:
                    longitude = float(longitude)
                    latitude = float(latitude)
                except ValueError:
                    continue  # Skip invalid coordinates

                # Scale or transform coordinates if needed
                x = -longitude * 10
                y = latitude * 10

                # Prepare tooltip
                tooltip = f"{iata} (lon={longitude}, lat={latitude})"
                tooltip = escape_xml(tooltip)

                # Write <node> element
                node_xml = f'''    <node id="{node_id}">
      <data key="x">{x}</data>
      <data key="tooltip">{tooltip}</data>
      <data key="y">{y}</data>
    </node>
'''
                xmlfile.write(node_xml)

                # Record the node ID mapping and track the airport as “selected”
                airport_id_to_node_id[airport_id] = node_id
                selected_airport_ids.add(airport_id)
                node_id += 1

        #
        # Step 2: Process routes and generate edges
        #
        xmlfile.write('\n    <!-- edges -->\n')
        edge_id = 0
        added_edges = set()

        with open('./databasemaker/routes_data.csv', 'r', encoding='utf-8') as csvfile_routes:
            reader = csv.reader(csvfile_routes, delimiter=',')
            for row in reader:
                # Skip rows that are too short
                if not row or len(row) < len(route_headers):
                    continue

                route_data = dict(zip(route_headers, row))

                source_id = route_data['Source airport ID'].strip().strip('"')
                dest_id = route_data['Destination airport ID'].strip().strip('"')

                if not source_id or not dest_id:
                    continue

                # Only create an edge if both airports are in the same selected continent
                if source_id in selected_airport_ids and dest_id in selected_airport_ids:
                    src_node = airport_id_to_node_id[source_id]
                    dst_node = airport_id_to_node_id[dest_id]

                    # Avoid duplicates in an undirected graph
                    edge_nodes = frozenset([src_node, dst_node])
                    if edge_nodes in added_edges:
                        continue

                    added_edges.add(edge_nodes)

                    edge_xml = f'''    <edge id="{edge_id}" source="{src_node}" target="{dst_node}">
    </edge>
'''
                    xmlfile.write(edge_xml)
                    edge_id += 1

        # Write closing tags
        xml_footer = '''  </graph>
</xml>
'''
        xmlfile.write(xml_footer)

    print(f"GraphML file '{output_filename}' created for continent: {target_continent}")

if __name__ == '__main__':
    main()
