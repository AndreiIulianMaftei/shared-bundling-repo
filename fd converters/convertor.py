import xml.etree.ElementTree as ET

def convert_graphml(input_path, output_path):
    # Namespaces
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

    # Parse the input GraphML
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Create a new root element for the output GraphML
    new_root = ET.Element("graphml", {
        "xmlns": "http://graphml.graphdrawing.org/xmlns",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://graphml.graphdrawing.org/xmlns "
                              "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"
    })

    # Define the keys used in the second format
    ET.SubElement(new_root, "key", {
        "id": "X",
        "for": "node",
        "attr.name": "X",
        "attr.type": "double"
    })
    ET.SubElement(new_root, "key", {
        "id": "Y",
        "for": "node",
        "attr.name": "Y",
        "attr.type": "double"
    })
    ET.SubElement(new_root, "key", {
        "id": "tooltip",
        "for": "node",
        "attr.name": "tooltip",
        "attr.type": "string"
    })

    # Create a <graph> element
    new_graph = ET.SubElement(new_root, "graph", {
        "edgedefault": "undirected"
    })

    # Locate the original <graph> element
    old_graph = root.find("g:graph", ns)
    if old_graph is None:
        raise ValueError("No <graph> element found in the input file.")

    # Convert each node
    for old_node in old_graph.findall("g:node", ns):
        node_id = old_node.get("id")
        x_value, y_value = None, None

        # Extract x/y data from the old node
        for data_el in old_node.findall("g:data", ns):
            data_key = data_el.get("key")
            text_val = data_el.text
            if data_key == "d6":  # Or "d4", or whichever key has x
                x_value = text_val
            elif data_key == "d7":  # Or "d5", or whichever key has y
                y_value = text_val

        # Build the new node
        new_node = ET.SubElement(new_graph, "node", {"id": node_id})
        if x_value is not None:
            x_data = ET.SubElement(new_node, "data", {"key": "X"})
            x_data.text = x_value
        if y_value is not None:
            y_data = ET.SubElement(new_node, "data", {"key": "Y"})
            y_data.text = y_value

        # Example tooltip: just store the node id
        tooltip_data = ET.SubElement(new_node, "data", {"key": "tooltip"})
        tooltip_data.text = node_id

    # Convert each edge
    for old_edge in old_graph.findall("g:edge", ns):
        source = old_edge.get("source")
        target = old_edge.get("target")
        # Create new edge in the new graph
        _ = ET.SubElement(new_graph, "edge", {
            "source": source,
            "target": target
        })
        # If you want to preserve distance (key="d8") as a data field,
        # create a <data> element here. Example:
        #
        # dist_val = old_edge.find("g:data[@key='d8']", ns)
        # if dist_val is not None:
        #     dist_data = ET.SubElement(new_edge, "data", {"key": "dist"})
        #     dist_data.text = dist_val.text

    # Use ElementTree’s indent for pretty-printing (Python 3.9+)
    ET.indent(new_root, space="  ")

    # Write out the new GraphML, adding an XML declaration
    new_tree = ET.ElementTree(new_root)
    new_tree.write(output_path, encoding="utf-8", xml_declaration=True)

if __name__ == "__main__":
    input_file = "fd converters/first_format.graphml"    # your input file
    output_file = "second_format.graphml"  # your output file
    convert_graphml(input_file, output_file)
    print(f"Converted {input_file} to {output_file} with line breaks and indentation.")
