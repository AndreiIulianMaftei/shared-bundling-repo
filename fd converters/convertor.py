import xml.etree.ElementTree as ET
import argparse

def remove_namespace(tag):
    return tag.split('}')[-1] if '}' in tag else tag

def transform_airlines_to_japan(input_file, output_file):
    # Load the XML files
    airlines_tree = ET.parse(input_file)
    airlines_root = airlines_tree.getroot()
    
    # Remove namespace prefixes
    for elem in airlines_root.iter():
        elem.tag = remove_namespace(elem.tag)
    
    # Ensure we locate the graph element in the Airlines XML
    airlines_graph = airlines_root.find(".//graph")
    if airlines_graph is None:
        print("Error: No <graph> element found in the input file.")
        return

    # Remove all existing key definitions
    for key in airlines_root.findall("key"):
        airlines_root.remove(key)

    # Define new keys matching the Japan format
    new_keys = [
        {"id": "x", "for": "node", "attr.name": "x", "attr.type": "double"},
        {"id": "y", "for": "node", "attr.name": "y", "attr.type": "double"},
        {"id": "tooltip", "for": "node", "attr.name": "tooltip", "attr.type": "string"}
    ]

    for key_attrs in new_keys:
        new_key = ET.Element("key", key_attrs)
        airlines_root.append(new_key)

    # Process nodes and transform data structure
    for node in airlines_graph.findall("node"):
        data_elements = {data.get("key"): data for data in node.findall("data")}
        
        # Extract old attributes
        x_val = data_elements.get("d4").text if "d4" in data_elements else "0.0"
        y_val = data_elements.get("d5").text if "d5" in data_elements else "0.0"
        tooltip_val = f"ID({node.get('id')})"  # Placeholder tooltip
        
        # Remove old data elements
        for key in ["d4", "d5", "d6", "d7"]:
            if key in data_elements:
                node.remove(data_elements[key])
        
        # Add new attributes following Japan format
        ET.SubElement(node, "data", key="x").text = x_val
        ET.SubElement(node, "data", key="y").text = y_val
        ET.SubElement(node, "data", key="tooltip").text = tooltip_val
    
    # Save the transformed XML file
    airlines_tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Transformed XML saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Airlines XML format to Japan XML format.")
    parser.add_argument("input_file", help="Path to the airlines_undirected.xml file")
    parser.add_argument("output_file", help="Path to save the transformed file")
    args = parser.parse_args()
    
    transform_airlines_to_japan(args.input_file, args.output_file)
