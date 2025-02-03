from lxml import etree

def invert_coordinates(input_file, output_file):
    # Parse the XML file
    tree = etree.parse(input_file)
    root = tree.getroot()

    # Namespaces used in the GraphML file
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}

    # XPath expressions to find nodes and data elements
    node_xpath = './/g:node'
    data_xpath = 'g:data'

    # Iterate over all nodes in the GraphML file
    for node in root.findall(node_xpath, namespaces=ns):
        # Iterate over all data elements within the node
        for data in node.findall(data_xpath, namespaces=ns):
            key = data.get('key')
            value = data.text
            if value is not None:
                try:
                    if key == 'x':
                        inverted_x = -float(value)
                        data.text = str(inverted_x)
                    elif key == 'y':
                        inverted_y = -float(value)
                        data.text = str(inverted_y)
                except ValueError:
                    print(f"Warning: Invalid coordinate '{value}' for key '{key}' in node '{node.get('id')}'. Skipping.")

    # Write the modified XML back to the output file
    tree.write(output_file, xml_declaration=True, encoding='UTF-8', pretty_print=True)

input_graphml = 'input_data/airlines.graphml'
output_graphml = 'input_data/flipped.graphml'
invert_coordinates(input_graphml, output_graphml)
