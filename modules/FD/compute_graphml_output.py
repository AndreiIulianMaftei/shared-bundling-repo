import xml.etree.ElementTree as ET

def indent(elem, level=0):
    """Function to indent XML elements for pretty-printing."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level+1)
        if not child.tail or not child.tail.strip():
            child.tail = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def main():
    nodes = {}
    edges = []

    # Read nodes.node file
    with open('experiment/other_code/FD/nodes.node', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                node_id = parts[0]
                x = - float(parts[1])
                y = float(parts[2])
                nodes[node_id] = {'x': x, 'y': y}

    # Read edges.edge file
    with open('experiment/other_code/FD/edges.edge', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                src = parts[0]
                dest = parts[1]
                coords = list(map(float, parts[2:]))
                x_coords = [-x for x in coords[::2]]
                y_coords = coords[1::2]
                edges.append({
                    'src': src,
                    'dest': dest,
                    'x_coords': x_coords,
                    'y_coords': y_coords
                })

    # Register namespaces
    ET.register_namespace('', "http://graphml.graphdrawing.org/xmlns")
    ET.register_namespace('xsi', "http://www.w3.org/2001/XMLSchema-instance")

    # Create the root element
    graphml = ET.Element('graphml', {
        'xmlns': "http://graphml.graphdrawing.org/xmlns",
        'xmlns:xsi': "http://www.w3.org/2001/XMLSchema-instance",
        'xsi:schemaLocation': "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"
    })

    # Add key elements
    ET.SubElement(graphml, 'key', {
        'id': 'd3',
        'for': 'edge',
        'attr.name': 'Spline_Y',
        'attr.type': 'string'
    })
    ET.SubElement(graphml, 'key', {
        'id': 'd2',
        'for': 'edge',
        'attr.name': 'Spline_X',
        'attr.type': 'string'
    })
    ET.SubElement(graphml, 'key', {
        'id': 'd1',
        'for': 'node',
        'attr.name': 'Y',
        'attr.type': 'double'
    })
    ET.SubElement(graphml, 'key', {
        'id': 'd0',
        'for': 'node',
        'attr.name': 'X',
        'attr.type': 'double'
    })

    # Create graph element
    graph = ET.SubElement(graphml, 'graph', {'edgedefault': 'undirected'})

    # Add nodes
    for node_id, coords in nodes.items():
        node_elem = ET.SubElement(graph, 'node', {'id': node_id})
        data_x = ET.SubElement(node_elem, 'data', {'key': 'd0'})
        data_x.text = str(coords['x'])
        data_y = ET.SubElement(node_elem, 'data', {'key': 'd1'})
        data_y.text = str(coords['y'])

    # Add edges
    for edge in edges:
        edge_elem = ET.SubElement(graph, 'edge', {
            'source': edge['src'],
            'target': edge['dest']
        })
        data_d2 = ET.SubElement(edge_elem, 'data', {'key': 'd2'})
        data_d2.text = ' '.join(map(str, edge['x_coords']))
        data_d3 = ET.SubElement(edge_elem, 'data', {'key': 'd3'})
        data_d3.text = ' '.join(map(str, edge['y_coords']))

    # Indent the XML elements for pretty-printing
    indent(graphml)

    # Write to result.graphml
    tree = ET.ElementTree(graphml)
    tree.write('output/airlines/fd.graphml', encoding='utf-8', xml_declaration=True)
    print("GraphML output written to output/airlines/fd.graphml")

if __name__ == '__main__':
    main()
