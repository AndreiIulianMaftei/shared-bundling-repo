#!/usr/bin/env python3
"""
Convert Twin Towns CSV to GraphML XML format.

This script reads the twin towns CSV file and converts it to the same XML format
as the airports.xml file, creating nodes for each town and edges for twin relationships.
"""

import csv
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Set
import os


def clean_coordinate(coord_str: str) -> float:
    """Clean and convert coordinate string to float."""
    try:
        return float(coord_str.strip())
    except (ValueError, AttributeError):
        return 0.0


def process_coordinates(longitude: float, latitude: float) -> Tuple[float, float]:
    """
    Process coordinates to match the airports.xml format.
    
    In airports.xml:
    - x coordinate seems to be longitude scaled/shifted
    - y coordinate seems to be latitude scaled/shifted
    
    Looking at the airport data, it appears:
    - Negative longitudes become positive x values (longitude * -10)
    - Positive latitudes become positive y values (latitude * 10)
    """
    # Convert longitude: negative becomes positive, scale by 10
    x = abs(longitude) * 10
    
    # Convert latitude: scale by 10
    y = latitude * 10
    
    return x, y


def create_tooltip(town_name: str, longitude: float, latitude: float) -> str:
    """Create tooltip string with just the town name."""
    return town_name


def read_twin_towns_csv(file_path: str) -> Tuple[Dict[str, Dict], List[Tuple[str, str]]]:
    """
    Read the twin towns CSV and extract unique towns and twin relationships.
    
    Returns:
        Tuple of (towns_dict, twin_pairs)
        - towns_dict: {town_key: {name, country, lat, lng}}
        - twin_pairs: [(uk_town_key, twin_town_key), ...]
    """
    towns = {}
    twin_pairs = []
    
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            # Extract UK town data
            uk_town = row['UK town'].strip()
            uk_lat = clean_coordinate(row['UK town Latitude'])
            uk_lng = clean_coordinate(row['UK town Longitude'])
            uk_key = f"{uk_town}_UK"
            
            # Extract twin town data
            twin_town_full = row['TWIN TOWN'].strip()
            twin_country = row['TWIN TOWN COUNTRY'].strip()
            twin_lat = clean_coordinate(row['Twin town Latitude'])
            twin_lng = clean_coordinate(row['Twin town Longitude'])
            twin_key = f"{twin_town_full}_{twin_country}"
            
            # Add UK town to towns dict
            if uk_key not in towns:
                towns[uk_key] = {
                    'name': uk_town,
                    'country': 'United Kingdom',
                    'latitude': uk_lat,
                    'longitude': uk_lng
                }
            
            # Add twin town to towns dict
            if twin_key not in towns:
                towns[twin_key] = {
                    'name': twin_town_full,
                    'country': twin_country,
                    'latitude': twin_lat,
                    'longitude': twin_lng
                }
            
            # Add twin relationship
            twin_pairs.append((uk_key, twin_key))
    
    return towns, twin_pairs


def create_graphml_xml(towns: Dict[str, Dict], twin_pairs: List[Tuple[str, str]]) -> ET.Element:
    """Create the GraphML XML structure."""
    
    # Create root element with namespaces
    xml_root = ET.Element('xml')
    xml_root.set('xmlns', 'http://graphml.graphdrawing.org/xmlns')
    xml_root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    xml_root.set('xsi:schemaLocation', 
                 'http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd')
    
    # Add key definitions
    x_key = ET.SubElement(xml_root, 'key')
    x_key.set('id', 'x')
    x_key.set('for', 'node')
    x_key.set('attr.name', 'x')
    x_key.set('attr.type', 'double')
    
    tooltip_key = ET.SubElement(xml_root, 'key')
    tooltip_key.set('id', 'tooltip')
    tooltip_key.set('for', 'node')
    tooltip_key.set('attr.name', 'tooltip')
    tooltip_key.set('attr.type', 'string')
    
    y_key = ET.SubElement(xml_root, 'key')
    y_key.set('id', 'y')
    y_key.set('for', 'node')
    y_key.set('attr.name', 'y')
    y_key.set('attr.type', 'double')
    
    # Create graph element
    graph = ET.SubElement(xml_root, 'graph')
    graph.set('edgedefault', 'undirected')
    
    # Add nodes comment
    nodes_comment = ET.Comment(' nodes ')
    graph.append(nodes_comment)
    
    # Create town key to node id mapping
    town_keys = list(towns.keys())
    town_to_id = {key: idx for idx, key in enumerate(town_keys)}
    
    # Add nodes
    for town_key, town_data in towns.items():
        node_id = town_to_id[town_key]
        
        # Process coordinates
        x, y = process_coordinates(town_data['longitude'], town_data['latitude'])
        
        # Create node
        node = ET.SubElement(graph, 'node')
        node.set('id', str(node_id))
        
        # Add x coordinate
        x_data = ET.SubElement(node, 'data')
        x_data.set('key', 'x')
        x_data.text = str(x)
        
        # Add tooltip
        tooltip_data = ET.SubElement(node, 'data')
        tooltip_data.set('key', 'tooltip')
        tooltip_data.text = create_tooltip(town_data['name'], town_data['longitude'], town_data['latitude'])
        
        # Add y coordinate
        y_data = ET.SubElement(node, 'data')
        y_data.set('key', 'y')
        y_data.text = str(y)
    
    # Add edges comment
    edges_comment = ET.Comment(' edges ')
    graph.append(edges_comment)
    
    # Add edges for twin relationships
    for edge_id, (town1_key, town2_key) in enumerate(twin_pairs):
        edge = ET.SubElement(graph, 'edge')
        edge.set('id', str(edge_id))
        edge.set('source', str(town_to_id[town1_key]))
        edge.set('target', str(town_to_id[town2_key]))
    
    return xml_root


def format_xml(element: ET.Element) -> str:
    """Format XML with proper indentation."""
    from xml.dom import minidom
    
    rough_string = ET.tostring(element, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ", encoding=None)


def main():
    """Main function to convert CSV to XML."""
    # Define file paths
    csv_file = '/home/andrei2/c++/shared-bundling-repo/twintowns/kepler.gl_Final All Twin Towns Geocoded Excel 2.csv.csv'
    xml_file = '/home/andrei2/c++/shared-bundling-repo/twintowns/twin_towns.xml'
    
    print("Reading twin towns CSV file...")
    towns, twin_pairs = read_twin_towns_csv(csv_file)
    
    print(f"Found {len(towns)} unique towns and {len(twin_pairs)} twin relationships.")
    
    print("Creating GraphML XML structure...")
    xml_root = create_graphml_xml(towns, twin_pairs)
    
    print("Formatting and writing XML file...")
    formatted_xml = format_xml(xml_root)
    
    # Write to file
    with open(xml_file, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        # Skip the first line of the formatted XML (which includes XML declaration)
        f.write('\n'.join(formatted_xml.split('\n')[1:]))
    
    print(f"Successfully created {xml_file}")
    print(f"Statistics:")
    print(f"  - Total towns: {len(towns)}")
    print(f"  - Twin relationships: {len(twin_pairs)}")
    
    # Show sample of UK towns
    uk_towns = [key for key in towns.keys() if key.endswith('_UK')]
    print(f"  - UK towns: {len(uk_towns)}")
    
    # Show sample of countries
    countries = set()
    for town_data in towns.values():
        countries.add(town_data['country'])
    print(f"  - Countries represented: {len(countries)}")
    print(f"  - Sample countries: {sorted(list(countries))[:10]}")


if __name__ == "__main__":
    main()