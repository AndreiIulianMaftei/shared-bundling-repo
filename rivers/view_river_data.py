#!/usr/bin/env python3
"""
Script to view and explore the HydroRIVERS shapefile data.
This script displays information about the river network data including
attributes, geometry types, and sample records.
"""

import os
import sys

def view_with_geopandas():
    """View shapefile data using geopandas (recommended method)"""
    try:
        import geopandas as gpd
        import pandas as pd
        
        # Set display options for better readability
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        print("=" * 80)
        print("HYDRORIVERS SHAPEFILE DATA VIEWER")
        print("=" * 80)
        
        # Read the shapefile
        shapefile_path = os.path.join(os.path.dirname(__file__), 'HydroRIVERS_v10_ar.shp')
        print(f"\nLoading shapefile: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        
        # Basic information
        print("\n" + "=" * 80)
        print("BASIC INFORMATION")
        print("=" * 80)
        print(f"Number of features: {len(gdf)}")
        print(f"Coordinate Reference System: {gdf.crs}")
        print(f"Geometry type: {gdf.geometry.type.unique()}")
        
        # Bounding box
        print("\n" + "=" * 80)
        print("BOUNDING BOX")
        print("=" * 80)
        bounds = gdf.total_bounds
        print(f"Min X (West):  {bounds[0]:.6f}")
        print(f"Min Y (South): {bounds[1]:.6f}")
        print(f"Max X (East):  {bounds[2]:.6f}")
        print(f"Max Y (North): {bounds[3]:.6f}")
        
        # Column information
        print("\n" + "=" * 80)
        print("COLUMNS/ATTRIBUTES")
        print("=" * 80)
        print("\nColumn names and data types:")
        for col in gdf.columns:
            if col != 'geometry':
                dtype = gdf[col].dtype
                non_null = gdf[col].notna().sum()
                print(f"  {col:<20} - {dtype:<10} ({non_null}/{len(gdf)} non-null)")
        
        # Summary statistics for numeric columns
        print("\n" + "=" * 80)
        print("NUMERIC COLUMN STATISTICS")
        print("=" * 80)
        numeric_cols = gdf.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            print(gdf[numeric_cols].describe().to_string())
        else:
            print("No numeric columns found.")
        
        # Sample data
        print("\n" + "=" * 80)
        print("SAMPLE DATA (First 5 records)")
        print("=" * 80)
        # Drop geometry column for better display
        sample_df = gdf.drop(columns=['geometry']).head(5)
        print(sample_df.to_string())
        
        # Additional statistics
        print("\n" + "=" * 80)
        print("ADDITIONAL INFORMATION")
        print("=" * 80)
        
        # Check for specific common HydroRIVERS attributes
        interesting_cols = ['ORD_STRA', 'ORD_CLAS', 'ORD_FLOW', 'LENGTH_KM', 'UPLAND_SKM', 'DIST_DN_KM']
        for col in interesting_cols:
            if col in gdf.columns:
                if gdf[col].dtype in ['float64', 'int64']:
                    print(f"\n{col}:")
                    print(f"  Min: {gdf[col].min()}")
                    print(f"  Max: {gdf[col].max()}")
                    print(f"  Mean: {gdf[col].mean():.2f}")
        
        # Interactive options
        print("\n" + "=" * 80)
        print("INTERACTIVE OPTIONS")
        print("=" * 80)
        print("\nWould you like to:")
        print("1. View more records")
        print("2. Filter by specific criteria")
        print("3. Export to CSV (without geometry)")
        print("4. Show detailed info about a specific record")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            try:
                n = int(input("How many records to display? "))
                print("\n" + gdf.drop(columns=['geometry']).head(n).to_string())
            except ValueError:
                print("Invalid number.")
        
        elif choice == '2':
            print("\nAvailable columns for filtering:")
            for i, col in enumerate([c for c in gdf.columns if c != 'geometry'], 1):
                print(f"  {i}. {col}")
            # User can extend this functionality
            print("\n(Filter functionality can be extended as needed)")
        
        elif choice == '3':
            output_path = os.path.join(os.path.dirname(__file__), 'river_data.csv')
            gdf.drop(columns=['geometry']).to_csv(output_path, index=False)
            print(f"\nData exported to: {output_path}")
        
        elif choice == '4':
            try:
                idx = int(input("Enter record index (0-based): "))
                if 0 <= idx < len(gdf):
                    print("\n" + "=" * 80)
                    record = gdf.iloc[idx]
                    for col in gdf.columns:
                        if col != 'geometry':
                            print(f"{col}: {record[col]}")
                        else:
                            print(f"{col}: {record[col].geom_type} with {len(record[col].coords)} coordinates")
                else:
                    print(f"Index out of range. Valid range: 0-{len(gdf)-1}")
            except ValueError:
                print("Invalid index.")
        
        return True
        
    except ImportError:
        return False
    except Exception as e:
        print(f"Error using geopandas: {e}")
        import traceback
        traceback.print_exc()
        return False


def view_with_fiona():
    """View shapefile data using fiona (alternative method)"""
    try:
        import fiona
        from collections import Counter
        
        print("=" * 80)
        print("HYDRORIVERS SHAPEFILE DATA VIEWER (using Fiona)")
        print("=" * 80)
        
        shapefile_path = os.path.join(os.path.dirname(__file__), 'HydroRIVERS_v10_ar.shp')
        
        with fiona.open(shapefile_path, 'r') as src:
            print(f"\nDriver: {src.driver}")
            print(f"CRS: {src.crs}")
            print(f"Number of features: {len(src)}")
            print(f"Bounds: {src.bounds}")
            
            print("\n" + "=" * 80)
            print("SCHEMA")
            print("=" * 80)
            print(f"Geometry type: {src.schema['geometry']}")
            print("\nProperties:")
            for prop, prop_type in src.schema['properties'].items():
                print(f"  {prop:<20} - {prop_type}")
            
            print("\n" + "=" * 80)
            print("SAMPLE FEATURES (First 5)")
            print("=" * 80)
            
            for i, feature in enumerate(src):
                if i >= 5:
                    break
                print(f"\nFeature {i}:")
                print(f"  Geometry: {feature['geometry']['type']}")
                print("  Properties:")
                for key, value in feature['properties'].items():
                    print(f"    {key}: {value}")
        
        return True
        
    except ImportError:
        return False
    except Exception as e:
        print(f"Error using fiona: {e}")
        import traceback
        traceback.print_exc()
        return False


def view_with_pyshp():
    """View shapefile data using pyshp (pure Python fallback)"""
    try:
        import shapefile
        
        print("=" * 80)
        print("HYDRORIVERS SHAPEFILE DATA VIEWER (using PyShp)")
        print("=" * 80)
        
        shapefile_path = os.path.join(os.path.dirname(__file__), 'HydroRIVERS_v10_ar.shp')
        
        sf = shapefile.Reader(shapefile_path)
        
        print(f"\nNumber of features: {len(sf)}")
        print(f"Shape type: {sf.shapeTypeName}")
        print(f"Bounding box: {sf.bbox}")
        
        print("\n" + "=" * 80)
        print("FIELDS/ATTRIBUTES")
        print("=" * 80)
        for i, field in enumerate(sf.fields[1:], 1):  # Skip deletion field
            print(f"{i}. {field[0]:<15} - Type: {field[1]}, Size: {field[2]}, Decimals: {field[3]}")
        
        print("\n" + "=" * 80)
        print("SAMPLE RECORDS (First 5)")
        print("=" * 80)
        
        for i, record in enumerate(sf.records()[:5]):
            print(f"\nRecord {i}:")
            for j, field in enumerate(sf.fields[1:]):
                field_name = field[0]
                print(f"  {field_name}: {record[j]}")
        
        return True
        
    except ImportError:
        return False
    except Exception as e:
        print(f"Error using pyshp: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to try different libraries"""
    
    # Try libraries in order of preference
    methods = [
        ("geopandas", view_with_geopandas),
        ("fiona", view_with_fiona),
        ("pyshp", view_with_pyshp)
    ]
    
    for lib_name, method in methods:
        try:
            if method():
                return
        except Exception as e:
            print(f"Failed to use {lib_name}: {e}")
            continue
    
    # If all methods failed
    print("\n" + "=" * 80)
    print("ERROR: No suitable library found!")
    print("=" * 80)
    print("\nTo view shapefile data, please install one of the following:")
    print("  1. geopandas:  pip install geopandas")
    print("  2. fiona:      pip install fiona")
    print("  3. pyshp:      pip install pyshp")
    print("\nRecommended: pip install geopandas (most feature-rich)")
    sys.exit(1)


if __name__ == "__main__":
    main()
