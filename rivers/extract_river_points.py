#!/usr/bin/env python3
"""
Script to extract all coordinate points that rivers travel through.
Each river segment (LineString) contains multiple coordinate points.
You can trace individual rivers or export all points.
"""

import os
import sys

def extract_points_geopandas():
    """Extract river points using geopandas"""
    try:
        import geopandas as gpd
        import pandas as pd
        import json
        
        print("=" * 80)
        print("RIVER COORDINATE POINTS EXTRACTOR")
        print("=" * 80)
        
        shapefile_path = os.path.join(os.path.dirname(__file__), 'HydroRIVERS_v10_ar.shp')
        print(f"\nLoading shapefile: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        
        print(f"Loaded {len(gdf)} river segments")
        
        # Main menu
        while True:
            print("\n" + "=" * 80)
            print("OPTIONS")
            print("=" * 80)
            print("1. Extract points from a specific river segment (by index)")
            print("2. Extract points from a river by HYRIV_ID")
            print("3. Trace entire river network (follow NEXT_DOWN connections)")
            print("4. Export all points from all rivers to CSV")
            print("5. Export all points from all rivers to GeoJSON")
            print("6. Show statistics about river points")
            print("7. Find longest rivers")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == '1':
                extract_by_index(gdf)
            elif choice == '2':
                extract_by_id(gdf)
            elif choice == '3':
                trace_river_network(gdf)
            elif choice == '4':
                export_all_points_csv(gdf)
            elif choice == '5':
                export_all_points_geojson(gdf)
            elif choice == '6':
                show_point_statistics(gdf)
            elif choice == '7':
                find_longest_rivers(gdf)
            elif choice == '8':
                print("\nExiting...")
                break
            else:
                print("Invalid choice. Please try again.")
        
        return True
        
    except ImportError:
        print("Error: geopandas not installed. Please run: pip install geopandas")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_by_index(gdf):
    """Extract points from a specific river segment by index"""
    try:
        idx = int(input(f"Enter river segment index (0 to {len(gdf)-1}): "))
        if idx < 0 or idx >= len(gdf):
            print(f"Invalid index. Must be between 0 and {len(gdf)-1}")
            return
        
        river = gdf.iloc[idx]
        coords = list(river.geometry.coords)
        
        print(f"\n--- River Segment {idx} ---")
        print(f"HYRIV_ID: {river['HYRIV_ID']}")
        print(f"Length: {river['LENGTH_KM']} km")
        print(f"Number of coordinate points: {len(coords)}")
        print(f"NEXT_DOWN: {river['NEXT_DOWN']}")
        print(f"MAIN_RIV: {river['MAIN_RIV']}")
        
        print(f"\nCoordinate points (Longitude, Latitude):")
        for i, (lon, lat) in enumerate(coords):
            print(f"  Point {i}: ({lon:.6f}, {lat:.6f})")
        
        # Option to save
        save = input("\nSave these points to a file? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"river_{river['HYRIV_ID']}_points.csv"
            filepath = os.path.join(os.path.dirname(__file__), filename)
            with open(filepath, 'w') as f:
                f.write("point_index,longitude,latitude\n")
                for i, (lon, lat) in enumerate(coords):
                    f.write(f"{i},{lon:.6f},{lat:.6f}\n")
            print(f"Saved to: {filepath}")
            
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"Error: {e}")


def extract_by_id(gdf):
    """Extract points from a river by HYRIV_ID"""
    try:
        hyriv_id = int(input("Enter HYRIV_ID: "))
        river = gdf[gdf['HYRIV_ID'] == hyriv_id]
        
        if len(river) == 0:
            print(f"No river found with HYRIV_ID: {hyriv_id}")
            return
        
        river = river.iloc[0]
        coords = list(river.geometry.coords)
        
        print(f"\n--- River HYRIV_ID: {hyriv_id} ---")
        print(f"Length: {river['LENGTH_KM']} km")
        print(f"Number of coordinate points: {len(coords)}")
        print(f"NEXT_DOWN: {river['NEXT_DOWN']}")
        print(f"Stream Order: {river['ORD_STRA']}")
        
        print(f"\nCoordinate points (Longitude, Latitude):")
        for i, (lon, lat) in enumerate(coords):
            print(f"  Point {i}: ({lon:.6f}, {lat:.6f})")
            
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"Error: {e}")


def trace_river_network(gdf):
    """Trace an entire river network following NEXT_DOWN connections"""
    try:
        start_id = int(input("Enter starting HYRIV_ID: "))
        
        # Build a lookup dictionary for faster access
        river_dict = {row['HYRIV_ID']: row for _, row in gdf.iterrows()}
        
        if start_id not in river_dict:
            print(f"No river found with HYRIV_ID: {start_id}")
            return
        
        print(f"\n--- Tracing river network from HYRIV_ID: {start_id} ---")
        
        current_id = start_id
        all_points = []
        segment_count = 0
        total_length = 0
        
        visited = set()  # Prevent infinite loops
        
        while current_id != 0 and current_id not in visited:
            if current_id not in river_dict:
                print(f"Warning: River segment {current_id} not found in dataset")
                break
            
            visited.add(current_id)
            river = river_dict[current_id]
            coords = list(river.geometry.coords)
            
            segment_count += 1
            total_length += river['LENGTH_KM']
            
            print(f"\nSegment {segment_count}:")
            print(f"  HYRIV_ID: {current_id}")
            print(f"  Length: {river['LENGTH_KM']} km")
            print(f"  Points: {len(coords)}")
            print(f"  Next: {river['NEXT_DOWN']}")
            
            # Add points (avoiding duplicates at segment connections)
            if segment_count == 1:
                all_points.extend(coords)
            else:
                # Skip first point if it duplicates the last point from previous segment
                all_points.extend(coords[1:])
            
            current_id = river['NEXT_DOWN']
        
        print(f"\n{'=' * 80}")
        print(f"SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total segments: {segment_count}")
        print(f"Total length: {total_length:.2f} km")
        print(f"Total unique points: {len(all_points)}")
        
        # Option to save
        save = input("\nSave complete river path to a file? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"river_network_{start_id}_complete.csv"
            filepath = os.path.join(os.path.dirname(__file__), filename)
            with open(filepath, 'w') as f:
                f.write("point_index,longitude,latitude,segment_number\n")
                segment_num = 0
                points_in_segment = 0
                for i, (lon, lat) in enumerate(all_points):
                    f.write(f"{i},{lon:.6f},{lat:.6f},{segment_num}\n")
            print(f"Saved to: {filepath}")
            
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def export_all_points_csv(gdf):
    """Export all points from all rivers to CSV"""
    try:
        print("\nThis will extract all coordinate points from all river segments.")
        print(f"Total segments: {len(gdf)}")
        confirm = input("This may take a while and create a large file. Continue? (y/n): ").strip().lower()
        
        if confirm != 'y':
            return
        
        filename = "all_river_points.csv"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        print("\nExtracting points...")
        total_points = 0
        
        with open(filepath, 'w') as f:
            f.write("river_id,segment_index,point_index,longitude,latitude,length_km,stream_order\n")
            
            for idx, river in gdf.iterrows():
                coords = list(river.geometry.coords)
                for point_idx, (lon, lat) in enumerate(coords):
                    f.write(f"{river['HYRIV_ID']},{idx},{point_idx},{lon:.6f},{lat:.6f},"
                           f"{river['LENGTH_KM']},{river['ORD_STRA']}\n")
                    total_points += 1
                
                if (idx + 1) % 10000 == 0:
                    print(f"  Processed {idx + 1} / {len(gdf)} segments...")
        
        print(f"\nComplete!")
        print(f"Total points extracted: {total_points}")
        print(f"Saved to: {filepath}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def export_all_points_geojson(gdf):
    """Export all river segments with their points to GeoJSON"""
    try:
        print("\nExporting to GeoJSON format...")
        filename = "all_rivers.geojson"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # Limit for testing
        limit = input("Export all rivers? (y/n, if n you can specify a limit): ").strip().lower()
        if limit == 'n':
            max_features = int(input("How many river segments to export? "))
            export_gdf = gdf.head(max_features)
        else:
            export_gdf = gdf
        
        export_gdf.to_file(filepath, driver='GeoJSON')
        print(f"Saved {len(export_gdf)} river segments to: {filepath}")
        
    except ValueError:
        print("Invalid input.")
    except Exception as e:
        print(f"Error: {e}")


def show_point_statistics(gdf):
    """Show statistics about points in river segments"""
    print("\nCalculating point statistics...")
    
    point_counts = []
    for _, river in gdf.iterrows():
        coords = list(river.geometry.coords)
        point_counts.append(len(coords))
    
    import statistics
    
    print(f"\n{'=' * 80}")
    print("POINT STATISTICS")
    print(f"{'=' * 80}")
    print(f"Total river segments: {len(gdf)}")
    print(f"Total points across all rivers: {sum(point_counts)}")
    print(f"\nPoints per segment:")
    print(f"  Minimum: {min(point_counts)}")
    print(f"  Maximum: {max(point_counts)}")
    print(f"  Mean: {statistics.mean(point_counts):.2f}")
    print(f"  Median: {statistics.median(point_counts):.2f}")
    
    # Show some examples
    print(f"\nExamples of segments with many points:")
    # Get indices of top 5 segments with most points
    sorted_indices = sorted(range(len(point_counts)), key=lambda i: point_counts[i], reverse=True)[:5]
    for i, idx in enumerate(sorted_indices, 1):
        river = gdf.iloc[idx]
        print(f"  {i}. HYRIV_ID {river['HYRIV_ID']}: {point_counts[idx]} points, "
              f"{river['LENGTH_KM']:.2f} km long")


def find_longest_rivers(gdf):
    """Find the longest rivers by total length"""
    print("\nFinding longest rivers...")
    
    # Group by MAIN_RIV to find river systems
    river_systems = gdf.groupby('MAIN_RIV').agg({
        'LENGTH_KM': 'sum',
        'HYRIV_ID': 'count'
    }).reset_index()
    
    river_systems.columns = ['MAIN_RIV', 'Total_Length_KM', 'Segment_Count']
    river_systems = river_systems.sort_values('Total_Length_KM', ascending=False)
    
    print(f"\n{'=' * 80}")
    print("TOP 10 LONGEST RIVER SYSTEMS")
    print(f"{'=' * 80}")
    print(f"{'Rank':<6} {'Main River ID':<15} {'Total Length (km)':<20} {'Segments':<10}")
    print("-" * 80)
    
    for i, row in river_systems.head(10).iterrows():
        print(f"{i+1:<6} {int(row['MAIN_RIV']):<15} {row['Total_Length_KM']:<20.2f} {int(row['Segment_Count']):<10}")
    
    # Allow user to select one to trace
    trace = input("\nTrace one of these river systems? Enter MAIN_RIV ID (or 0 to skip): ").strip()
    if trace != '0':
        try:
            main_riv_id = int(trace)
            # Find the furthest upstream segment of this river system
            river_segments = gdf[gdf['MAIN_RIV'] == main_riv_id]
            # The one with max DIST_UP_KM is furthest upstream
            start_segment = river_segments.loc[river_segments['DIST_UP_KM'].idxmax()]
            print(f"\nStarting trace from HYRIV_ID: {start_segment['HYRIV_ID']} (furthest upstream)")
            
            # Simulate the trace function call
            print("\nWould trace through entire river system...")
            print(f"Total segments to trace: {len(river_segments)}")
            print(f"Total length: {river_segments['LENGTH_KM'].sum():.2f} km")
            
        except:
            print("Invalid input.")


def main():
    """Main function"""
    if not extract_points_geopandas():
        print("\nFalling back to basic extraction with fiona...")
        extract_points_fiona()


def extract_points_fiona():
    """Fallback extraction using fiona"""
    try:
        import fiona
        
        shapefile_path = os.path.join(os.path.dirname(__file__), 'HydroRIVERS_v10_ar.shp')
        
        idx = int(input("Enter river segment index to extract points: "))
        
        with fiona.open(shapefile_path, 'r') as src:
            if idx < 0 or idx >= len(src):
                print(f"Invalid index. Must be between 0 and {len(src)-1}")
                return
            
            for i, feature in enumerate(src):
                if i == idx:
                    coords = feature['geometry']['coordinates']
                    print(f"\nRiver segment {idx}:")
                    print(f"HYRIV_ID: {feature['properties']['HYRIV_ID']}")
                    print(f"Number of points: {len(coords)}")
                    print(f"\nCoordinates (Longitude, Latitude):")
                    for j, (lon, lat) in enumerate(coords):
                        print(f"  Point {j}: ({lon:.6f}, {lat:.6f})")
                    break
                    
    except ImportError:
        print("Please install geopandas or fiona: pip install geopandas")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
