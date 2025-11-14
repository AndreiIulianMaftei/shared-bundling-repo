#!/usr/bin/env python3
"""
Script to organize river data by complete river systems.
Groups all segments belonging to the same river and exports organized data.
"""

import os
import sys

def organize_rivers_by_system():
    """Organize all river data by complete river systems"""
    try:
        import geopandas as gpd
        import pandas as pd
        from collections import defaultdict
        
        print("=" * 80)
        print("RIVER SYSTEM ORGANIZER")
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
            print("1. Show river systems summary")
            print("2. Export all rivers organized by system (CSV)")
            print("3. Export all rivers organized by system (JSON)")
            print("4. View details of a specific river system")
            print("5. Export individual river system to file")
            print("6. Create separate files per river system")
            print("7. Export river systems with all coordinates")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == '1':
                show_river_systems_summary(gdf)
            elif choice == '2':
                export_rivers_organized_csv(gdf)
            elif choice == '3':
                export_rivers_organized_json(gdf)
            elif choice == '4':
                view_river_system_details(gdf)
            elif choice == '5':
                export_single_river_system(gdf)
            elif choice == '6':
                create_files_per_river(gdf)
            elif choice == '7':
                export_rivers_with_coordinates(gdf)
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


def show_river_systems_summary(gdf):
    """Show summary of all river systems"""
    print("\nAnalyzing river systems...")
    
    # Group by MAIN_RIV
    river_systems = gdf.groupby('MAIN_RIV').agg({
        'HYRIV_ID': 'count',
        'LENGTH_KM': 'sum',
        'UPLAND_SKM': 'max',
        'ORD_STRA': 'max',
        'DIS_AV_CMS': 'max'
    }).reset_index()
    
    river_systems.columns = ['River_ID', 'Num_Segments', 'Total_Length_km', 
                              'Catchment_Area_km2', 'Max_Stream_Order', 'Max_Discharge_m3s']
    river_systems = river_systems.sort_values('Total_Length_km', ascending=False)
    
    print(f"\n{'=' * 80}")
    print(f"RIVER SYSTEMS SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total unique river systems: {len(river_systems)}")
    print(f"Total river segments: {len(gdf)}")
    print(f"\nRiver system size distribution:")
    print(f"  Single segment rivers: {(river_systems['Num_Segments'] == 1).sum()}")
    print(f"  2-10 segments: {((river_systems['Num_Segments'] >= 2) & (river_systems['Num_Segments'] <= 10)).sum()}")
    print(f"  11-100 segments: {((river_systems['Num_Segments'] >= 11) & (river_systems['Num_Segments'] <= 100)).sum()}")
    print(f"  100+ segments: {(river_systems['Num_Segments'] > 100).sum()}")
    
    print(f"\n{'=' * 80}")
    print("TOP 20 LONGEST RIVER SYSTEMS")
    print(f"{'=' * 80}")
    print(f"{'Rank':<6} {'River ID':<12} {'Segments':<10} {'Length (km)':<15} {'Max Order':<12} {'Catchment (km²)':<18}")
    print("-" * 80)
    
    for i, row in river_systems.head(20).iterrows():
        print(f"{i+1:<6} {int(row['River_ID']):<12} {int(row['Num_Segments']):<10} "
              f"{row['Total_Length_km']:<15.2f} {int(row['Max_Stream_Order']):<12} "
              f"{row['Catchment_Area_km2']:<18.2f}")
    
    # Save summary
    save = input("\nSave river systems summary to CSV? (y/n): ").strip().lower()
    if save == 'y':
        filepath = os.path.join(os.path.dirname(__file__), 'river_systems_summary.csv')
        river_systems.to_csv(filepath, index=False)
        print(f"Saved to: {filepath}")


def export_rivers_organized_csv(gdf):
    """Export all rivers organized by system to CSV"""
    print("\nOrganizing rivers by system...")
    
    # Sort by MAIN_RIV, then by DIST_DN_KM (downstream to upstream)
    organized_df = gdf.sort_values(['MAIN_RIV', 'DIST_DN_KM'], ascending=[True, False])
    
    # Drop geometry for CSV export
    export_df = organized_df.drop(columns=['geometry'])
    
    filepath = os.path.join(os.path.dirname(__file__), 'rivers_organized_by_system.csv')
    export_df.to_csv(filepath, index=False)
    
    print(f"\n✓ Exported {len(export_df)} segments organized by river system")
    print(f"  Saved to: {filepath}")
    print(f"  Segments are grouped by MAIN_RIV and sorted from downstream to upstream")


def export_rivers_organized_json(gdf):
    """Export all rivers organized by system to JSON"""
    import json
    import pandas as pd
    
    print("\nOrganizing rivers into JSON structure...")
    
    # Group by river system
    river_systems = {}
    
    for main_riv_id in gdf['MAIN_RIV'].unique():
        river_segments = gdf[gdf['MAIN_RIV'] == main_riv_id].sort_values('DIST_DN_KM', ascending=False)
        
        river_systems[int(main_riv_id)] = {
            'river_id': int(main_riv_id),
            'num_segments': len(river_segments),
            'total_length_km': float(river_segments['LENGTH_KM'].sum()),
            'max_stream_order': int(river_segments['ORD_STRA'].max()),
            'catchment_area_km2': float(river_segments['UPLAND_SKM'].max()),
            'segments': []
        }
        
        for idx, segment in river_segments.iterrows():
            segment_data = {
                'hyriv_id': int(segment['HYRIV_ID']),
                'next_down': int(segment['NEXT_DOWN']),
                'length_km': float(segment['LENGTH_KM']),
                'dist_downstream_km': float(segment['DIST_DN_KM']),
                'dist_upstream_km': float(segment['DIST_UP_KM']),
                'stream_order': int(segment['ORD_STRA']),
                'discharge_m3s': float(segment['DIS_AV_CMS']) if pd.notna(segment['DIS_AV_CMS']) else None
            }
            river_systems[int(main_riv_id)]['segments'].append(segment_data)
    
    filepath = os.path.join(os.path.dirname(__file__), 'rivers_organized_by_system.json')
    with open(filepath, 'w') as f:
        json.dump(river_systems, f, indent=2)
    
    print(f"\n✓ Exported {len(river_systems)} river systems to JSON")
    print(f"  Saved to: {filepath}")


def view_river_system_details(gdf):
    """View detailed information about a specific river system"""
    try:
        river_id = int(input("Enter MAIN_RIV ID to view: "))
        
        river_segments = gdf[gdf['MAIN_RIV'] == river_id].sort_values('DIST_UP_KM', ascending=False)
        
        if len(river_segments) == 0:
            print(f"No river system found with ID: {river_id}")
            return
        
        print(f"\n{'=' * 80}")
        print(f"RIVER SYSTEM {river_id} - COMPLETE DETAILS")
        print(f"{'=' * 80}")
        print(f"Total segments: {len(river_segments)}")
        print(f"Total length: {river_segments['LENGTH_KM'].sum():.2f} km")
        print(f"Maximum stream order: {river_segments['ORD_STRA'].max()}")
        print(f"Catchment area: {river_segments['UPLAND_SKM'].max():.2f} km²")
        print(f"Maximum discharge: {river_segments['DIS_AV_CMS'].max():.3f} m³/s")
        
        print(f"\n{'=' * 80}")
        print("ALL SEGMENTS (from upstream to downstream)")
        print(f"{'=' * 80}")
        print(f"{'#':<4} {'HYRIV_ID':<12} {'Next':<12} {'Length':<10} {'Order':<7} {'Dist DN':<10} {'Points':<8}")
        print("-" * 80)
        
        for i, (idx, segment) in enumerate(river_segments.iterrows(), 1):
            num_points = len(list(segment.geometry.coords))
            print(f"{i:<4} {segment['HYRIV_ID']:<12} {segment['NEXT_DOWN']:<12} "
                  f"{segment['LENGTH_KM']:<10.2f} {segment['ORD_STRA']:<7} "
                  f"{segment['DIST_DN_KM']:<10.2f} {num_points:<8}")
        
        print(f"\n{'=' * 80}")
        
    except ValueError:
        print("Invalid input. Please enter a number.")


def export_single_river_system(gdf):
    """Export a single river system with all its coordinates"""
    try:
        river_id = int(input("Enter MAIN_RIV ID to export: "))
        
        river_segments = gdf[gdf['MAIN_RIV'] == river_id].sort_values('DIST_UP_KM', ascending=False)
        
        if len(river_segments) == 0:
            print(f"No river system found with ID: {river_id}")
            return
        
        print(f"\nExporting river system {river_id}...")
        print(f"  Segments: {len(river_segments)}")
        print(f"  Total length: {river_segments['LENGTH_KM'].sum():.2f} km")
        
        # Export with coordinates
        filepath = os.path.join(os.path.dirname(__file__), f'river_system_{river_id}_complete.csv')
        
        with open(filepath, 'w') as f:
            f.write("segment_num,hyriv_id,next_down,segment_length_km,stream_order,"
                   "dist_downstream_km,point_index,longitude,latitude\n")
            
            total_points = 0
            for seg_num, (idx, segment) in enumerate(river_segments.iterrows(), 1):
                coords = list(segment.geometry.coords)
                for point_idx, (lon, lat) in enumerate(coords):
                    f.write(f"{seg_num},{segment['HYRIV_ID']},{segment['NEXT_DOWN']},"
                           f"{segment['LENGTH_KM']},{segment['ORD_STRA']},"
                           f"{segment['DIST_DN_KM']},{point_idx},{lon:.6f},{lat:.6f}\n")
                    total_points += 1
        
        print(f"\n✓ Exported river system with {total_points} coordinate points")
        print(f"  Saved to: {filepath}")
        
        # Also export as GeoJSON
        geojson_path = os.path.join(os.path.dirname(__file__), f'river_system_{river_id}.geojson')
        river_segments.to_file(geojson_path, driver='GeoJSON')
        print(f"  Also saved GeoJSON to: {geojson_path}")
        
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"Error: {e}")


def create_files_per_river(gdf):
    """Create separate files for each river system (or top N rivers)"""
    try:
        print("\nThis will create separate files for river systems.")
        
        choice = input("Export (a)ll rivers or (t)op N longest? (a/t): ").strip().lower()
        
        if choice == 't':
            n = int(input("How many of the longest rivers? "))
            # Get top N rivers by total length
            river_lengths = gdf.groupby('MAIN_RIV')['LENGTH_KM'].sum().sort_values(ascending=False)
            river_ids = river_lengths.head(n).index.tolist()
        else:
            river_ids = gdf['MAIN_RIV'].unique().tolist()
        
        print(f"\nCreating files for {len(river_ids)} river systems...")
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), 'river_systems')
        os.makedirs(output_dir, exist_ok=True)
        
        for i, river_id in enumerate(river_ids, 1):
            river_segments = gdf[gdf['MAIN_RIV'] == river_id].sort_values('DIST_UP_KM', ascending=False)
            
            filepath = os.path.join(output_dir, f'river_{int(river_id)}.csv')
            
            with open(filepath, 'w') as f:
                f.write("segment_num,hyriv_id,next_down,length_km,stream_order,"
                       "dist_downstream_km,dist_upstream_km,point_index,longitude,latitude\n")
                
                for seg_num, (idx, segment) in enumerate(river_segments.iterrows(), 1):
                    coords = list(segment.geometry.coords)
                    for point_idx, (lon, lat) in enumerate(coords):
                        f.write(f"{seg_num},{segment['HYRIV_ID']},{segment['NEXT_DOWN']},"
                               f"{segment['LENGTH_KM']},{segment['ORD_STRA']},"
                               f"{segment['DIST_DN_KM']},{segment['DIST_UP_KM']},"
                               f"{point_idx},{lon:.6f},{lat:.6f}\n")
            
            if i % 100 == 0:
                print(f"  Processed {i}/{len(river_ids)} rivers...")
        
        print(f"\n✓ Created {len(river_ids)} files in: {output_dir}")
        
    except ValueError:
        print("Invalid input.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def export_rivers_with_coordinates(gdf):
    """Export all river systems with their coordinates organized"""
    try:
        print("\nThis will create a comprehensive file with all rivers and coordinates.")
        print(f"Total river systems: {gdf['MAIN_RIV'].nunique()}")
        print(f"Total segments: {len(gdf)}")
        
        confirm = input("This may create a very large file. Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            return
        
        filepath = os.path.join(os.path.dirname(__file__), 'all_rivers_with_coordinates.csv')
        
        print("\nProcessing...")
        with open(filepath, 'w') as f:
            f.write("river_system_id,segment_num,hyriv_id,next_down,segment_length_km,"
                   "total_river_length_km,stream_order,dist_downstream_km,point_index,"
                   "longitude,latitude,num_points_in_segment\n")
            
            total_points = 0
            processed_rivers = 0
            
            # Group by river system
            for river_id in sorted(gdf['MAIN_RIV'].unique()):
                river_segments = gdf[gdf['MAIN_RIV'] == river_id].sort_values('DIST_UP_KM', ascending=False)
                total_river_length = river_segments['LENGTH_KM'].sum()
                
                for seg_num, (idx, segment) in enumerate(river_segments.iterrows(), 1):
                    coords = list(segment.geometry.coords)
                    num_points = len(coords)
                    
                    for point_idx, (lon, lat) in enumerate(coords):
                        f.write(f"{int(river_id)},{seg_num},{segment['HYRIV_ID']},"
                               f"{segment['NEXT_DOWN']},{segment['LENGTH_KM']},"
                               f"{total_river_length},{segment['ORD_STRA']},"
                               f"{segment['DIST_DN_KM']},{point_idx},"
                               f"{lon:.6f},{lat:.6f},{num_points}\n")
                        total_points += 1
                
                processed_rivers += 1
                if processed_rivers % 1000 == 0:
                    print(f"  Processed {processed_rivers}/{gdf['MAIN_RIV'].nunique()} river systems...")
        
        print(f"\n✓ Export complete!")
        print(f"  River systems: {processed_rivers}")
        print(f"  Total coordinate points: {total_points}")
        print(f"  Saved to: {filepath}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    organize_rivers_by_system()


if __name__ == "__main__":
    main()
