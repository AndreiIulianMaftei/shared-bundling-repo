import os
import networkx as nx

def analyze_graphs(root_dir):
    list_3x = []
    list_4x = []
    total_graphs = 0

    print(f"Analyzing graphs in {root_dir}...")

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".graphml"):
                file_path = os.path.join(root, file)
                try:
                    G = nx.read_graphml(file_path)
                    num_nodes = G.number_of_nodes()
                    num_edges = G.number_of_edges()
                    
                    if num_nodes == 0:
                        continue

                    total_graphs += 1
                    
                    if num_edges >= 3 * num_nodes:
                        list_3x.append((file_path, num_nodes, num_edges))
                    
                    if num_edges >= 4 * num_nodes:
                        list_4x.append((file_path, num_nodes, num_edges))

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    print("-" * 30)
    print(f"Total graphs analyzed: {total_graphs}")
    
    print(f"\nGraphs with edges >= 3 * nodes ({len(list_3x)}):")
    for path, nodes, edges in list_3x:
        print(f"{path}: Nodes={nodes}, Edges={edges}")

    print(f"\nGraphs with edges >= 4 * nodes ({len(list_4x)}):")
    for path, nodes, edges in list_4x:
        print(f"{path}: Nodes={nodes}, Edges={edges}")

if __name__ == "__main__":
    target_dir = os.path.join("inputs", "all_outputs")
    if os.path.exists(target_dir):
        analyze_graphs(target_dir)
    else:
        print(f"Directory not found: {target_dir}")
