import fs from 'fs';
// import * as d3 from "./d3.v7.min.js";
import  ForceEdgeBundling  from './d3-ForceEdgeBundling.js';



async function bundleFD(pathToBundle) { 
    /*
    Reads an graphml file found at the pathToBundle argument. 
    The graphml must contain two keys, specifying x and y positions for nodes. 
    Every node must have two <data> entries for their x and y positions. 
    Applies the force-directed bundling algorithm with default parameters from https://github.com/upphiminn/d3.ForceBundle
    Writes the output to a .edge file in the ../../outputs directory.
    */
    var json_obj = JSON.parse(fs.readFileSync(pathToBundle, 'utf8'));
    console.log(json_obj);

    var edges = json_obj.edges;
    var nodes = json_obj.nodes;

    var fbundling = ForceEdgeBundling().nodes(nodes).edges(edges);
    var bundle = fbundling();

    //Has the following format for each edge: [source target x1 y1 x2 y2 ... xk yk]
    let polylines = bundle.map((edge, index) => {
        return [edges[index].source, edges[index].target, 
        ...edge.map((cpoint => [cpoint.x, cpoint.y])).flat()
        ]
    });

    console.log(polylines);
    fs.writeFileSync('outputs/edges.edge', polylines.map(polyline => polyline.join(" ")).join("\n"));

} 

let pathToBundle = process.argv[2];
await bundleFD(pathToBundle);