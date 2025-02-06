import fs from 'fs';
import * as d3 from "./d3.v7.min.js";
import  ForceEdgeBundling  from './d3-ForceEdgeBundling.js';



async function bundleFD(pathToBundle) { 
    /*
    Reads an graphml file found at the pathToBundle argument. 
    The graphml must contain two keys, specifying x and y positions for nodes. 
    Every node must have two <data> entries for their x and y positions. 
    Applies the force-directed bundling algorithm with default parameters from https://github.com/upphiminn/d3.ForceBundle
    Writes the output to a .edge file in the ../../outputs directory.
    */

    fs.readFile(pathToBundle,'utf8' ,(err, xml) => {
        if (err) {
            console.error("xml load fail", err);
            return;
        }

        //Regex magic. find the keys which map to x and y.
        const keyvalue_regex = /<key id="([^"]+)"(?:.*?attr\.name="([^"]+)")?/g;
        const matches = [...xml.matchAll(keyvalue_regex)];
        let keyLookup = {};
        matches.forEach(match => keyLookup[match[2]] = match[1]);

        // This finds all <node * </node> blocks
        const nodeRegex = /<node[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/node>/g;

        // This finds all <data * </data> blocks
        const dataRegex = /<data key="([^"]+)">([\d.e+-]+)<\/data>/g;

        //This will find all nodes in the xml and assign them their id, x, and y values based on the keys in the header.
        let nodes = [...xml.matchAll(nodeRegex)].map( ([_,id,content]) => {
            let data = Object.fromEntries([...content.matchAll(dataRegex)].map(([__, key, value]) => [key, parseFloat(value)]));
            return {'id': id, "x": data[keyLookup.x], "y": data[keyLookup.y]}
        });

        //the forcebundling expects an object instead of an array...
        let fb_nodes = Object.fromEntries(nodes.map(nodeobj => {
            return [nodeobj.id, {'x': nodeobj.x, 'y': nodeobj.y}]
        }));


        const edgeRegex = /<edge source="(\d+)" target="(\d+)"\/>/g;
        let edges = [...xml.matchAll(edgeRegex)].map(([_,source,target]) => (
            {
                'source': source,
                'target': target
            }
        ));


        var fbundling = ForceEdgeBundling().nodes(fb_nodes).edges(edges);
        var bundle = fbundling();

        //Has the following format for each edge: [source target x1 y1 x2 y2 ... xk yk]
        let polylines = bundle.map((edge, index) => {
            return [edges[index].source, edges[index].target, 
            ...edge.map((cpoint => [cpoint.x, cpoint.y])).flat()
            ]
        });
    
        fs.writeFile('outputs/edges.edge', 
            polylines.map(polyline => polyline.join(" ")).join("\n"), 
            (err) => {
                // console.log("finished")
                // In case of a error throw err.
                if (err) throw err;
            })

    });
} 

let pathToBundle = process.argv[2];
await bundleFD(pathToBundle);