import { DOMParser } from 'xmldom';
import fs from 'fs';
import CustomD3 from './d3.js';

function someFunction() { 

    //const fs = require('fs')

    //const { DOMParser } = require('xmldom');

    //const d3 = require('d3');

    fs.readFile('./airports.xml','utf8' ,(err, xml) => {
        if (err) {
            console.error("xml load fail", err);
            return;
        }
        /// add the d3 code here
        (function () {
            CustomD3.ForceEdgeBundling = function () {
                var data_nodes = {}, // {'nodeid':{'x':,'y':},..}
                    data_edges = [], // [{'source':'nodeid1', 'target':'nodeid2'},..]
                    compatibility_list_for_edge = [],
                    subdivision_points_for_edge = [],
                    K = 0.1, // global bundling constant controlling edge stiffness
                    S_initial = 0.1, // init. distance to move points
                    P_initial = 1, // init. subdivision number
                    P_rate = 2, // subdivision rate increase
                    C = 6, // number of cycles to perform
                    I_initial = 90, // init. number of iterations for cycle
                    I_rate = 0.6666667, // rate at which iteration number decreases i.e. 2/3
                    compatibility_threshold = 0.6,
                    eps = 1e-6;
        
        
                /*** Geometry Helper Methods ***/
                function vector_dot_product(p, q) {
                    return p.x * q.x + p.y * q.y;
                }
        
                function edge_as_vector(P) {
                    return {
                        'x': data_nodes[P.target].x - data_nodes[P.source].x,
                        'y': data_nodes[P.target].y - data_nodes[P.source].y
                    }
                }
        
                function edge_length(e) {
                    // handling nodes that are on the same location, so that K/edge_length != Inf
                    if (Math.abs(data_nodes[e.source].x - data_nodes[e.target].x) < eps &&
                        Math.abs(data_nodes[e.source].y - data_nodes[e.target].y) < eps) {
                        return eps;
                    }
        
                    return Math.sqrt(Math.pow(data_nodes[e.source].x - data_nodes[e.target].x, 2) +
                        Math.pow(data_nodes[e.source].y - data_nodes[e.target].y, 2));
                }
        
                function custom_edge_length(e) {
                    return Math.sqrt(Math.pow(e.source.x - e.target.x, 2) + Math.pow(e.source.y - e.target.y, 2));
                }
        
                function edge_midpoint(e) {
                    var middle_x = (data_nodes[e.source].x + data_nodes[e.target].x) / 2.0;
                    var middle_y = (data_nodes[e.source].y + data_nodes[e.target].y) / 2.0;
        
                    return {
                        'x': middle_x,
                        'y': middle_y
                    };
                }
        
                function compute_divided_edge_length(e_idx) {
                    var length = 0;
        
                    for (var i = 1; i < subdivision_points_for_edge[e_idx].length; i++) {
                        var segment_length = euclidean_distance(subdivision_points_for_edge[e_idx][i], subdivision_points_for_edge[e_idx][i - 1]);
                        length += segment_length;
                    }
        
                    return length;
                }
        
                function euclidean_distance(p, q) {
                    return Math.sqrt(Math.pow(p.x - q.x, 2) + Math.pow(p.y - q.y, 2));
                }
        
                function project_point_on_line(p, Q) {
                    var L = Math.sqrt((Q.target.x - Q.source.x) * (Q.target.x - Q.source.x) + (Q.target.y - Q.source.y) * (Q.target.y - Q.source.y));
                    var r = ((Q.source.y - p.y) * (Q.source.y - Q.target.y) - (Q.source.x - p.x) * (Q.target.x - Q.source.x)) / (L * L);
        
                    return {
                        'x': (Q.source.x + r * (Q.target.x - Q.source.x)),
                        'y': (Q.source.y + r * (Q.target.y - Q.source.y))
                    };
                }
        
                /*** ********************** ***/
        
                /*** Initialization Methods ***/
                function initialize_edge_subdivisions() {
                    for (var i = 0; i < data_edges.length; i++) {
                        if (P_initial === 1) {
                            subdivision_points_for_edge[i] = []; //0 subdivisions
                        } else {
                            subdivision_points_for_edge[i] = [];
                            subdivision_points_for_edge[i].push(data_nodes[data_edges[i].source]);
                            subdivision_points_for_edge[i].push(data_nodes[data_edges[i].target]);
                        }
                    }
                }
        
                function initialize_compatibility_lists() {
                    for (var i = 0; i < data_edges.length; i++) {
                        compatibility_list_for_edge[i] = []; //0 compatible edges.
                    }
                }
        
                function filter_self_loops(edgelist) {
                    var filtered_edge_list = [];
        
                    for (var e = 0; e < edgelist.length; e++) {
                        if (data_nodes[edgelist[e].source].x != data_nodes[edgelist[e].target].x ||
                            data_nodes[edgelist[e].source].y != data_nodes[edgelist[e].target].y) { //or smaller than eps
                            filtered_edge_list.push(edgelist[e]);
                        }
                    }
        
                    return filtered_edge_list;
                }
        
                /*** ********************** ***/
        
                /*** Force Calculation Methods ***/
                function apply_spring_force(e_idx, i, kP) {
                    var prev = subdivision_points_for_edge[e_idx][i - 1];
                    var next = subdivision_points_for_edge[e_idx][i + 1];
                    var crnt = subdivision_points_for_edge[e_idx][i];
                    var x = prev.x - crnt.x + next.x - crnt.x;
                    var y = prev.y - crnt.y + next.y - crnt.y;
        
                    x *= kP;
                    y *= kP;
        
                    return {
                        'x': x,
                        'y': y
                    };
                }
        
                function apply_electrostatic_force(e_idx, i) {
                    var sum_of_forces = {
                        'x': 0,
                        'y': 0
                    };
                    var compatible_edges_list = compatibility_list_for_edge[e_idx];
        
                    for (var oe = 0; oe < compatible_edges_list.length; oe++) {
                        var force = {
                            'x': subdivision_points_for_edge[compatible_edges_list[oe]][i].x - subdivision_points_for_edge[e_idx][i].x,
                            'y': subdivision_points_for_edge[compatible_edges_list[oe]][i].y - subdivision_points_for_edge[e_idx][i].y
                        };
        
                        if ((Math.abs(force.x) > eps) || (Math.abs(force.y) > eps)) {
                            var diff = (1 / Math.pow(custom_edge_length({
                                'source': subdivision_points_for_edge[compatible_edges_list[oe]][i],
                                'target': subdivision_points_for_edge[e_idx][i]
                            }), 1));
        
                            sum_of_forces.x += force.x * diff;
                            sum_of_forces.y += force.y * diff;
                        }
                    }
        
                    return sum_of_forces;
                }
        
        
                function apply_resulting_forces_on_subdivision_points(e_idx, P, S) {
                    var kP = K / (edge_length(data_edges[e_idx]) * (P + 1)); // kP=K/|P|(number of segments), where |P| is the initial length of edge P.
                    // (length * (num of sub division pts - 1))
                    var resulting_forces_for_subdivision_points = [{
                        'x': 0,
                        'y': 0
                    }];
                
                    for (var i = 1; i < P + 1; i++) { // exclude initial end points of the edge 0 and P+1
                        var resulting_force = {
                            'x': 0,
                            'y': 0
                        };
                
                        let spring_force = apply_spring_force(e_idx, i, kP);
                        let electrostatic_force = apply_electrostatic_force(e_idx, i);
                
                        resulting_force.x = S * (spring_force.x + electrostatic_force.x);
                        resulting_force.y = S * (spring_force.y + electrostatic_force.y);
                
                        resulting_forces_for_subdivision_points.push(resulting_force);
                    }
                
                    resulting_forces_for_subdivision_points.push({
                        'x': 0,
                        'y': 0
                    });
                
                    return resulting_forces_for_subdivision_points;
                }
                /*** ********************** ***/
        
                /*** Edge Division Calculation Methods ***/
                function update_edge_divisions(P) {
                    for (var e_idx = 0; e_idx < data_edges.length; e_idx++) {
                        if (P === 1) {
                            subdivision_points_for_edge[e_idx].push(data_nodes[data_edges[e_idx].source]); // source
                            subdivision_points_for_edge[e_idx].push(edge_midpoint(data_edges[e_idx])); // mid point
                            subdivision_points_for_edge[e_idx].push(data_nodes[data_edges[e_idx].target]); // target
                        } else {
                            var divided_edge_length = compute_divided_edge_length(e_idx);
                            var segment_length = divided_edge_length / (P + 1);
                            var current_segment_length = segment_length;
                            var new_subdivision_points = [];
                            new_subdivision_points.push(data_nodes[data_edges[e_idx].source]); //source
        
                            for (var i = 1; i < subdivision_points_for_edge[e_idx].length; i++) {
                                var old_segment_length = euclidean_distance(subdivision_points_for_edge[e_idx][i], subdivision_points_for_edge[e_idx][i - 1]);
        
                                while (old_segment_length > current_segment_length) {
                                    var percent_position = current_segment_length / old_segment_length;
                                    var new_subdivision_point_x = subdivision_points_for_edge[e_idx][i - 1].x;
                                    var new_subdivision_point_y = subdivision_points_for_edge[e_idx][i - 1].y;
        
                                    new_subdivision_point_x += percent_position * (subdivision_points_for_edge[e_idx][i].x - subdivision_points_for_edge[e_idx][i - 1].x);
                                    new_subdivision_point_y += percent_position * (subdivision_points_for_edge[e_idx][i].y - subdivision_points_for_edge[e_idx][i - 1].y);
                                    new_subdivision_points.push({
                                        'x': new_subdivision_point_x,
                                        'y': new_subdivision_point_y
                                    });
        
                                    old_segment_length -= current_segment_length;
                                    current_segment_length = segment_length;
                                }
                                current_segment_length -= old_segment_length;
                            }
                            new_subdivision_points.push(data_nodes[data_edges[e_idx].target]); //target
                            subdivision_points_for_edge[e_idx] = new_subdivision_points;
                        }
                    }
                }
        
                /*** ********************** ***/
        
                /*** Edge compatibility measures ***/
                function angle_compatibility(P, Q) {
                    return Math.abs(vector_dot_product(edge_as_vector(P), edge_as_vector(Q)) / (edge_length(P) * edge_length(Q)));
                }
        
                function scale_compatibility(P, Q) {
                    var lavg = (edge_length(P) + edge_length(Q)) / 2.0;
                    return 2.0 / (lavg / Math.min(edge_length(P), edge_length(Q)) + Math.max(edge_length(P), edge_length(Q)) / lavg);
                }
        
                function position_compatibility(P, Q) {
                    var lavg = (edge_length(P) + edge_length(Q)) / 2.0;
                    var midP = {
                        'x': (data_nodes[P.source].x + data_nodes[P.target].x) / 2.0,
                        'y': (data_nodes[P.source].y + data_nodes[P.target].y) / 2.0
                    };
                    var midQ = {
                        'x': (data_nodes[Q.source].x + data_nodes[Q.target].x) / 2.0,
                        'y': (data_nodes[Q.source].y + data_nodes[Q.target].y) / 2.0
                    };
        
                    return lavg / (lavg + euclidean_distance(midP, midQ));
                }
        
                function edge_visibility(P, Q) {
                    var I0 = project_point_on_line(data_nodes[Q.source], {
                        'source': data_nodes[P.source],
                        'target': data_nodes[P.target]
                    });
                    var I1 = project_point_on_line(data_nodes[Q.target], {
                        'source': data_nodes[P.source],
                        'target': data_nodes[P.target]
                    }); //send actual edge points positions
                    var midI = {
                        'x': (I0.x + I1.x) / 2.0,
                        'y': (I0.y + I1.y) / 2.0
                    };
                    var midP = {
                        'x': (data_nodes[P.source].x + data_nodes[P.target].x) / 2.0,
                        'y': (data_nodes[P.source].y + data_nodes[P.target].y) / 2.0
                    };
        
                    return Math.max(0, 1 - 2 * euclidean_distance(midP, midI) / euclidean_distance(I0, I1));
                }
        
                function visibility_compatibility(P, Q) {
                    return Math.min(edge_visibility(P, Q), edge_visibility(Q, P));
                }
        
                function compatibility_score(P, Q) {
                    return (angle_compatibility(P, Q) * scale_compatibility(P, Q) * position_compatibility(P, Q) * visibility_compatibility(P, Q));
                }
        
                function are_compatible(P, Q) {
                    return (compatibility_score(P, Q) >= compatibility_threshold);
                }
        
                function compute_compatibility_lists() {
                    for (var e = 0; e < data_edges.length - 1; e++) {
                        for (var oe = e + 1; oe < data_edges.length; oe++) { // don't want any duplicates
                            if (are_compatible(data_edges[e], data_edges[oe])) {
                                compatibility_list_for_edge[e].push(oe);
                                compatibility_list_for_edge[oe].push(e);
                            }
                        }
                    }
                }
        
                /*** ************************ ***/
        
                /*** Main Bundling Loop Methods ***/
                var forcebundle = function () {
                    var S = S_initial;
                    var I = I_initial;
                    var P = P_initial;
        
                    initialize_edge_subdivisions();
                    initialize_compatibility_lists();
                    update_edge_divisions(P);
                    compute_compatibility_lists();
        
                    for (var cycle = 0; cycle < C; cycle++) {
                        for (var iteration = 0; iteration < I; iteration++) {
                            var forces = [];
                            for (var edge = 0; edge < data_edges.length; edge++) {
                                forces[edge] = apply_resulting_forces_on_subdivision_points(edge, P, S);
                            }
                            for (var e = 0; e < data_edges.length; e++) {
                                for (var i = 0; i < P + 1; i++) {
                                    subdivision_points_for_edge[e][i].x += forces[e][i].x;
                                    subdivision_points_for_edge[e][i].y += forces[e][i].y;
                                }
                            }
                        }
                        // prepare for next cycle
                        S = S / 2;
                        P = P * P_rate;
                        I = I_rate * I;
        
                        update_edge_divisions(P);
                        //console.log('C' + cycle);
                        //console.log('P' + P);
                        //console.log('S' + S);
                    }
                    return subdivision_points_for_edge;
                };
                /*** ************************ ***/
        
        
                /*** Getters/Setters Methods ***/
                forcebundle.nodes = function (nl) {
                    if (arguments.length === 0) {
                        return data_nodes;
                    } else {
                        data_nodes = nl;
                    }
        
                    return forcebundle;
                };
        
                forcebundle.edges = function (ll) {
                    if (arguments.length === 0) {
                        return data_edges;
                    } else {
                        data_edges = filter_self_loops(ll); //remove edges to from to the same point
                    }
        
                    return forcebundle;
                };
        
                forcebundle.bundling_stiffness = function (k) {
                    if (arguments.length === 0) {
                        return K;
                    } else {
                        K = k;
                    }
        
                    return forcebundle;
                };
        
                forcebundle.step_size = function (step) {
                    if (arguments.length === 0) {
                        return S_initial;
                    } else {
                        S_initial = step;
                    }
        
                    return forcebundle;
                };
        
                forcebundle.cycles = function (c) {
                    if (arguments.length === 0) {
                        return C;
                    } else {
                        C = c;
                    }
        
                    return forcebundle;
                };
        
                forcebundle.iterations = function (i) {
                    if (arguments.length === 0) {
                        return I_initial;
                    } else {
                        I_initial = i;
                    }
        
                    return forcebundle;
                };
        
                forcebundle.iterations_rate = function (i) {
                    if (arguments.length === 0) {
                        return I_rate;
                    } else {
                        I_rate = i;
                    }
        
                    return forcebundle;
                };
        
                forcebundle.subdivision_points_seed = function (p) {
                    if (arguments.length == 0) {
                        return P;
                    } else {
                        P = p;
                    }
        
                    return forcebundle;
                };
        
                forcebundle.subdivision_rate = function (r) {
                    if (arguments.length === 0) {
                        return P_rate;
                    } else {
                        P_rate = r;
                    }
        
                    return forcebundle;
                };
        
                forcebundle.compatibility_threshold = function (t) {
                    if (arguments.length === 0) {
                        return compatibility_threshold;
                    } else {
                        compatibility_threshold = t;
                    }
        
                    return forcebundle;
                };
        
                /*** ************************ ***/
        
                return forcebundle;
            }
        })();
        /// add the extraction code here
        fs.writeFile('Edge.xml', xml, (err) => {

            if(err) console.error("write fail", err);
        });
        const doc = new DOMParser().parseFromString(xml, 'application/xml');

        var raw_edges = doc.documentElement.getElementsByTagName("edge");
        var raw_nodes = doc.documentElement.getElementsByTagName("node");
        
        var eedges = [];
        var nnodes = {};
        var min_x = Number.MAX_VALUE;
        var max_x = 0;
        var min_y = Number.MAX_VALUE;
        var max_y = 0;

        for (var i = 0; i < raw_nodes.length; i++) {
            var key = raw_nodes[i].getAttribute('id');
            //modified hee
            var x = Math.abs(parseFloat(raw_nodes[i].childNodes[1]?.firstChild?.nodeValue || 0));
            var name = raw_nodes[i].childNodes[3]?.firstChild?.nodeValue || '';
            var y = Math.abs(parseFloat(raw_nodes[i].childNodes[5]?.firstChild?.nodeValue || 0));

            nnodes[key] = {
                'x': x,
                'y': y
            };
            min_x = Math.min(min_x, x);
            max_x = Math.max(max_x, x);
            min_y = Math.min(min_y, y);
            max_y = Math.max(max_y, y);
        }

        for (var i = 0; i < raw_edges.length; i++) {
            eedges.push({
                'source': raw_edges[i].getAttribute('source'),
                'target': raw_edges[i].getAttribute('target')
            });
        }
        var fbundling = CustomD3.ForceEdgeBundling().nodes(nnodes).edges(eedges);
        var results = fbundling();

        var s = "";
        for (var i = 0; i < raw_nodes.length; i++) {
            s = s + i + ' ' + nnodes[i].x + ' ' + nnodes[i].y + '\n';
        }
        //console.log("nodes");
        //console.log(s);
        fs.writeFile('nodes.node', s, (err) => {
    
            // In case of a error throw err.
            if (err) throw err;
        })
        s = "";
        for (var i = 0; i < results.length; i++) {
            s += eedges[i].source + ' ' + eedges [i].target;

            for (var j = 0; j < results[i].length; j++) {
                s += ' ' + results[i][j].x + ' ' + results[i][j].y;
            }

            s += '\n';
        }
        fs.writeFile('edges.edge', s, (err) => {
    
            // In case of a error throw err.
            if (err) throw err;
        })

    });
    // Data which will write in a file.
    let data = "Learning how to write in a file 2  "
    
    // Write data in 'Output.txt' .
    fs.writeFile('Output.txt', data, (err) => {
    
        // In case of a error throw err.
        if (err) throw err;
    })
    console.log("success");
} 
export default someFunction;
someFunction();