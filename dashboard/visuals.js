const margin = 10;
const HISTO_COLOR = "#6497b1";
const romaO = ["#733957", "#743956", "#753954", "#753853", "#763851", "#773850", "#77384f", "#78384d", "#79384c", "#79384b", "#7a3849", "#7b3848", "#7c3847", "#7c3946", "#7d3945", "#7e3943", "#7e3942", "#7f3a41", "#803a40", "#813b3f", "#813b3e", "#823c3d", "#833c3c", "#843d3b", "#843d3a", "#853e39", "#863f38", "#874037", "#874037", "#884136", "#894235", "#8a4334", "#8b4433", "#8c4533", "#8c4632", "#8d4731", "#8e4831", "#8f4930", "#904a30", "#914c2f", "#924d2f", "#934e2e", "#94502e", "#94512d", "#95522d", "#96542d", "#97552c", "#98572c", "#99582c", "#9a5a2c", "#9b5b2c", "#9c5d2b", "#9d5f2b", "#9e602b", "#9f622b", "#a0642c", "#a2662c", "#a3672c", "#a4692c", "#a56b2d", "#a66d2d", "#a76f2d", "#a8712e", "#a9732e", "#aa752f", "#ab7730", "#ad7930", "#ae7b31", "#af7d32", "#b07f33", "#b18134", "#b28335", "#b48636", "#b58837", "#b68a39", "#b78c3a", "#b88e3b", "#b9913d", "#bb933f", "#bc9540", "#bd9842", "#be9a44", "#bf9c46", "#c19f47", "#c2a149", "#c3a34b", "#c4a54e", "#c5a850", "#c6aa52", "#c8ac54", "#c9af57", "#cab159", "#cbb35c", "#ccb55e", "#cdb761", "#ceba63", "#cfbc66", "#cfbe68", "#d0c06b", "#d1c26e", "#d2c470", "#d2c673", "#d3c876", "#d4c978", "#d4cb7b", "#d4cd7e", "#d5ce81", "#d5d083", "#d5d186", "#d6d388", "#d6d48b", "#d6d58e", "#d6d790", "#d6d893", "#d5d995", "#d5da98", "#d5db9a", "#d4dc9c", "#d4dd9f", "#d3dda1", "#d3dea3", "#d2dfa5", "#d1dfa7", "#d0e0a9", "#cfe0ab", "#cee0ad", "#cde1af", "#cce1b1", "#cbe1b3", "#cae1b5", "#c8e1b6", "#c7e1b8", "#c5e1b9", "#c4e1bb", "#c2e1bc", "#c1e1be", "#bfe1bf", "#bde0c0", "#bbe0c2", "#b9dfc3", "#b8dfc4", "#b6dec5", "#b4dec6", "#b1ddc7", "#afdcc8", "#addcc8", "#abdbc9", "#a9daca", "#a7d9cb", "#a4d8cb", "#a2d7cc", "#a0d6cc", "#9dd5cd", "#9bd4cd", "#99d3ce", "#96d1ce", "#94d0ce", "#92cfce", "#8fcecf", "#8dcccf", "#8bcbcf", "#88c9cf", "#86c8cf", "#84c6cf", "#81c5cf", "#7fc3cf", "#7dc2ce", "#7bc0ce", "#78bece", "#76bdce", "#74bbcd", "#72b9cd", "#70b8cd", "#6eb6cc", "#6cb4cc", "#6ab2cb", "#68b1cb", "#67afca", "#65adca", "#63abc9", "#62a9c9", "#60a8c8", "#5fa6c7", "#5da4c7", "#5ca2c6", "#5aa0c5", "#599ec4", "#589cc4", "#579bc3", "#5699c2", "#5597c1", "#5495c0", "#5393bf", "#5291be", "#528fbd", "#518dbc", "#508bbb", "#508aba", "#4f88b9", "#4f86b8", "#4f84b7", "#4f82b6", "#4e80b4", "#4e7eb3", "#4e7cb2", "#4e7ab0", "#4f78af", "#4f76ae", "#4f75ac", "#4f73ab", "#5071a9", "#506fa8", "#516da6", "#516ba4", "#5269a3", "#5267a1", "#53669f", "#54649e", "#54629c", "#55609a", "#565f98", "#575d96", "#575b94", "#585993", "#595891", "#5a568f", "#5b558d", "#5c538b", "#5c5289", "#5d5087", "#5e4f85", "#5f4d83", "#604c81", "#614b7f", "#62497d", "#63487b", "#634779", "#644677", "#654576", "#664474", "#674372", "#684270", "#68416e", "#69406c", "#6a3f6b", "#6b3f69", "#6c3e67", "#6c3d65", "#6d3d64", "#6e3c62", "#6f3b60", "#6f3b5f", "#703a5d", "#713a5c", "#723a5a", "#723959"];
const colormap = d3.interpolateRgbBasis(romaO);

class Bundling {
    constructor(svg) {
        this.svg = svg;
        this.draw_area = this.svg.append('g');
    }

    add_data(nodes, edges) {
        this.nodes = nodes;
        this.edges = edges;
        this.edges.forEach(edge => {
            var X = edge.X;
            var Y = edge.Y;

            var polyline = []
            for(var i = 0; i < X.length; i++) {
                polyline.push({'x': X[i], 'y': Y[i]})
            }

            edge.polyline = polyline;
        });
    }

    draw() {
        var tthis = this;
        function handleZoom(e) {
            tthis.svg.select('g').attr('transform', e.transform);
        }

        const zoom = d3.zoom().on('zoom', handleZoom);
        // this.svg.call(zoom);
        
        this.draw_area.selectAll('.edge')
            .data(this.edges)
            .join('path')
            .attr('class', 'edge')
            .attr('stroke', d => colormap(d['SL_angle']))
            .attr('opacity', 0.4)

        this.draw_area.selectAll('.node')
            .data(this.nodes)
            .join('circle')
            .attr('class', 'node')
            .attr('r', 1);


        this.resize();
    }

    resize() {
        var extentX = d3.extent(this.nodes, d => d.X);
        var extentY = d3.extent(this.nodes, d => d.Y);

        var b_width = extentX[1] - extentX[0]
        var b_height = extentY[1] - extentY[0] 

        // console.log(b_width, b_height)
        // console.log(this.svg.node().parentNode.parentNode)

        var width = this.svg.node().getBoundingClientRect().width;
        var height = width * b_height / b_width;

        // console.log(width, height)

        this.svg.attr('height', height);

        var xScale = d3.scaleLinear().range([margin, width - margin]).domain(extentX);
        var yScale = d3.scaleLinear().range([height - margin, margin]).domain(extentY);

        this.svg.selectAll('.node')
            .attr('cx', d => xScale(d.X))
            .attr('cy', d => yScale(d.Y))

        const line = d3.line().x(d => xScale(d.x)).y(d => yScale(d.y));
        this.svg.selectAll('.edge')
            .attr('d', d => line(d.polyline))       
    }

    async show_nodes(value) {
        // console.log(value)
        this.svg.selectAll('.node').style('opacity', value ? 1.0 : 0.0);
    }

}

class Histogram{
    constructor(svg, container, stats) {
        this.svg = svg;
        this.container = container;
        this.stats = stats;
    }

    async load_data(data, accessor) {
        var edges = data.edges;
        this.edges = data.edges;
        this.accessor = accessor;

        this.extentX = d3.extent(edges, d => d[accessor])
        
        var mean = d3.mean(edges, d => d[accessor]);
        var median = d3.median(edges, d => d[accessor]);

        var text = this.stats.append('div').attr('class', 'text-metric')
        text.append("div").text(`mean: ${mean.toFixed(3)}`)
        text.append("div").text(`median: ${median.toFixed(3)}`)

    }

    async draw () {
        this.svg.selectAll('*').remove();
        var width = this.svg.node().getBoundingClientRect().width;
        var height = this.svg.node().getBoundingClientRect().height;
        //this.svg.attr('height', 300);

        var xScale = d3.scaleLinear().domain(this.extentX).range([3 * margin, width - margin]);

        this.axisBottom = this.svg.append("g")
            .attr("transform", "translate(" + 0 * margin + "," + (height - 2 * margin) + ")")
            .call(d3.axisBottom(xScale).ticks(5));

        this.histogram = d3.histogram()
                            .value(d => {return d[this.accessor]})
                            .domain(xScale.domain())
                            .thresholds(xScale.ticks(50));

        this.bins = this.histogram(this.edges);
        this.extentY = [0, d3.max(this.bins, function(d) { return d.length; })];

        var yScale = d3.scaleLinear().range([height - 2 * margin, margin]).domain([0, d3.max(this.bins, function(d) { return d.length; })]); 
        this.svg.append("g")
            .attr("transform", "translate(" + 3 * margin + "," + 0 + ")")
            .call(d3.axisLeft(yScale).ticks(5));

        this.svg.selectAll("rect")
        .data(this.bins)
        .enter()
        .append("rect")
            .attr('x', 1)
            .attr('y', 0)
            .style("fill", HISTO_COLOR)

        this.resize();
    }



    async resize () {

        var width = this.svg.node().getBoundingClientRect().width;
        var height = this.svg.node().getBoundingClientRect().height;
        var xScale = d3.scaleLinear().domain(this.extentX).range([3 * margin, width - margin]);
        var yScale = d3.scaleLinear().range([height - 2 * margin, margin]).domain(this.extentY); 
        // console.log(xScale.domain(), xScale.range())

        this.svg.selectAll("rect")
            .attr("transform", function(d) { return "translate(" + xScale(d.x0) + "," + (yScale(d.length)) + ")"; })
            .attr("width", function(d) {return xScale(d.x1) - xScale(d.x0); })
            .attr("height", function(d) {return height - 2 * margin - yScale(d.length); })

        // this.svg.append("g")
        //     .attr("transform", "translate(" + 3 * margin + "," + margin + ")")
        //     .call(d3.axisLeft(yScale).ticks(5));
    }

    async resize_bins() {

    }

    get_extent() {
        return [d3.extent(this.edges, d => d[this.accessor]), this.extentY];
    }

    set_extent(extentX, extentY) {
        this.extentX = extentX;
        this.extentY = extentY;
    }
}

class TextElement{
    constructor(container, name) {
        this.container = container;
        this.name = name;       
    }

    async load_data(data, accessor) {
        this.data = data.graph;
        this.accessor = accessor;

 
        this.container.text(this.name + ": " + this.data[accessor].toFixed(3));
    }

    async draw () {
        
    }

    async resize () {
    }

    get_extent() {
        return [1000,0];
    }

    set_extent(extentX, extentY) {
    }
}

class Container{
    constructor(file, container, metrics, algorithm) {
        this.algorithm = algorithm;
        this.container = container;
        this.file = file;

        this.container.append('h6').text(algorithm);

        var bundlingSvg = this.container.append('svg').attr('class', 'bundlingSVG');
        this.bundling = new Bundling(bundlingSvg);
         
        this.metrics = {}

        metrics.forEach(metric => {

            if(metric.type === 'local') {
                var mCTop = this.container.append('div').attr('class', 'metric-container').attr('class', 'metric-histo')
                mCTop.append('h6').text(metric.name);
                var mCVis = mCTop.append('svg').attr('class', 'metricSVG');
                var mCStat = mCTop.append('div');

                this.metrics[metric.accessor] = new Histogram(mCVis, mCTop, mCStat)
            } else {
                var mCTop = this.container.append('div').attr('class', 'metric-container').attr('class', 'text-global');
                this.metrics[metric.accessor] = new TextElement(mCTop, metric.name);
            }

            mCTop.style('display', 'none');
            mCTop.style('visibility', 'collapse');
        });
    }

    async load_data() {
        var tthis = this;
        await d3.json(this.file).then(function(data) {
            tthis.data = data;

            tthis.bundling.add_data(data.nodes, data.edges);

            for (const [key, value] of Object.entries(tthis.metrics)) {
                value.load_data(data, key);
            };
        });
    }

    async resize() {
        this.bundling.resize();
    }

    async draw_network() {
        this.bundling.draw();
    }

    visibility(flag) {
        // console.log(flag ? 'none' : 'visible')
        this.container.style('visibility', flag ? 'visible' : 'collapse');
        this.container.style('display', flag ? 'block' : 'none');
    }

    metric_extent(metric) {
        return this.metrics[metric].get_extent();
    }

    metric_set_extent(metric, extent) {
        return this.metrics[metric].set_extent(extent);
    }

    metric_visibility(metric, flag) {
        var mC = this.metrics[metric]

        mC.container.style('display', flag ? 'block' : 'none');
        mC.container.style('visibility', flag ? 'visible' : 'collapse');

        mC.draw();
    }

    async show_nodes(value) {
        this.bundling.show_nodes(value);
    }

}

class Scatter{
/*
    Initiate with:      
    svg = d3.select("#visualization-container").append("svg")
                    .attr("id", "mainVis")
                    .style("border", "3px solid black");            
            vis = new EuclideanVis("#mainVis", data.nodes, null, null);
*/
    #nodeRadiusLarge = 15;
    #nodeRadiusSmall = 4;
    #colors = [ "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"];
    #margin = {top: 15, bottom: 15, left:15, right:15};    

    constructor(svgid,parallel){
        this.svg = d3.select(svgid);

        this.layer1 = this.svg.append("g");
        this.width = this.svg.node().getBoundingClientRect().width;
        this.height = this.svg.node().getBoundingClientRect().height;

        this.parallel = parallel;

        this.curBundleBox = null;
        this.selection    = null;
    }

    add_data(nodes){

        this.nodes = nodes;

        this.idMap = new Map();
        this.nodes.forEach((n,index) => {
            n.id = n.id.toString();
            this.idMap.set(n.id, index)
        });

        let xextent = d3.extent(this.nodes, d => d.tsnex);
        let yextent = d3.extent(this.nodes, d => d.tsney);

        
        let xscale = d3.scaleLinear().domain(xextent).range([this.#margin.left, this.width  - this.#margin.right] );
        let yscale = d3.scaleLinear().domain(yextent).range([this.height - this.#margin.bottom, this.#margin.top]);

        let unqAlgs = Array.from(new Set(nodes.map(d => d.alg))).sort();
        let cscale = d3.scaleOrdinal().domain(["sepb", "wr", "fd", "cubu", "epb" ]).range([ "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]);
        // let thing = "directionality_mag";
        // let cscale = d3.scaleLinear().domain(d3.extent(this.nodes, d => d[thing]))
        //     .range(["red", "blue"]);

        this.nodes.forEach(d => {
            d.x = xscale(d.tsnex);
            d.ox = xscale(d.tsnex);
            d.y = yscale(d.tsney);
            d.oy = yscale(d.tsney);
            d.class = cscale(d.alg);

            d.selected = false;
        });


    }

    draw(){
        this.layer1.selectAll(".nodes")
            .data(this.nodes, d => d.id)
            .join(
                enter => enter.append('circle')
                    .attr("class", 'nodes')
                    .attr("stroke", 'black')
                    .attr("fill", d => d.class)
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y)
                    .attr("r", this.#nodeRadiusSmall), 
                update => update, 
                exit => exit.remove()
            )
    }

    hover(){
        var tthis = this;
        this.layer1.selectAll(".nodes")
            .on("mouseenter", (e,d) => {
                document.getElementById("graph-title").innerHTML = d.graph;

                tthis.layer1.selectAll(".nodes")
                    .filter(u => u.graph === d.graph)
                    .classed("hover", true)

                if (! this.selection){
                    tthis.parallel.filterLines('id', d.id);   
                }
            })
            .on("mouseleave", () => tthis.layer1.selectAll(".nodes").classed("hover", false))
    }

    overlap(){
        const astr = 1;
        const repstr = 2;
        this.simulation = d3.forceSimulation(this.nodes)
            .force("charge", d3.forceCollide().radius(this.#nodeRadiusSmall+1).iterations(3))
            // .force("attract", d3.forceManyBody().strength(astr))
            .force("attract", d3.forceX(n => n.ox).strength(0.1))
            .force("attract2", d3.forceY(n => n.oy).strength(0.1));
            
        this.simulation.on("tick", () => {
            this.nodes.forEach(n => {
                let m = 10;
                n.x = n.x > this.width - m ? this.width - m : n.x;
                n.y = n.y > this.height - m ? this.height - m : n.y;
                
                n.x = n.x < m-5 ? m-5 : n.x; 
                n.y = n.y < m-5 ? m-5 : n.y;
            })
        
          this.layer1.selectAll(".nodes")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
        });

    }

    click(bundleid, parallel){
        let bundlediv = d3.select(bundleid);
        const ALGORITHMS = ["S-EPB", "WR", "FD", "EPB", "CUBu"].sort(); 
        const converter = {"S-EPB": 'sepb', "WR": 'wr', "FD": 'fd', "EPB": 'epb', "CUBu": 'cubu'}; 
        var bundlebox = [];

        this.layer1.selectAll(".nodes")
            .on("click", (e,d) => {
                this.nodes.forEach(v => v.selected = v.graph === this.selection);
                bundlebox = [];

                bundlediv.selectAll("*").remove();
                ALGORITHMS.forEach(al => {               
                    var container = bundlediv.append('div').attr('class', 'algorithm-container');
                    var file = `output_dashboard/${d.graph}/${converter[al]}.json`;
        
                    var obj = new Container(file, container, [], al);
                    obj.gname = d.graph;
                    bundlebox.push(obj);
                });
        
                bundlebox.forEach(obj => {
                    obj.load_data().then(() => obj.draw_network())
                    .catch(error => console.error("Error:", error));
                });       

                this.nodes.filter(v => v.graph === d.graph).forEach(v => v.selected = true);
                parallel.filterLines('graph', d.graph);   
                
                this.curBundleBox = bundlebox;
            });
    }

    saveCurrent(bundleid){
        let bundlediv = d3.select(bundleid);
        bundlediv.selectAll(".algorithm-container").remove();
        var bundlebox = this.curBundleBox.map(d => {
            let container = bundlediv.append('div').attr("class", "algorithm-container");
            let obj = new Container(d.file, container, [], d.algorithm);
            obj.gname = d.gname;
            return obj;
        });


        bundlebox.forEach(obj => {
            obj.load_data().then(() => obj.draw_network())
            .catch(error => console.error("Error:", error));
        });

        this.selection = bundlebox[0].gname;
    }

    clear(bundle1, bundle2, parallel){
        this.selection = null;
        this.nodes.forEach(v => v.selected = false);
        d3.select(bundle1).selectAll('*').remove();
        d3.select(bundle2).selectAll("*").remove();
        parallel.unfilter();
    }
}

class Parallel{
    constructor(svgid, METRICS,data){
        this.svg = d3.select(svgid);
        this.selected = null;

        this.unqAlgs = Array.from(new Set(data.map(d => d.alg))).sort();

        data.forEach(n => {
            n.name = n.alg;
        })

        this.data = data;

        this.metrics = METRICS.map(m => m.accessor);
        this.clickedAxes = [];        

        this.draw();

        
    }

    draw(){

        let data = this.data;

        var width = this.svg.node().getBoundingClientRect().width - 50;
        var height = this.svg.node().getBoundingClientRect().height - 50;

        this.svg = this.svg.append("g").attr("transform", "translate(25,25)");

        let svg = this.svg;
        let unqAlgs = this.unqAlgs;
        // this.svg = svg;

        var color = d3.scaleOrdinal().domain(["sepb", "wr", "fd", "cubu", "epb" ]).range([ "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"])

        var y = {};
        var metrics = [];

        for (let i in METRICS) {
          let name = METRICS[i].accessor;
          metrics.push(name);

          var extent = d3.extent(data, d => d[name]);


          y[name] = d3.scaleLinear()
            .domain( extent )
            .range([height, 0])
        }

        let x = d3.scalePoint()
        .range([0, width])
        .domain(metrics);

        this.scaleX = x;

        function path(d) {
            return d3.line()(metrics.map(function(p) { return [x(p), y[p](d[p])]; }));
        }

        this.lines = svg
        .selectAll("myPath")
        .data(data)
        .enter()
        .append("path")
          .attr("class", function (d) { return "line " + d.name } ) // 2 class for each line: 'line' and the group name
          .classed("myPath", true)
          .attr("d",  path)
          .style("fill", "none" )
          .style("stroke", function(d){ return( color(d.name))} )
          .style("opacity", 0.5)
          
        this.axes = svg.selectAll("myAxis")
          .data(METRICS).enter()
          .append("g")
          .attr("class", "axis")
          .attr("transform", function(d) {  return "translate(" + x(d.accessor) + ")"; })
          .each(function(d) { d3.select(this).call(d3.axisLeft().ticks(5).scale(y[d.accessor])); });

          this.axes.append("text")
            .style("text-anchor", "middle")
            .attr("y", -9)
            .text(function(d) { return d.name; })
            .style("fill", "black")

        var group = svg.selectAll("legend")
            .data(unqAlgs).enter()
            .append("g")
            .attr("transform", (d,i) => "translate(" + i * 100 + "," + (height + 15) + ")")
            
        group.append("line")
            .attr("x1", 0)
            .attr("y1", 0)
            .attr("x2", 10)
            .attr("y2", 0)
            .style("fill", "none" )
            .style("stroke", function(d){ return( color(d))} )
            .style("stroke-width", 5)

        group.append("text")
            .text(d => d)
            .attr("transform", "translate(" + 15 + "," + 5 + ")")
            .attr("font-size", 14)        

        this.y = y;
        this.metrics = metrics;            
    }

    click() {
        this.axes.on("click", (event, d) => {
            const clicked = d.accessor;
    
            if (!this.clickedAxes.includes(clicked)) {
                this.clickedAxes.push(clicked);
            }
    
            if (this.clickedAxes.length === 2) {
                const [a, b] = this.clickedAxes;
    
                // Swap the two axis positions in the metrics array
                const idxA = this.metrics.indexOf(a);
                const idxB = this.metrics.indexOf(b);
                [this.metrics[idxA], this.metrics[idxB]] = [this.metrics[idxB], this.metrics[idxA]];

    
                this.updateAxesAndLines();
    
                // Reset clicked axes
                this.clickedAxes = [];
            }
        });
    }

    updateAxesAndLines() {
        // const x = d3.scalePoint()
        //     .range([0, this.svg.node().getBoundingClientRect().width - 50])
        //     .domain(this.metrics);
        this.scaleX.domain(this.metrics);
    
        // const path = d3.line();
        const path = d => d3.line()(this.metrics.map(p => [this.scaleX(p), this.y[p](d[p])]));
        
        // Update axis positions
        var tthis = this;
        this.svg.selectAll(".axis")
            .transition()
            .duration(750)
            .attr("transform", d => "translate(" + this.scaleX(d.accessor) + ")")
            
        // this.svg.selectAll(".axis")    
        // // .each(function(d) { d3.select(this).call(d3.axisLeft().ticks(5).scale(tthis.y[d.accessor])); })
        //     .append("text")
        //       .style("text-anchor", "middle")
        //       .attr("y", -9)
        //       .text(function(d) { return d.name; })
        //       .style("fill", "black")            
    
        // Update lines
        this.lines
            .transition()
            .duration(750)
            .attr("d", path);
    }    

    filterLines(prop, val){
        this.lines
            .style("opacity", l => (l[prop] === val) || (l.selected) ? 1.0 : 0.01)
            .style("stroke-width", 3)
            // .remove();
        
        this.lines.filter(l => l[prop] === val).raise();
    }

    unfilter(){
        this.lines
            .style('opacity', 0.5)
            .style('stroke-width', 1)
    }
}