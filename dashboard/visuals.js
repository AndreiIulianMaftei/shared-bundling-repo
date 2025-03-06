const margin = 10;
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
        console.log(this.edges);
        this.edges.forEach(edge => {
            var X = edge.X;
            var Y = edge.Y;

            //var angle = 

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
        this.svg.call(zoom);
        
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

        var width = this.svg.node().getBoundingClientRect().width;
        var height = width * b_height / b_width;

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
        console.log(value)
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

        this.stats.append('span').text(`mean: ${mean.toFixed(2)}   median: ${median.toFixed(2)}`)

    }

    async draw () {
        this.svg.selectAll('*').remove();
        var width = this.svg.node().getBoundingClientRect().width;
        this.svg.attr('height', width);

        var xScale = d3.scaleLinear().domain(this.extentX).range([3 * margin, width - margin]);

        this.axisBottom = this.svg.append("g")
            .attr("transform", "translate(" + 0 * margin + "," + (width - 2 * margin) + ")")
            .call(d3.axisBottom(xScale).ticks(5));

        this.histogram = d3.histogram()
                            .value(d => {return d[this.accessor]})
                            .domain(xScale.domain())
                            .thresholds(xScale.ticks(50));

        this.bins = this.histogram(this.edges);
        this.extentY = [0, d3.max(this.bins, function(d) { return d.length; })];

        var yScale = d3.scaleLinear().range([width - 2 * margin, margin]).domain([0, d3.max(this.bins, function(d) { return d.length; })]); 
        this.svg.append("g")
            .attr("transform", "translate(" + 3 * margin + "," + 0 + ")")
            .call(d3.axisLeft(yScale).ticks(5));

        this.svg.selectAll("rect")
        .data(this.bins)
        .enter()
        .append("rect")
            .attr('x', 1)
            .attr('y', 0)
            .style("fill", "#69b3a2")

        this.resize();
    }



    async resize () {

        var width = this.svg.node().getBoundingClientRect().width;
        var xScale = d3.scaleLinear().domain(this.extentX).range([3 * margin, width - margin]);
        var yScale = d3.scaleLinear().range([width - 2 * margin, margin]).domain(this.extentY); 
        console.log(xScale.domain(), xScale.range())

        this.svg.selectAll("rect")
            .attr("transform", function(d) { return "translate(" + xScale(d.x0) + "," + (yScale(d.length)) + ")"; })
            .attr("width", function(d) {return xScale(d.x1) - xScale(d.x0); })
            .attr("height", function(d) {return width - 2 * margin - yScale(d.length); })

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

 
        this.container.append("span").text(this.name + ": " + this.data[accessor].toFixed(3));
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
                var mCTop = this.container.append('div').attr('class', 'metric-container')
                mCTop.append('h6').text(metric.name);
                var mCVis = mCTop.append('svg').attr('class', 'metricSVG');
                var mCStat = mCTop.append('div');

                this.metrics[metric.accessor] = new Histogram(mCVis, mCTop, mCStat)
            } else {
                var mCTop = this.container.append('div').attr('class', 'metric-container');
                this.metrics[metric.accessor] = new TextElement(mCTop, metric.name);
            }

            mCTop.style('diplay', 'none');
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
        console.log(flag ? 'none' : 'visible')
        this.container.style('visibility', flag ? 'visible' : 'collapse');
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