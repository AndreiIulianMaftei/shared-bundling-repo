var margin = 10;

class Bundling {
    constructor(svg) {
        this.svg = svg;
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
        console.log(this.nodes);
        this.svg.selectAll('.node')
            .data(this.nodes)
            .join('circle')
            .attr('class', 'node')
            .attr('r', 3);

        this.svg.selectAll('.edge')
            .data(this.edges)
            .join('path')
            .attr('class', 'edge')

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
            .attr('stroke', d => 'black')
            .attr('opacity', 0.4)
        
    }
}

class Histogram{
    constructor(svg, container) {
        this.svg = svg;
        this.container = container;
    }

    async load_data(edges, accessor) {
        this.edges = edges;
        this.accessor = accessor;

        var width = this.svg.node().getBoundingClientRect().width;
        
        this.svg.attr('height', width);

        this.extentX = d3.extent(edges, d => d[accessor])
        
        var xScale = d3.scaleLinear().domain(this.extentX).range([3 * margin, width - margin]);

        this.axisBottom = this.svg.append("g")
            .attr("transform", "translate(" + 0 * margin + "," + (width - 2 * margin) + ")")
            .call(d3.axisBottom(xScale).ticks(5));

        this.histogram = d3.histogram()
                            .value(d => {return d[accessor]})
                            .domain(xScale.domain())
                            .thresholds(xScale.ticks(20));

        this.bins = this.histogram(edges);

        var yScale = d3.scaleLinear().range([width - 2 * margin, margin]).domain([0, d3.max(this.bins, function(d) { return d.length; })]); 
        this.svg.append("g")
            .attr("transform", "translate(" + 3 * margin + "," + 0 + ")")
            .call(d3.axisLeft(yScale).ticks(5));
    }

    async draw () {
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
        console.log(this.bins);
        var width = this.svg.node().getBoundingClientRect().width;
        var xScale = d3.scaleLinear().domain(this.extentX).range([3 * margin, width - margin]);
        var yScale = d3.scaleLinear().range([width - 2 * margin, margin]).domain([0, d3.max(this.bins, function(d) { return d.length; })]); 
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
            var mCTop = this.container.append('div')
            mCTop.append('h6').text(metric);
            var mCVis = mCTop.append('svg').attr('class', 'metricSVG')
        
            mCTop.style('visibility', 'collapse');

            this.metrics[metric] = new Histogram(mCVis, mCTop)
        });
    }

    async load_data() {
        var tthis = this;
        await d3.json(this.file).then(function(data) {
            tthis.data = data;

            tthis.bundling.add_data(data.nodes, data.edges);

            for (const [key, value] of Object.entries(tthis.metrics)) {
                value.load_data(data.edges, 'distortion');
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

    metric_visibility(metric, flag) {
        var mC = this.metrics[metric]

        mC.container.style('visibility', flag ? 'visible' : 'collapse');

        mC.draw();
    }

}