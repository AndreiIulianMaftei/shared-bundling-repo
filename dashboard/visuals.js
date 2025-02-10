var margin = 5;

class Bundling {
    constructor(svg) {
        this.svg = svg;
    }

    add_data(nodes, edges) {
        this.nodes = nodes;
        this.edges = edges;
    }

    draw() {
        console.log(this.nodes);
        this.svg.selectAll('.node')
            .data(this.nodes)
            .join('circle')
            .attr('class', 'node')
            .attr('r', 3);

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

        console.log(width, height)

        var xScale = d3.scaleLinear().range([margin, width - margin]).domain(extentX);
        var yScale = d3.scaleLinear().range([height - margin, margin]).domain(extentY);

        console.log(extentX, extentY);
        this.svg.selectAll('.node')
            .attr('cx', d => xScale(d.X))
            .attr('cy', d => yScale(d.Y))
    }
}

class Histogram{
    constructor(svg_id, edges, accessor) {

    }

    draw () {

    }

    resize () {

    }
}

class Container{
    constructor(file, container, metrics) {
        this.container = container;
        this.file = file;

        var bundlingSvg = this.container.append('svg').attr('class', 'bundlingSVG');
        this.bundling = new Bundling(bundlingSvg);
        
        this.metrics = []

        metrics.forEach(metric => {
            var mC = this.container.append('div')
            mC.append('svg').attr('class', 'metricSVG')
        });
    }

    async load_data() {
        var tthis = this;
        await d3.json(this.file).then(function(data) {
            tthis.data = data;

            tthis.bundling.add_data(data.nodes, data.edges);
        });
    }

    async draw_network() {
        this.bundling.draw();
    }

}