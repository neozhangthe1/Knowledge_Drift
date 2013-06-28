<!-- <html><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"><meta charset="utf-8">
<title>Sankey Diagram</title> -->
<link rel="stylesheet" type="text/css" href="/static/css/topictrend.css">

<div class="navbar-form pull-left" style="padding-bottom:20px">
  <input type="text" class="span2" id="topic-trend-search-text">
  <!-- <button type="submit" class="btn" id="topic-trend-search">Submit</button> -->
</div>

<div id="chart" class="pull-left">
</div>
<script src="/static/d3.v3.min.js"></script>
<script src="/static/js/sankey.js"></script>
<script>

var margin = {top: 1, right: 1, bottom: 6, left: 1},
    width = 1280- margin.left - margin.right,
    height = 300 - margin.top - margin.bottom;

var formatNumber = d3.format(",.0f"),
    format = function(d) { return formatNumber(d) + " TWh"; },
    color = d3.scale.category20();

var svg = d3.select("#chart").append("svg")
    .attr("width", 1600)//width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)


var sankey = d3.sankey()
    .nodeWidth(1)
    .nodePadding(10)
    .size([width, height]);

var path = sankey.link();

var area = d3.svg.area()
          .x(function(d){
            return d.x;
          })
          .y0(function(d){
            return d.y0;
          })
          .y1(function(d){
            return d.y1;
          });

var y = d3.scale.linear()
    .range([height, 0]);

d3.select("#topic-trend-search").on("click",function(e){
  render_topic($("#topic-trend-search-text").val(), 0.0001);
})


function resize_chart(){
  d3.select("#chart").style("width", (window.width - 2 * 50)+"px");
}

resize_chart();
window.onresize = resize_chart();
render_topic("interaction design", 0.0001);
document.getElementById("topic-trend-search-text").value ="interaction design"

function render_topic(q, threshold){
  svg.remove();
  svg = d3.select("#chart").append("svg")
    .attr("width", 1600)//width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .attr("id","trend")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  d3.json("/academic/render?q="+q+"&threshold="+threshold, function(energy) {

  svg.append("linearGradient")
    .attr("id", "temperature-gradient")
    .attr("gradientUnits", "userSpaceOnUse")
    .attr("x1", 0).attr("y1", y(5))
    .attr("x2", 0).attr("y2", y(10))
  .selectAll("stop")
    .data([
      {offset: "0%", color: "steelblue"},
      {offset: "50%", color: "gray"},
      {offset: "100%", color: "red"}
    ])
  .enter().append("stop")
    .attr("offset", function(d) { return d.offset; })
    .attr("stop-color", function(d) { return d.color; });

  var x = d3.scale.linear()
        .range([2002,2013])

  sankey
      .nodes(energy.nodes)
      .links(energy.links)
      .layout(32);

  var link = svg.append("g").selectAll(".link")
      .data(energy.links)
      .enter().append("path")
      .attr("class", "link")
      .attr("d", path)
      .style("stroke-width", function(d) { return 20 })
      .style("fill", function(d) { 
        var key = "gradient-"+d.source_index+"-"+d.target_index;
        svg.append("linearGradient")
        .attr("id", key)
        .attr("gradientUnits", "userSpaceOnUse")
        .attr("x1", d.source.x).attr("y1", 0)
        .attr("x2", d.target.x).attr("y2", 0)
        .selectAll("stop")
        .data([
          {offset: "0%", color: color(d.source.name[0])},
          // {offset: "50%", color: "gray"},
          {offset: "100%", color: color(d.target.name[0])}
        ])
      .enter().append("stop")
        .attr("offset", function(d) { return d.offset; })
        .attr("stop-color", function(d) { return d.color; });
          return d.color = "url(#"+key+")";//color(d.source.name[0]);//
      })
      .sort(function(a, b) { return b.dy - a.dy; });

  link.append("title")
      .text(function(d) { return d.source.name + " → " + d.target.name + "\n" + format(d.value); });

  var node = svg.append("g").selectAll(".node")
      .data(energy.nodes)
      .enter().append("g")
      .attr("class", "node")
      .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
      .call(d3.behavior.drag()
      .origin(function(d) { return d; })
      .on("dragstart", function() { this.parentNode.appendChild(this); })
      .on("drag", dragmove));

  node.append("rect")
      .attr("height", function(d) { return d.dy; })
      .attr("width", sankey.nodeWidth())
      .style("fill", function(d) {
       return d.color = color(d.name[0]);
     })
      .style("stroke", function(d) { return d.color;})//d3.rgb(d.color).darker(2); })
    .append("title")
      .text(function(d) { return d.name + "\n" + format(d.value); });

  node.append("text")
      .attr("x", -6)
      .attr("y", function(d) { return d.dy / 2; })
      .attr("dy", ".35em")
      .attr("text-anchor", "end")
      .attr("transform", null)
      .text(function(d) { return d.name.split("-")[0]; })
    .filter(function(d) { return d.x < width / 2; })
      .attr("x", 6 + sankey.nodeWidth())
      .attr("text-anchor", "start");

  function dragmove(d) {
    d3.select(this).attr("transform", "translate(" + d.x + "," + (d.y = Math.max(0, Math.min(height - d.dy, d3.event.y))) + ")");
    sankey.relayout();
    link.attr("d", path);
  }
});
}


</script>

%rebase layout