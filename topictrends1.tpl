<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
	<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
	<title>SocialNetwork Analysis Engine</title>

	<script type="text/javascript" src="/static/jquery-2.0.1.min.js"></script>
	<script type="text/javascript" src="/static/js/timeline.min.js"></script>
	<script type="text/javascript" src="/static/js/sketchpad.min.js"></script>
	<script type="text/javascript" src="/static/json/data.json"></script>
	<script type="text/javascript" src="/static/json/pubs_dump.json"></script>
	<script type="text/javascript" src="/static/js/jsfr.min.js"></script>
	<link rel="stylesheet" type="text/css" href="/static/css/frame.css"></head>
<body>

<div id="graph">
	<div id="graph-inner" style="left: 0px; top: 0px;">
		<canvas id="canvas-graph" width="1126" height="513"></canvas>
		<canvas id="canvas-graph-nodes" width="1126" height="513"></canvas>
		<canvas id="canvas-graph-previous-statistics" width="1126" height="513" style="display: none;"></canvas>
		<canvas id="canvas-graph-overlay" width="1126" height="513" style="display: none;"></canvas>
		<canvas id="canvas-graph-over-lines" width="1126" height="513"></canvas>
		<canvas id="canvas-graph-over" width="1126" height="513"></canvas>
		<div id="graph-texts" class="hide"></div>
	</div>
	<div id="graph-sketchpad" style="width: 1126px; height: 513px;">
		<canvas width="1126" height="513"></canvas>
		<div class="toolbox">
			<span class="button btn-arrow">箭头</span>
			<span class="button btn-text">文本</span>
			<input type="text" class="input-text">
			<span class="btn-color btn-color-0-0-0" style="color: rgb(0, 0, 0); background-color: rgb(0, 0, 0);"></span>
			<span class="btn-color btn-color-31-119-180" style="color: rgb(31, 119, 180); background-color: rgb(31, 119, 180);"></span>
			<span class="btn-color btn-color-255-127-14" style="color: rgb(255, 127, 14); background-color: rgb(255, 127, 14);"></span>
			<span class="btn-color btn-color-44-160-44" style="color: rgb(44, 160, 44); background-color: rgb(44, 160, 44);"></span>
			<span class="button btn-remove">删除</span>
			<span class="button btn-close">关闭</span>
		</div>
	</div>
</div>

<div id="timeline" class="pkuvis-timeline" style="height: 120px; cursor: crosshair;">
	<!--   <span class="set-height" style="display: none;">120</span>
<span class="set-colorscheme" style="display: none;">black</span>
<span class="-enable-yticks" style="display: none;"></span>
-->
<!--<span class="disable-track-range"></span>
-->
</div>


<script type="text/javascript" src="/static/js/frame.min.js"></script>
<script type="text/javascript" src="/static/js/main.min.js"></script>

</body>
</html>