<form class="search" method="get">
	<h2>Search...</h2>
	<input class="input-block-level" name="q" value="{{query}}"/>
	<button class="btn btn-small btn-primary" type="submit">Search</button>
</form>

<p>Hot queries:
<span><a href="search?q=data%20mining">data mining</a></span>
<span><a href="search?q=machine%20learning">machine learning</a></span>

<p><a href="topictrends?{{encoded_query}}">topic trends</a>

<p>{{ count }} results
<div class="results">
<ul>
%for item in results:
	<li class="result-item">
		<div class="name">
			{{item['name']}}
			<a href="http://arnetminer.org/person/-{{item['id']}}.html">AMiner</a>
			<a href="{{item['id']}}/influence">Influence Analysis</a>
		</div>
	</li>
%end
</ul>
</div>

%rebase layout
