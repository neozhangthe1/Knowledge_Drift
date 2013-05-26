<p> searching: {{ query }}
<p>{{ result.total_count }} results

<div class="result">
%for author in result.authors:
	<div class="result-author">
		<div>{{author.naid}}</div>
		<div class="result-author-name">{{author.names[0]}}</div>
		<div class="result-author-email">{{author.email}}</div>
	</div>
%end
</div>

%rebase layout
