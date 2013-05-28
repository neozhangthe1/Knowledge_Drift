<div class="hero-unit">
    <h2>Influence graph of <b>{{name}}</b></h2>
    <div class="row-fluid">
        <div class="span2">
            <img src="{{imgurl}}" alt="{{name}}" style="width: 100%; height: auto;"/>
            <p>{{name}}
        </div>
        <div class="span7">
            <img src="/static/inf.png" alt="Influence Trends" style="width: 100%; height: auto;"/>
        </div>
        <div class="span3">
            <img src="/static/pie.png" alt="Influence Distribution" style="width: 100%; height: auto;"/>
        </div>
    </div>
</div>

<div class="topics">
    <div class="row-fluid topic">
        <div class="span4">
            <ul class="topic-list">
                %for topic in topics:
                <li>{{topic['topic']}} ({{topic['score']}})</li>
                %end
            </ul>
        </div>
        <div class="span8">
            %counter = 0
            %for topic in topics:
            <div class="topic-analysis index{{counter}}" style="{{"display:none" if counter != 0 else ""}}">
                <p> Influenced by {{name}}:
                %for influencee in topic["influencees"]:
                    <span>{{influencee[0]}} (because he/she is {{name}}'s {{influencee[2]}}) </span>
                %end
                <p> Influencers:
                %for influencer in topic["influencers"]:
                    <span>{{influencer[0]}} (because he/she is {{name}}'s {{influencer[2]}}) </span>
                %end
                %counter += 1
            </div>
            %end
        </div>
    </div>
</div>

%rebase layout
