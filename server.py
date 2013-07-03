#!env python2

import os.path
from bottle import route, run, template, view, static_file, request, urlencode
from dcclient import DataCenterClient
from topic_trend_analysis import TopicTrend
import sample_data

client = DataCenterClient("tcp://10.1.1.211:32011")
print "building topic trend"
topic_trend_client = TopicTrend()
print "building topic trend finished"


@route('/')
def index():
    return template('index')


@route('/academic/search')
@view('search')
def search():
    q = request.query.q or ''
    print 'searching', q, 'in academic'
    result = client.searchAuthors(q)
    return dict(
        query=q,
        count=result.total_count,
        results=[
            dict(
                id=a.naid,
                name=a.names[0],
                email=a.email
            ) for a in result.authors
        ],
        encoded_query=urlencode({"q": q})
    )


@route('/patent/search')
@view('search')
def search():
    q = request.query.q or ''
    print 'searching', q, 'in patent'
    return dict(
        query=q,
        count=0,
        results=[],
        encoded_query=urlencode({"q": q})
    )


@route('/weibo/search')
@view('search')
def search():
    q = request.query.q or ''
    print 'searching', q, 'in weibo'
    return dict(
        query=q,
        count=0,
        results=[],
        encoded_query=urlencode({"q": q})
    )


@route('/<data>/topictrends')
@view('topictrends')
def search(data):
    q = request.query.q or ''
    print 'rendering trends for', q, 'on', data
    return dict(
        query=q
    )

@route('/<data>/terms')
def search(data):
    q = request.query.q or ''
    print 'rendering terms for', q, 'on', data
    return topic_trend_client.query_terms(q)

@route('/<data>/render')
def topic_trends(data):
    q = request.query.q or ''
    threshold = request.query.threshold or ''
    print 'rendering trends for', q, threshold, 'on', data
    return topic_trend_client.query_topic_trends(q, float(threshold))


@route('/<data>/<uid:int>/influence/trends.tsv')
def influence_trends(data, uid):
    return open('static/influence.tsv')


@route('/<data>/<uid:int>/influence/topics/<date>')
@view('influence_topics')
def influence_topics(data, uid, date):
    # TODO return topics for the given data
    return sample_data.influence_topics


@route('/<data>/<uid:int>/influence')
@view('influence')
def influence(data, uid):
    return sample_data.influence_index


@route('/static/<path:path>')
def static(path):
    curdir = os.path.dirname(os.path.realpath(__file__))
    return static_file(path, root=curdir + '/static/')

run(server='auto', host='0.0.0.0', port=8081, reloader=True, debug=True)
