#!env python2

import os.path
from bottle import route, run, template, view, static_file, request
from dcclient import DataCenterClient

client = DataCenterClient("tcp://10.1.1.211:32011")


@route('/')
def index():
    return template('index')


@route('/search')
@view('search')
def search():
    q = request.query.q or ""
    print 'searching', q
    result = client.searchAuthors(q)
    return dict(
        query=q,
        result=result
    )


@route('/influence')
@view('influence')
def influence():
    uid = int(request.query.uid)
    return dict(
        name="Jiawei Han",
        imgurl="http://pic1.aminer.org/picture/02927/Jiawei_Han_1330682737987.jpg",
        topics=[
            {
                "topic": "data mining",
                "score": 80,
                "influencers": [
                    ("Ke Wang", 100, "advisor"),
                    ("Latifur Khan", 80, "advisor")
                ],
                "influencees": [
                    ("Yizhou Sun", 80, "advisee"),
                    ("Bolin Ding", 60, "coauthor")
                ]
            },
            {
                "topic": "information retrival",
                "score": 40,
                "influencers": [
                    ("Ke Wang", 100, "advisor"),
                    ("Latifur Khan", 80, "advisor")
                ],
                "influencees": [
                    ("Yizhou Sun", 80, "advisee"),
                    ("Bolin Ding", 60, "coauthor")
                ]
            },
            {
                "topic": "XML Data",
                "score": 20,
                "influencers": [
                    ("Ke Wang", 100, "advisor"),
                    ("Latifur Khan", 80, "advisor")
                ],
                "influencees": [
                    ("Yizhou Sun", 80, "advisee"),
                    ("Bolin Ding", 60, "coauthor")
                ]
            }
        ]
    )


@route('/static/<path:path>')
def static(path):
    curdir = os.path.dirname(os.path.realpath(__file__))
    return static_file(path, root=curdir + "/static/")

run(server='auto', host='0.0.0.0', port=8080, reloader=True, debug=True)
