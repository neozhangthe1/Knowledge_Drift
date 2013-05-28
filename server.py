#!env python2

import os.path
from bottle import route, run, template, view, debug, static_file, request
from dcclient import DataCenterClient

debug(True)
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


@route('/static/<path:path>')
def static(path):
    curdir = os.path.dirname(os.path.realpath(__file__))
    return static_file(path, root=curdir + "/static/")

run(server='auto', host='0.0.0.0', port=8080, reloader=True)
