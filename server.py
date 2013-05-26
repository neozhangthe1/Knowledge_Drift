#!env python2

from bottle import route, run, template, view, debug
from dcclient import DataCenterClient

debug(True)
client = DataCenterClient("tcp://10.1.1.211:32011")


@route('/')
def index():
    return template('index')


@route('/search')
@view('search')
def search(q='data mining'):
    result = client.searchAuthors("data mining")
    return dict(
        query=q,
        result=result
    )

run(host='0.0.0.0', port=8080, reloader=True)
