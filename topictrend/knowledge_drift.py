import pickle
import mongo
import numpy as np
from bs4 import UnicodeDammit
from collections import defaultdict

MONGO_IP = "10.1.1.111"
MONGO_PORT = 27106
terms_given_document = pickle.load(open("/home/yutao/Data/knowledge-drift/t.pickle","rb"))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#get aminer data from data center
def get_data_from_datacenter():
    from dcclient import DataCenterClient
    data_center = DataCenterClient("tcp://10.1.1.211:32011")
    f = open("E:\\ids.txt")
    f.next()
    ids = []
    for line in f:
        x = line.split("\n")
        ids.append(int(x[0]))
    print len(ids)
    for i in range(len(ids)/1000):
        print "DUMP %s"%(i*1000)
        x = c.getPublicationsById(ids[i*1000:(i+1)*1000])
        id_set = set(ids)
        count = 0
        abs = {}
        conf = {}
        authors = {}
        title = {}
        year = {}
        for p in x.publications:
            abs[p.id] = p.abs.replace("\n"," ").replace("\t"," ")
            conf[p.id] = p.jconf_name
            authors[p.id] = ",".join([str(a) for a in p.author_ids])
            title[p.id] = p.title
            year[p.id] = p.year
        for p in abs:
            if len(abs[p]) > 2:
                f_out.write("%s\n%s\n%s\n%s\n%s\n%s\n"%(p,year[p],conf[p],authors[p],title[p],UnicodeDammit(abs[p]).markup))

def get_document_meta():
    document_meta = {}
    with open("/home/yutao/Data/knowledge-drift/publication.txt") as pub_f:
        for line in pub_f:
            x = line.strip("\n").split("\t")
            document_meta[int(x[0])] = {"year":int(x[3]), "title":x[1]}
    return document_meta


def get_term_freq():
    term_dict = defaultdict(int)
    for d in terms_given_document:
        for t in terms_given_document[d]:
            term_dict[t.lower()] += 1
    return term_dict

def get_term_freq_given_time(sorted_term_freq, document_meta):
    start_time = 1970
    end_time = 2013
    term_index = {}
    term_list = []
    idx = 0
    for t in sorted_term_freq:
        term_list.append(t[0])
        term_index[t[0]] = idx
        idx += 1
    num_terms = idx

    #term year freq matrix
    term_freq_given_time = np.zeros((num_terms, end_time-start_time+1))
    for d in terms_given_document:
        if document_meta.has_key(int(d)):
            time = document_meta[int(d)]["year"]
            if time > end_time or time < start_time:
                continue
            for t in terms_given_document[d]:
                term_freq_given_time[term_index[t.lower()], time-start_time] += 1
    return term_freq_given_time, term_index, term_list


def build_term_document_matrix():
    pass

def build_term_venue_matrix():
    pass

if __name__ == "__main__":
    term_dict = get_term_freq()
    sorted_term_freq = sorted(term_dict.items(), key=lambda x:x[1], reverse=True)
    document_meta = get_document_meta()
    term_freq_given_time, term_index, term_list = get_term_freq_given_time(sorted_term_freq, document_meta)


