from dcclient.dcclient import DataCenterClient
from teclient.teclient import TermExtractorClient
from collections import defaultdict
from bs4 import UnicodeDammit
import numpy
import json
data_center = DataCenterClient("tcp://10.1.1.211:32011")
term_extractor = TermExtractorClient()


class TopicTrend(object):
    def __init__(self):
        print "INIT TOPIC TREND"
        self.parse_topic_model()
        self.parse_topic_graph()

    def parse_topic_model(self):
        print "PARSING TOPIC MODEL"
        f = open("G:\\topic-evolution\\model_dat_130627.out")
        f_v = open("G:\\word.in")
        count = -1 #line count
        index = -1 #current person or word
        flag = 0 #skip the 2 param line
        num_topic = None
        num_person = None
        num_word = None
        num_time = None
        cur_time = 0
        p_topic_given_person_y = defaultdict(dict)
        p_topic_given_term_y = defaultdict(dict)
        vocab = {}
        for line in f_v:
            x = line.strip().split(" ")
            vocab[" ".join(x[1:])] = int(x[0])
        for line in f:
            count += 1
            if count == 0:
                x = line.strip().split(" ")
                num_topic = int(x[0])
                num_person = int(x[1])
                num_word = int(x[2])
                num_time = int(x[3])
            elif flag < 2:
                y = line.strip().split(" ")
                print "SKIP", cur_time, index, count, len(y)
                flag += 1
                index = -1
            else:            
                index += 1
                if index < num_person:
                    y = line.strip().split(" ")
                    if len(y) != num_topic:
                        print "ERROR when parsing person", cur_time, index, count, len(y)
                    pt = [float(p) for p in y]
                    p_topic_given_person_y[cur_time][index] = pt
                elif index < num_person+num_word:
                    y = line.strip().split(" ")
                    if len(y) != num_topic:
                        print "ERROR when parsing term", cur_time, index, count, len(y)
                    pt = [float(p) for p in y]
                    p_topic_given_term_y[cur_time][index - num_person] = pt
                    if index == num_person+num_word-1:
                        flag = 0
                        cur_time += 1
        self.num_topic = num_topic
        self.num_person = num_person
        self.num_word = num_word
        self.num_time = num_time
        self.p_topic_given_person_y = p_topic_given_person_y
        self.p_topic_given_term_y = p_topic_given_term_y
        self.vocab = vocab

    def parse_topic_graph(self):
        print "PARSING TOPIC GRAPH"
        f_graph = open("G:\\topic-evolution\\topicgraph_130627.out")
        f_topic = open("G:\\topic-evolution\\topic_130627.out")
        flag = 0
        count = 0
        index = 0 #index of nodes
        start_time = 2002
        cur_year = None 
        cur_time = None
        node_meta = defaultdict(dict)
        link_meta = []
        node_dict = {}
        graph = {"nodes":[], "links":[]}
        for line in f_topic:
            if "#" in line:
                x = line.strip().strip(":").split("#")
                cur_year = int(x[1]) + start_time
                continue
            elif ":" in line and line[:5] == "Topic":
                flag = 0
                x = line.strip().split(":")
                w = float(x[1].strip())
                id = x[0].split(" ")[1]
                continue
            else:
                if flag == 0:
                    l1 = line.strip().split("0")[0].strip()
                    flag += 1
                elif flag == 1:
                    l2 = line.strip().split("0")[0].strip()
                    flag += 1
                elif flag == 2:
                    flag += 1
                    l3 = line.strip().split("0")[0].strip()
                    node_meta[cur_year-start_time][int(id)] = {"label":l1+"-"+l2+"-"+l3, "weight":w, "source":[], "target":[], "sigma_1":0., "sigma_2":0.} #simga is the sum of link weight use for normalization
        for line in f_graph:
            if ":" in line:
                cur_time = int(line.split(":")[0])
            else:
                x = line.strip("\n").split(" ")
                node_meta[cur_time][int(x[0])]["target"].append({"key":str(cur_time+1)+"-"+str(x[1]), "weight":float(x[2])})
                node_meta[cur_time+1][int(x[1])]["source"].append({"key":str(cur_time)+"-"+str(x[0]), "weight":float(x[2])})
                node_meta[cur_time][int(x[0])]["sigma_1"] += float(x[2])
                node_meta[cur_time+1][int(x[1])]["sigma_2"] += float(x[2])
                link_meta.append({"source":str(cur_time)+"-"+str(x[0]), "target":str(cur_time+1)+"-"+str(x[1]), "similarity":float(x[2])})
        for y in node_meta:
            for n in node_meta[y]:
                graph["nodes"].append({"name":node_meta[y][n]["label"], "w":node_meta[y][n]["weight"], "pos":y})   
                node_dict[str(y)+"-"+str(n)] = index
                index += 1
        for y in node_meta:
            for n in node_meta[y]:
                source_weight = node_meta[y][n]["weight"] #weight of the source topic
                source_sigma = node_meta[y][n]["sigma_1"] #sum of the link weight of source topic
                for t in node_meta[y][n]["target"]:
                    x = t["key"].split("-")
                    link_weight = t["weight"]
                    target = node_meta[int(x[0])][int(x[1])]
                    target_weight = target["weight"]
                    target_sigma = target["sigma_2"]
                    graph["links"].append({"source":node_dict[str(y)+"-"+str(n)],
                                          "target":node_dict[t["key"]],
                                          "w1":source_weight * link_weight / source_sigma,
                                          "w2":target_weight * link_weight / target_sigma}) #topic weight * link weight / sigma
        self.node_meta = node_meta
        self.link_meta = link_meta
        self.node_dict = node_dict
        self.start_year = start_time
        self.topic_graph = graph

    def query_topic_trends(self, query, threshold=0.0001):
        print "MATCHING QUERY TO TOPICS", query, threshold
        query = query.lower()
        words = []
        choose_topic = defaultdict(list)
        #check if the term is in the vocabulary
        if query in self.vocab:
            print "FOUND WORD", query, self.vocab[query]
            words.append(self.vocab[query])
        #if not, check if the words in the term exists in the vocabulary
        else:
            terms = query.split(" ")
            for t in terms:
                if t in self.vocab:
                    print "FOUND WORD", t, self.vocab[t]
                    words.append(self.vocab[t]) 
        #choose topics related to the query term
        for y in self.p_topic_given_term_y:
            for t in words:
                p_topic = self.p_topic_given_term_y[y][t]
                for i in range(len(p_topic)):
                    if p_topic[i] > threshold:
                        choose_topic[y].append(i)
        print len(choose_topic), "topics are choosed"
        return self.render_topic_graph(choose_topic)

    """
    backup method
    """
    def query_topic_trends_or(self, query, threshold=0.0001):
        print "MATCHING QUERY TO TOPICS", query, threshold
        query = query.lower()
        words = []
        choose_topic = defaultdict(list)
        #check if the term is in the vocabulary
        if query in self.vocab:
            print "FOUND WORD", query, self.vocab[query]
            words.append(self.vocab[query])
        #if not, check if the words in the term exists in the vocabulary
        else:
            terms = query.split(" ")
            for t in terms:
                if t in self.vocab:
                    print "FOUND WORD", t, self.vocab[t]
                    words.append(self.vocab[t]) 
        #choose topics related to the query term
        for y in self.p_topic_given_term_y:
            for t in words:
                p_topic = self.p_topic_given_term_y[y][t]
                for i in range(len(p_topic)):
                    if p_topic[i] > threshold:
                        choose_topic[y].append(i)
        print len(choose_topic), "topics are choosed"
        return self.render_topic_graph_or(choose_topic)

    """
    backup method
    """
    def render_topic_graph(self, nodes):
        pass 

    def render_topic_graph(self, nodes):
        print "RENDERING TOPIC GRAPH"
        graph = {"nodes":[], "links":[]}
        node_meta = self.node_meta
        link_meta = self.link_meta
        node_dict = {}
        index = 0
        for y in nodes:
            for n in nodes[y]:
                graph["nodes"].append({"name":node_meta[y][n]["label"], "w":node_meta[y][n]["weight"], "pos":y})   
                node_dict[str(y)+"-"+str(n)] = index
                index += 1
        sigma_1 = defaultdict(float)
        sigma_2 = defaultdict(float)
        for link in link_meta:
            if node_dict.has_key(link["source"]) and node_dict.has_key(link["target"]):
                 sigma_1[link["source"]] += link["similarity"]
                 sigma_2[link["target"]] += link["similarity"]
        for y in nodes:
            for n in nodes[y]:
                source_weight = node_meta[y][n]["weight"] #weight of the source topic
                source_sigma = sigma_1[str(y)+"-"+str(n)] #sum of the link weight of source topic
                for t in node_meta[y][n]["target"]:
                    x = t["key"].split("-")
                    link_weight = t["weight"]
                    target = node_meta[int(x[0])][int(x[1])]
                    target_weight = target["weight"]
                    target_sigma = sigma_2[t["key"]]
                    if (node_dict.has_key(str(y)+"-"+str(n))) and (node_dict.has_key(t["key"])): 
                        graph["links"].append({"source":node_dict[str(y)+"-"+str(n)],
                                              "target":node_dict[t["key"]],
                                              "w1":source_weight * link_weight / source_sigma,
                                              "w2":target_weight * link_weight / target_sigma}) #topic weight * link weight / sigma
        return graph

    def render_topic_graph_old(self, nodes):
        f_graph = open("G:\\topic-evolution\\topicgraph_130627.out")
        f_topic = open("G:\\topic-evolution\\topic_130627.out")
        flag = 0
        count = 0
        index = 0 #index of nodes
        start_time = 2002
        cur_year = None 
        cur_time = None
        node_meta = defaultdict(dict)
        node_dict = {}
        graph = {"nodes":[], "links":[]}
        for line in f_topic:
            if "#" in line:
                x = line.strip().strip(":").split("#")
                cur_year = int(x[1]) + start_time
                continue
            elif ":" in line and line[:5] == "Topic":
                flag = 0
                x = line.strip().split(":")
                w = float(x[1].strip())
                id = x[0].split(" ")[1]
                continue
            else:
                if flag == 0:
                    l1 = line.strip().split("0")[0].strip()
                    flag += 1
                elif flag == 1:
                    l2 = line.strip().split("0")[0].strip()
                    flag += 1
                elif flag == 2:
                    flag += 1
                    l3 = line.strip().split("0")[0].strip()
                    node_meta[cur_year-start_time][int(id)] = {"label":l1+"-"+l2+"-"+l3, "weight":w}
        for y in nodes:
            for n in nodes[y]:
                graph["nodes"].append({"name":node_meta[y][n]["label"], "pos":y})   
                node_dict[str(y)+"-"+str(n)] = index
                index += 1
        for line in f_graph:
            if ":" in line:
                cur_time = int(line.split(":")[0])
            else:
                x = line.strip("\n").split(" ")
                if (node_dict.has_key(str(cur_time)+"-"+x[0])) and (node_dict.has_key(str(cur_time+1)+"-"+x[1])): 
                    graph["links"].append({"source":node_dict[str(cur_time)+"-"+str(x[0])],
                                           "target":node_dict[str(cur_time+1)+"-"+str(x[1])],
                                           "value":node_meta[cur_time+1][int(x[1])]["weight"]})
        return graph

    def render_topic_graph_complete(self):
        f_graph = open("G:\\topic-evolution\\topicgraph_130627.out")
        f_topic = open("G:\\topic-evolution\\topic_130627.out")
        flag = 0
        count = 0
        index = 0 #index of nodes
        start_time = 2002
        cur_year = None 
        cur_time = None
        node_meta = defaultdict(dict)
        link_meta = []
        node_dict = {}
        graph = {"nodes":[], "links":[]}
        for line in f_topic:
            if "#" in line:
                x = line.strip().strip(":").split("#")
                cur_year = int(x[1]) + start_time
                continue
            elif ":" in line and line[:5] == "Topic":
                flag = 0
                x = line.strip().split(":")
                w = float(x[1].strip())
                id = x[0].split(" ")[1]
                continue
            else:
                if flag == 0:
                    l1 = line.strip().split("0")[0].strip()
                    flag += 1
                elif flag == 1:
                    l2 = line.strip().split("0")[0].strip()
                    flag += 1
                elif flag == 2:
                    flag += 1
                    l3 = line.strip().split("0")[0].strip()
                    node_meta[cur_year-start_time][int(id)] = {"label":l1+"-"+l2+"-"+l3, "weight":w, "source":[], "target":[], "sigma_1":0., "sigma_2":0.} #simga is the sum of link weight use for normalization
        for line in f_graph:
            if ":" in line:
                cur_time = int(line.split(":")[0])
            else:
                x = line.strip("\n").split(" ")
                node_meta[cur_time][int(x[0])]["target"].append({"key":str(cur_time+1)+"-"+str(x[1]), "weight":float(x[2])})
                node_meta[cur_time+1][int(x[1])]["source"].append({"key":str(cur_time)+"-"+str(x[0]), "weight":float(x[2])})
                node_meta[cur_time][int(x[0])]["sigma_1"] += float(x[2])
                node_meta[cur_time+1][int(x[1])]["sigma_2"] += float(x[2])
                link_meta.append({"source":str(cur_time)+"-"+str(x[0]), "target":str(cur_time+1)+"-"+str(x[1]), "similarity":float(x[2])})
        for y in node_meta:
            for n in node_meta[y]:
                graph["nodes"].append({"name":node_meta[y][n]["label"], "w":node_meta[y][n]["weight"], "pos":y})   
                node_dict[str(y)+"-"+str(n)] = index
                index += 1
        for y in node_meta:
            for n in node_meta[y]:
                source_weight = node_meta[y][n]["weight"] #weight of the source topic
                source_sigma = node_meta[y][n]["sigma_1"] #sum of the link weight of source topic
                for t in node_meta[y][n]["target"]:
                    x = t["key"].split("-")
                    link_weight = t["weight"]
                    target = node_meta[int(x[0])][int(x[1])]
                    target_weight = target["weight"]
                    target_sigma = target["sigma_2"]
                    graph["links"].append({"source":node_dict[str(y)+"-"+str(n)],
                                          "target":node_dict[t["key"]],
                                          "w1":source_weight * link_weight / source_sigma,
                                          "w2":target_weight * link_weight / target_sigma}) #topic weight * link weight / sigma
        return graph

def main():
    pass
if __name__ == "__main__":
    pass
    #main()


def extractPublication(p):
    #extract key terms from abstract and title
    text = p.title.lower() + " . " + p.abs.lower()
    terms = term_extractor.extractTerms(text)
    #extract citation network
    children = []
    children_ids = []
    parents = []
    parents_ids = []
    for x in p.cited_by_pubs:
        children.append(str(x))
        children_ids.append(x)
    for x in p.cite_pubs:
        parents.append(str(x))
        parents_ids.append(x)

    return children_ids, parents_ids, terms

def get_topic(query):
    x =  data_center.searchPublications(query)
    key_term_set = set()
    terms_frequence = defaultdict(int)
    year_terms_frequence = defaultdict(lambda: defaultdict(int))
    
    #extract key terms and citation network from publications
    for p in x.publications:
        if p.year <= 1980:
            continue
        children, parents, terms = extractPublication(p)
        for t in terms:
            key_term_set.add(t)

    #counting term frequence
    for p in x.publications:
        for t in key_term_set:
            if t in p.title.lower() or t in p.abs.lower():
                terms_frequence[t] += 1
                year_terms_frequence[p.year][t] += 1

    #normalize year term frequence
    def normalize(ytf):
        nor_ytf = defaultdict(lambda: defaultdict(int))
        for y in ytf:
            year_max = 0
            for t in ytf[y]:
                if year_max < ytf[y][t]:
                    year_max = ytf[y][t]
            for t in ytf[y]:
                ytf[y][t] /= float(year_max)
        return nor_ytf
    normalized_year_terms_frequence = year_terms_frequence

    #build graph
    graph = {}
    graph["nodes"] = []
    graph["links"] = []
    start_year = min(year_terms_frequence.keys())
    node_index = 0
    for t in terms_frequence:
        pre = None
        for y in year_terms_frequence:
            if year_terms_frequence[y][t] > 0:
                graph["nodes"].append({"name":t, "pos":y-start_year})
                if pre is not None:
                    graph["links"].append({"source": pre, "target": node_index, 
                                           "value": year_terms_frequence[y][t]})
                pre = node_index
                node_index += 1
                
    #dump json        
    return json.dumps(graph)

def make_data():
    from collections import defaultdict
    f = open("G:\\topic-evolution\\topicgraph_cluster_2.out")
    f_1 = open("G:\\topic-evolution\\topics_cluster_2.out")
    graph = {"nodes":[], "links":[]}
    node_dict = {}
    count = 0
    start_time = 2002
    cur_time = None
    topic_label = defaultdict(dict)
    cur_year = None 
    flag = 0
    for line in f_1:
        if line[:4] == "Year":
            x = line.strip().split(" ")
            cur_year = int(x[1])
            continue
        elif ":" in line and line[:5] == "Topic":
            flag = 0
            x = line.strip().split(":")
            w = float(x[1].strip())
            id = x[0].split(" ")[1]
            continue
        else:
            if flag == 0:
                l1 = line.strip().split("0")[0].strip()
                flag += 1
            elif flag == 1:
                l2 = line.strip().split("0")[0].strip()
                flag += 1
            elif flag == 2:
                flag += 1
                l3 = f_1.next().strip().split("0")[0].strip()
                topic_label[cur_year][id] = {"label":l1, "weight":w}
    for line in f:
        if ":" in line:
            cur_time = int(line.split(":")[0]) + start_time
        else:
            x = line.strip("\n").split(" ")
            for i in range(2):
                if not str(x[i])+"-"+str(cur_time+i) in node_dict:
                    node_dict[(str(x[i])+"-"+str(cur_time+i))] = count
                    count+=1
                    #graph["nodes"].append({"name":str(x[i])+"-"+str(cur_time+i),"pos":cur_time-start_time+i})
                    graph["nodes"].append({"name":topic_label[cur_time+i][x[i]]["label"],"pos":cur_time-start_time+i})
            graph["links"].append({"source":node_dict[str(x[0])+"-"+str(cur_time)],
                                   "target":node_dict[str(x[1])+"-"+str(cur_time+1)],
                                   "value":topic_label[cur_time+1][x[1]]["weight"]})
    f_out = open("data.json","w")
    import json
    f_out.write(json.dumps(graph))
    f_out.close()
            
def make_data():
    from collections import defaultdict
    f = open("G:\\topic-evolution\\topicgraph_kdd.out")
    f_1 = open("G:\\topic-evolution\\topic_kdd.out")
    graph = {"nodes":[], "links":[]}
    node_dict = {}
    count = 0
    start_time = 2002
    cur_time = None
    topic_label = defaultdict(dict)
    cur_year = None 
    flag = 0
    topic_weight = parseTopic()
    for line in f_1:
        if "#" in line:
            x = line.strip().strip(":").split("#")
            cur_year = int(x[1]) + start_time
            continue
        elif ":" in line and line[:5] == "Topic":
            flag = 0
            x = line.strip().split(":")
            w = float(x[1].strip())
            id = x[0].split(" ")[1]
            continue
        else:
            if flag == 0:
                l1 = line.strip().split("0")[0].strip()
                flag += 1
            elif flag == 1:
                l2 = line.strip().split("0")[0].strip()
                flag += 1
            elif flag == 2:
                flag += 1
                l3 = f_1.next().strip().split("0")[0].strip()
                topic_label[cur_year][id] = {"label":l1, "weight":w}
    for line in f:
        if ":" in line:
            cur_time = int(line.split(":")[0]) + start_time
        else:
            x = line.strip("\n").split(" ")
            for i in range(2):
                if not str(x[i])+"-"+str(cur_time+i) in node_dict:
                    node_dict[(str(x[i])+"-"+str(cur_time+i))] = count
                    count+=1
                    #graph["nodes"].append({"name":str(x[i])+"-"+str(cur_time+i),"pos":cur_time-start_time+i})
                    graph["nodes"].append({"name":topic_label[cur_time+i][x[i]]["label"],"pos":cur_time-start_time+i})
            graph["links"].append({"source":node_dict[str(x[0])+"-"+str(cur_time)],
                                   "target":node_dict[str(x[1])+"-"+str(cur_time+1)],
                                   "value":topic_weight[cur_time-start_time+1][int(x[0])]})
    f_out = open("data.json","w")
    import json
    f_out.write(json.dumps(graph))
    f_out.close()

def build_model():
    import logging, gensim, bz2
    import re
    from collections import defaultdict
    from sklearn.feature_extraction.text import CountVectorizer
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #stoplist = 
    for y in [1981, 2010]:
        f = open("pubs\\"+str(y))
        flag = 0
        titles = []
        abstracts = []
        confs = []
        ids = []
        documents = []
        content = None
        for line in f:
            if flag == 0:
                ids.append(int(line.strip()))
                flag += 1
            elif flag == 1:
                confs.append(line.strip())
                flag += 1
            elif flag == 2:
                titles.append(line.strip())
                content = line
                flag += 1
            elif flag == 3:
                abstracts.append(line.strip())
                content += line
                content.replace("\n"," ").lower()
                documents.append(content)
                flag = 0
        vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,3), max_df=0.95, min_df=5)
        X = vectorizer.fit_transform(documents)
        X = X.tolil();
        doc_dict = defaultdict(list)
        doc_ids = X.nonzero()[0]
        word_ids = X.nonzero()[1]
        for i in range(len(doc_ids)):
            doc_dict[doc_ids[i]].append((word_ids[i], X[doc_ids[i],word_ids[i]]))    
        id2word = {}
        for w in vectorizer.vocabulary_:
            id2word[vectorizer.vocabulary_[w]] = w
        lda = gensim.models.ldamodel.LdaModel(corpus=doc_dict.values(), id2word=id2word, 
                                              num_topics=30, update_every=1, chunksize=10, passes=1)    
        hdp = gensim.models.hdpmodel.HdpModel(corpus=doc_dict.values(), id2word=id2word, chunksize=10)


def parseTopic():
    import numpy
    from collections import defaultdict
    f = open("G:\\topic-evolution\\model_dat.out")
    count = -1
    num_topic = None
    num_person = None
    num_word = None
    num_time = None
    cur_time = 0
    index = -1
    p_t = defaultdict(dict) #p(t) = sum(p(t|person) determine the hotness of a topic
    for line in f:
        count += 1
        if count == 0:
            x = line.strip().split(" ")
            num_topic = int(x[0])
            num_person = int(x[1])
            num_word = int(x[2])
            num_time = int(x[3])
        if count == 1:
            continue
        if count > 1:
            index += 1
            if index < num_topic:
                y = line.strip().split(" ")
                if len(y) != num_person:
                    print "ERROR", index, count, len(y)
                pt = [float(p) for p in y]
                pt_mean = numpy.mean(pt)
                weight = 0
                for p in pt:
                    #omit the person with low p(t|person)
                    if p > 0.1:
                        weight += p
                p_t[cur_time][index] = weight
            else:
                if (index) == (num_topic + num_word + 1):
                    index = -1
                    cur_time += 1
    return p_t

def parse_cluster():
    import numpy
    from collections import defaultdict
    from sklearn.cluster import Ward, KMeans
    from sklearn.decomposition import PCA
    f = open("G:\\topic-evolution\\model_dat.out")
    count = -1
    num_topic = None
    num_person = None
    num_word = None
    num_time = None
    cur_time = 0
    index = -1
    item_dict = defaultdict(dict) #index of item in X
    X = [] #feature_vector
    for line in f:
        count += 1
        if count == 0:
            x = line.strip().split(" ")
            num_topic = int(x[0])
            num_person = int(x[1])
            num_word = int(x[2])
            num_time = int(x[3])
        if count == 1:
            continue
        if count > 1:
            index += 1
            if index < num_topic:
                y = line.strip().split(" ")
                if len(y) != num_person:
                    print "ERROR", index, count, len(y)
                pt = [float(p) for p in y]
                X.append(pt)
                item_dict[cur_time][index] = index+(cur_time * num_topic)
            else:
                if (index) == (num_topic + num_word + 1):
                    index = -1
                    cur_time += 1
    ward = Ward(n_clusters = 20).fit(X)
    kmeans = KMeans(init='k-means++', n_clusters=20).fit(X)
            
    pca = PCA(n_components=200).fit(X)
    pca_kmeans = KMeans(init=pca.components_, n_clusters=200, n_init=1).fit(X)

    f = open("G:\\topic-evolution\\topicgraph_kdd.out")
    f_1 = open("G:\\topic-evolution\\topic_kdd.out")
    node_dict = {}
    count = 0
    start_time = 2002
    cur_time = None
    topic_label = defaultdict(dict)
    cur_year = None 
    flag = 0
    topic_weight = parseTopic()
    for line in f_1:
        if "#" in line:
            x = line.strip().strip(":").split("#")
            cur_year = int(x[1]) + start_time
            continue
        elif ":" in line and line[:5] == "Topic":
            flag = 0
            x = line.strip().split(":")
            w = float(x[1].strip())
            id = x[0].split(" ")[1]
            continue
        else:
            if flag == 0:
                l1 = line.strip().split("0")[0].strip()
                flag += 1
            elif flag == 1:
                l2 = line.strip().split("0")[0].strip()
                flag += 1
            elif flag == 2:
                flag += 1
                l3 = f_1.next().strip().split("0")[0].strip()
                topic_label[cur_year][int(id)] = {"label":l1, "weight":w}
    labels = defaultdict(lambda :defaultdict(list))
    nodes = defaultdict(dict)
    graph = {"nodes":[], "links":[]}
    for y in item_dict:
        for i in item_dict[y]:
            labels[y][pca_kmeans.labels_[item_dict[y][i]]].append(i)
    idx = 0
    for y in labels:
        for n in labels[y]:
            label = ""
            for i in labels[y][n]:
                label+=topic_label[y+start_time][i]["label"]
            graph["nodes"].append({"name":label , "pos":y, "cluster":int(n)})
            nodes[y][n] = idx
            idx += 1
    for y in labels:
        for n in labels[y]:
            if y > 0:
                if labels[y-1].has_key(n):
                    graph["links"].append({"cluster":int(n), 
                                           "source":nodes[y][n],
                                           "target":nodes[y-1][n],
                                           "value":len(labels[y][n]),
                                           })
    f_out = open("sankey.json","w")
    import json
    f_out.write(json.dumps(graph))
    f_out.close()


