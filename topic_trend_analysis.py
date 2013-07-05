from dcclient.dcclient import DataCenterClient
from teclient.teclient import TermExtractorClient
from utils.algorithms import jaccard_similarity
from collections import defaultdict
from bs4 import UnicodeDammit
import numpy as np
import json     
import gensim
import pickle
import networkx as nx
import logging
from sklearn.cluster import Ward, KMeans, MiniBatchKMeans
data_center = DataCenterClient("tcp://10.1.1.211:32011")
term_extractor = TermExtractorClient()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def extract_document(p):
    #extract key terms from abstract and title
    text = p.title.lower() + " . " + p.abs.lower()
    #terms = term_extractor.extractTerms(text)
    #extract citation network
    children = []
    children_ids = []
    parents = []
    parents_ids = []
    authors = []
    authors_ids = []
    for x in p.cited_by_pubs:
        children.append(str(x))
        children_ids.append(x)
    for x in p.cite_pubs:
        parents.append(str(x))
        parents_ids.append(x)
    for x in p.authors:
        authors.append(x)
    return children_ids, parents_ids, authors, authors_ids

class TopicTrend(object):
    def __init__(self):
        print "INIT TOPIC TREND"
        self.author_result = None
        #term info
        self.num_terms = 0
        self.term_list = None
        self.term_index = None
        self.term_freq = None
        self.term_freq_given_time = None
        self.term_freq_given_person = None
        self.term_freq_given_person_time = None
        self.co_word_maxtrix = None
        #author info
        self.num_authors = 0
        self.author_list = None
        self.author_index = None
        #document info
        self.num_documents = 0
        self.document_list = None
        self.document_index = None
        self.document_list_given_time = None        
        #time info
        self.time_window = None
        self.time_slides = None
        self.num_time_slides = None
        self.start_time = None
        self.end_time = None
        #cluster info
        self.num_local_clusters = 5
        self.num_global_clusters = 5
        self.local_clusters = None
        self.global_clusters = None
        self.gloabl_feature_vectors_index = None
        #self.parse_topic_model()
        #self.parse_topic_graph()
        self.graph = None

    def query_topic_trends(self, query, threshold=0.0001):
        logging.info("MATCHING QUERY TO TOPICS", query, threshold)
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
    setter
    """
    #there will be 10 time window by default
    def set_time_slides(self, time_window):
        if time_window is not None:
            self.time_window = time_window
        else:
            self.time_window = 1 + int(np.floor((float(self.end_time - self.start_time) / 11)))
        self.num_time_slides = int(np.ceil((float(self.end_time - self.start_time) / self.time_window)))
        self.time_slides = []
        cur_time = self.start_time
        for i in range(self.num_time_slides):
            cur_slide = []
            for j in range(self.time_window):
                cur_slide.append(cur_time)
                cur_time += 1
            self.time_slides.append(cur_slide)

    def set_time(self, time):
        if time < self.start_time or self.start_time is None:
            self.start_time = time
        if time > self.end_time or self.end_time is None:
            self.end_time = time

    def set_terms(self, term_set):
        self.term_list = list(term_set)
        index = 0
        for t in self.term_list:
            self.term_index[t] = index
            index += 1
        self.num_terms = index

    def append_authors(self, a):
        self.author_list.append(a)
        self.author_index[a.naid] = self.num_authors
        self.num_authors += 1

    def append_documents(self, p):
        self.document_list.append(p)
        self.document_list_given_time[p.year].append(p.id)
        self.document_index[p.id] = self.num_documents
        self.num_documents += 1

    def caculate_term_frequence(self):
        #init term frequence
        self.term_freq = np.zeros(self.num_terms)
        self.term_freq_given_time = np.zeros((self.num_time_slides, self.num_terms))
        self.term_freq_given_person = np.zeros((self.num_terms, self.num_authors))
        self.term_freq_given_person_time = [np.zeros((self.num_terms, self.num_authors)) for i in range(self.num_time_slides)]
        for i in range(self.num_time_slides):
            for y in self.time_slides[i]:
                for d in self.document_list_given_time[y]:
                    text = (self.document_list[self.document_index[d]].title.lower() + " . " + self.document_list[self.document_index[d]].abs.lower())
                    for t in range(self.num_terms):
                        if self.term_list[t] in text:
                            self.term_freq[t] += 1
                            self.term_freq_given_time[i, t] += 1
                            for a in self.document_list[self.document_index[d]].author_ids:
                                if self.author_index.has_key(a):
                                    #logging.info("i:%s,y:%s,d:%s,text:%s,t:%s,a:%s"%(i,y,d,text,t,a))
                                    self.term_freq_given_person[t, self.author_index[a]] += 1
                                    self.term_freq_given_person_time[i][t, self.author_index[a]] += 1

    def search_document_by_author(self, a, start_time=0, end_time=10000):
        logging.info("querying documents for %s" % a.names)
        result = data_center.getPublicationsByAuthorId([a.naid])
        logging.info("found %s documents" % len(result.publications))
        #text for extract key terms
        text = ""
        for p in result.publications:
            #update time info
            if p.year > start_time and p.year < end_time:
                self.set_time(p.year)
                text += (p.title.lower() + " . " + p.abs.lower() +" . ")
                #insert document
                self.append_documents(p)
        return text

    def search_author(self, q, time_window):
        self.author_result = data_center.searchAuthors(q)
        term_set = set()
        index = 0
        for a in self.author_result.authors:
            #insert author
            self.append_authors(a)
            #search for document
            text = self.search_document_by_author(a)
            #extract terms
            terms = term_extractor.extractTerms(text)
            for t in terms:
                term_set.add(t)
        self.set_terms(term_set)
        #update time slides
        self.set_time_slides(time_window)
        #caculate term frequence
        self.caculate_term_frequence()
        
    def local_clustering(self, time):
        num_clusters=self.num_local_clusters
        X = self.term_freq_given_person_time[time]
        num_item = len(X)
        logging.info("KMeans... item slides-%s", time)
        kmeans = KMeans(init='k-means++', n_clusters=num_clusters).fit(X)
        logging.info("KMeans finished")
        self.local_clusters[time] = [[] for i in range(self.num_local_clusters)]
        for i, c in enumerate(kmeans.labels_):
            self.local_clusters[time][c].append(i)

    def build_global_feature_vectors(self):
        index = 0
        self.gloabl_feature_vectors_index = [{} for i in range(self.num_time_slides)]
        dim = self.num_authors
        X = np.zeros((self.num_time_slides*self.num_local_clusters, dim))
        for t in range(self.num_time_slides):
            for i, cluster in enumerate(self.local_clusters[t]):
                self.gloabl_feature_vectors_index[t][i] = index
                for w in cluster:
                     X[index] += self.term_freq_given_person_time[t][w]
                index += 1
        return X       

    def global_clustering(self):
        num_clusters=self.num_global_clusters
        #clustering by authors as feature
        #build feature vectors
        X = self.build_global_feature_vectors()
        logging.info("Global KMeans... ")
        kmeans = KMeans(init='k-means++', n_clusters=num_clusters).fit(X)
        logging.info("Global KMeans finished")
        self.global_clusters = [[[] for i in range(num_clusters)] for j in range(self.num_time_slides)]
        labels = kmeans.labels_
        for time in range(self.num_time_slides):
            for i, cluster in enumerate(self.local_clusters[time]):
                l = labels[self.gloabl_feature_vectors_index[time][i]]
                self.global_clusters[time][l].append(i)
                #for w in self.local_clusters[time][c]:
                #    self.global_clusters[l].append(w)

    def build_graph(self):
        self.graph = {"nodes":[], "links":[]}
        global_clusters_index = {}
        index = 0
        for time in range(self.num_time_slides):
            cluster_weight_given_time = np.zeros(self.num_global_clusters)
            document_count = 0.
            for y in self.time_slides[time]:
                document_count += len(self.document_list_given_time[y])
            document_count /= len(self.time_slides[time])
            for i, cluster in enumerate(self.global_clusters[time]):
                for c in cluster:
                    for w in self.local_clusters[time][c]:
                        cluster_weight_given_time[i] += self.term_freq_given_time[time][w]
            cluster_weight_sum_given_time = sum(cluster_weight_given_time)
            for i, cluster in enumerate(self.global_clusters[time]):
                terms = []
                for c in cluster:
                    for w in self.local_clusters[time][c]:
                        terms.append(w)
                if len(terms) == 0:
                    continue
                sorted_terms = sorted(terms, key=lambda t: self.term_freq[t], reverse=True)
                sorted_terms_given_time = sorted(terms, key=lambda t: self.term_freq_given_time[time][t], reverse=True)
                self.graph["nodes"].append({"key":[{"term":self.term_list[k], "w":int(self.term_freq_given_time[time][k])} for k in sorted_terms_given_time], 
                                        "name":self.term_list[sorted_terms_given_time[0]]+"-"+self.term_list[sorttied_terms[0]],
                                        "pos":time, 
                                        "w":cluster_weight_given_time[i]/cluster_weight_sum_given_time*(document_count+1),
                                        "cluster":i})
                global_clusters_index[str(time)+"-"+str(i)] = index
                index += 1
        #caculate similarity
        global_clusters_sim_target = defaultdict(dict)
        global_clusters_sim_source = defaultdict(dict)
        for time in range(1, self.num_time_slides):
            for i1, c1 in enumerate(self.global_clusters[time]):
                key1 = str(time)+"-"+str(i1)
                if global_clusters_index.has_key(key1):
                    terms1 = []
                    for c in c1:
                        for w in self.local_clusters[time][c]:
                            terms1.append(w)
                    for i2, c2 in enumerate(self.global_clusters[time-1]):
                        key2 = str(time-1)+"-"+str(i2)
                        if global_clusters_index.has_key(key2):
                            terms2 = []
                            for c in c2:
                                for w in self.local_clusters[time][c]:
                                    terms2.append(w)
                            sim = jaccard_similarity(terms1, terms2)
                            if sim > 0:
                                global_clusters_sim_target[key1][key2] = sim
                                global_clusters_sim_source[key2][key1] = sim
            #for c in range(self.num_global_clusters):
            #    key1 = str(time)+"-"+str(i1)
            #    key2 = str(time-1)+"-"+str(i2)
            #    if global_clusters_index.has_key(key1) and global_clusters_index.has_key(key2):
            #        global_clusters_sim_target[key1][key2] = 1
            #        global_clusters_sim_source[key2][key1] = 1
        for key1 in global_clusters_sim_target:
            if global_clusters_index.has_key(key1):
                m1 = sum(global_clusters_sim_target[key1].values())
                for key2 in global_clusters_sim_target[key1]:
                    if global_clusters_index.has_key(key2):
                        m2 = sum(global_clusters_sim_source[key2].values())
                        self.graph["links"].append({"source":int(global_clusters_index[key2]),
                                    "target":int(global_clusters_index[key1]),
                                    "w1":global_clusters_sim_target[key1][key2]/float(m1),
                                    "w2":global_clusters_sim_target[key1][key2]/float(m2)})
        self.graph["time_slides"] = self.time_slides
        return self.graph

    def query_terms(self, q, time_window=None, start_time=None, end_time=None):
        #query documents and caculate term frequence
        self.author_list = []
        self.author_index = {}
        self.num_documents = 0
        self.document_list = []
        self.document_list_given_time = defaultdict(list)
        self.document_index = {}
        self.num_documents = 0
        self.term_index = {}
        self.num_terms = 0
        self.search_author(q, time_window, start_time, end_time)
        #local clustering
        self.local_clusters = [None for i in range(self.num_time_slides)]
        for time in range(self.num_time_slides):
            self.local_clustering(time)
        #global clustering
        self.global_clustering()
        graph = self.build_graph()
        return graph

    def query_terms_1(self, q):
        x = data_center.searchAuthors(q)
        pubs = []
        author_name_dict = {}
        document_list_given_time = defaultdict(list)
        term_freq = defaultdict(int)
        term_freq_given_time = defaultdict(lambda: defaultdict(int))
        term_freq_given_person = defaultdict(set)
        term_freq_given_person_time = defaultdict(lambda: defaultdict(set))
        person_index = {}
        documents = {}
        corpus = defaultdict(dict)
        #query authors and documents
        index = 0
        for a in x.authors:
            print "querying documents for", a.names
            result = data_center.getPublicationsByAuthorId([a.naid])
            print "found ", len(result.publications), "documents"
            person_index[a.naid] = index
            index += 1
            text = ""
            for p in result.publications:
                if p.year > 1990:
                    documents[p.id] = p
                    children_ids, parents_ids, authors, author_ids = extractdocument(p)
                    text += (p.title.lower() + " . " + p.abs.lower() +" . ")
                    corpus[p.year][p.id] = text
                    for i in range(len(author_ids)):
                        author_name_dict[author_ids[i]] = authors[i]
                    document_list_given_time[p.year].append(p.id)
            terms = term_extractor.extractTerms(text)
            for t in terms:
                term_freq_given_person[t].add(a.naid)
                
        for y in corpus:
            for d in corpus[y]:
                text = corpus[y][d]
                for t in term_freq_given_person:
                    if t in text:
                        term_freq[t] += 1
                        term_freq_given_time[y][t] += 1
                        for a in documents[d].author_ids:
                            term_freq_given_person_time[y][t].add(a)
        #build feature vectors
        feature_vectors = []
        feature_index = {}
        sorted_terms = sorted(term_freq.items(), key = lambda x: x[1], reverse=True)
        index = 0
        for k in sorted_terms[:100]:
            feature_index[k[0]] = index
            feature = [0.0 for i in range(len(x.authors))]
            for a in term_freq_given_person[k[0]]:
                feature[person_index[a]] = 1
            feature_vectors.append(feature)
            index += 1
        kmeans = KMeans(init='k-means++', n_clusters=10).fit(feature_vectors)
        clusters = defaultdict(list)
        for i in range(len(kmeans.labels_)):
            clusters[kmeans.labels_[i]].append(sorted_terms[i][0])
        cluster_year_weight = defaultdict(lambda :defaultdict(float))
        for y in term_freq_given_time:
            for c in clusters:
                # cluster_year_weight[y][c] = cluster_year_weight[y-1][c]
                for k in clusters[c]:
                    cluster_year_weight[y][c] += term_freq_given_time[y][k]
            z = max(cluster_year_weight[y].values())
            if z == 0:
                z = 1
            for c in clusters:
                cluster_year_weight[y][c] /= z
        person_year_cluster = defaultdict(lambda :defaultdict(set))
        for y in term_freq_given_time:
            for c in clusters:
                for k in clusters[c]:
                    for p in term_freq_given_person_time[y][k]:
                        person_year_cluster[y][c].add(p)

        graph = []
        for c in clusters:
            layer = {"terms":[],"nodes":[]}
            for y in cluster_year_weight:
                layer["nodes"].append({"year":y, "weight":cluster_year_weight[y][c]})
            for t in clusters[c]:
                layer["terms"].append(t)
            graph.append(layer)
        import json
        dump = open("trend.json","w")
        json.dump(graph, dump )
        dump.close()
        return graph

    def render_topic_graph_1(self, nodes):
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

    def query_document(self, q):
        x = data_center.searchdocuments(q)

    def query_community(self, q):
        import community
        x = data_center.searchAuthors(q)
        pubs = []
        author_name_dict = {}
        document_list_given_time = defaultdict(list)
        term_freq = defaultdict(int)
        term_freq_given_time = defaultdict(lambda: defaultdict(int))
        document_terms = {}
        document_authors = {}
        networks = {}
        #query authors and documents
        for a in x.authors:
            result = data_center.getdocumentsByAuthorId([a.naid])
            for p in result.documents:
                if p.year > 1990:
                    children_ids, parents_ids, terms, authors, author_ids = self.extractdocument(p)
                    for i in range(len(author_ids)):
                        author_name_dict[author_ids[i]] = authors[i]
                    document_list_given_time[p.year].append(p.id)
                    for t in terms:
                        term_freq[t.lower()] += 1
                        term_freq_given_time[p.year][t.lower()] += 1
                    document_terms[p.id] = terms
                    document_authors[p.id] = author_ids
        #build coauthor networks
        edge_dict = defaultdict(int)
        for y in document_list_given_time:
            networks[y] = nx.Graph()
            for p in document_list_given_time[y]:
                a = document_authors[p]
                for i in range(len(a)):
                    for j in range(j, len(a)):
                        if a[i] < a[j]:
                            x = (a[i], a[j])
                        else:
                            x = (a[j], a[i])
                        edge_dict[x] += 1
            for x in edge_dict:
                networks[y].add_edge(x[0], x[1], weight=edge_dict[x])
        communities = {}
        for y in networks:
            #better with karate_graph() as defined in networkx example.
            #erdos renyi don't have true community structure
            G = networks[y]
            #first compute the best partition
            partition = community.best_partition(G)
            #drawing
            size = float(len(set(partition.values())))
            communities[y] = partition
        return communities

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

    """
    util: extracting document result
    """
    def extractdocument(self, p):
        #extract key terms from abstract and title
        #text = p.title.lower() + " . " + p.abs.lower()
        #terms = term_extractor.extractTerms(text)
        #extract citation network
        children = []
        children_ids = []
        parents = []
        parents_ids = []
        authors = []
        authors_ids = []
        for x in p.cited_by_pubs:
            children.append(str(x))
            children_ids.append(x)
        for x in p.cite_pubs:
            parents.append(str(x))
            parents_ids.append(x)
        for x in p.authors:
            authors.append(x)
        return children_ids, parents_ids, authors, authors_ids

    """
    util: extracting person result
    """
    def extractPerson(self, p):
        #extract key terms from abstract and title
        name = p.title.lower() + " . " + p.abs.lower()
        terms = term_extractor.extractTerms(text)
        #extract citation network
        children = []
        children_ids = []
        parents = []
        parents_ids = []
        authors = []
        authors_ids = []
        for x in p.cited_by_pubs:
            children.append(str(x))
            children_ids.append(x)
        for x in p.cite_pubs:
            parents.append(str(x))
            parents_ids.append(x)
        for x in p.authors:
            authors.append(x)
        return children_ids, parents_ids, terms, authors, authors_ids

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
    def render_topic_graph_or(self, nodes):
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


def query_community(self, q):
    import community
    x = data_center.searchAuthors(q)
    pubs = []
    author_name_dict = {}
    document_list_given_time = defaultdict(list)
    term_freq = defaultdict(int)
    term_freq_given_time = defaultdict(lambda: defaultdict(int))
    term_freq_given_person = defaultdict(set)
    person_index = {}
    document_terms = {}
    document_authors = {}
    networks = {}
    #query authors and documents
    index = 0
    for a in x.authors:
        print "querying documents for", a.names
        result = data_center.getdocumentsByAuthorId([a.naid])
        print "found ", len(result.documents), "documents"
        person_index[a.naid] = index
        for p in result.documents:
            if p.year > 1990:
                children_ids, parents_ids, terms, authors, author_ids = extractdocument(p)
                for i in range(len(author_ids)):
                    author_name_dict[author_ids[i]] = authors[i]
                document_list_given_time[p.year].append(p.id)
                for t in terms:
                    term_freq[t.lower()] += 1
                    term_freq_given_time[p.year][t.lower()] += 1
                    term_freq_given_person[t.lower()].add(a.naid)
                document_terms[p.id] = terms
                document_authors[p.id] = author_ids
        index += 1
    #build feature vectors
    feature_vectors = []
    feature_index = {}
    sorted_terms = sorted(term_freq.items(), key = lambda x: x[1], reverse=True)
    index = 0
    for k in sorted_terms[:100]:
        feature_index[k[0]] = index
        feature = [0.0 for i in range(len(x.authors))]
        for a in term_freq_given_person[k[0]]:
            feature[person_index[a]] = 1
        feature_vectors.append(feature)
        index += 1
    kmeans = KMeans(init='k-means++', n_clusters=10).fit(feature_vectors)
    clusters = defaultdict(list)
    for i in range(len(kmeans.labels_)):
        clusters[kmeans.labels_[i]].append(sorted_terms[i][0])
    cluster_year_weight = defaultdict(lambda :defaultdict(float))
    for y in term_freq_given_time:
        for c in clusters:
            # cluster_year_weight[y][c] = cluster_year_weight[y-1][c]
            for k in clusters[c]:
                cluster_year_weight[y][c] += term_freq_given_time[y][k]
        z = max(cluster_year_weight[y].values())
        if z == 0:
            z = 1
        for c in clusters:
            cluster_year_weight[y][c] /= z
    graph = []
    for c in clusters:
        layer = {"terms":[],"nodes":[]}
        for y in cluster_year_weight:
            layer["nodes"].append({"year":y, "weight":cluster_year_weight[y][c]})
        for t in clusters[c]:
            layer["terms"].append(t)
        graph.append(layer)
    import json
    dump = open("trend.json","w")
    json.dump(graph, dump )
    dump.close()
        


    #build coauthor networks

def main():
    trend = TopicTrend()
    trend.query_terms("deep learning")


if __name__ == "__main__":
    main()