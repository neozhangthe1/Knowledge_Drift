from dcclient.dcclient import DataCenterClient
from teclient.teclient import TermExtractorClient
from collections import defaultdict
from bs4 import UnicodeDammit
import numpy as np
import json     
import gensim
import pickle
import networkx as nx
from sklearn.cluster import Ward, KMeans, MiniBatchKMeans
data_center = DataCenterClient("tcp://10.1.1.211:32011")
term_extractor = TermExtractorClient()
def extractPublication(p):
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
        self.parse_topic_model()
        self.parse_topic_graph()

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

    def query_terms(self, q):
        x = data_center.searchAuthors(q)
        pubs = []
        author_name_dict = {}
        year_publication = defaultdict(list)
        key_terms = defaultdict(int)
        year_terms = defaultdict(lambda: defaultdict(int))
        term_person = defaultdict(set)
        person_index = {}
        document_terms = {}
        document_authors = {}
        term_index = {}
        #query authors and publications
        index = 0
        for a in x.authors:
            print "querying publications for", a.names
            result = data_center.getPublicationsByAuthorId([a.naid])
            print "found ", len(result.publications), "publications"
            person_index[a.naid] = index
            corpus = ""
            for p in result.publications:
                if p.year > 1990:
                    children_ids, parents_ids, authors, author_ids = extractPublication(p)
                    corpus += (p.title.lower() + " . " + p.abs.lower() +" . ")
                    for i in range(len(author_ids)):
                        author_name_dict[author_ids[i]] = authors[i]
                    year_publication[p.year].append(p.id)
                    #for t in terms:
                    #    key_terms[t.lower()] += 1
                    #    year_terms[p.year][t.lower()] += 1
                    #    term_person[t.lower()].add(a.naid)
                    #document_terms[p.id] = terms
                    #document_authors[p.id] = author_ids
            terms = term_extractor.extractTerms(corpus)
            for t in terms:
                term_person[t].add(a.naid)
            for p in result.publications:
                if p.year > 1990:
                    for t in terms:
                        key_terms[t.lower()] += 1
                        year_terms[p.year][t.lower()] += 1
                        term_person[t.lower()].add(a.naid)
                    document_terms[p.id] = terms
                    document_authors[p.id] = author_ids
            index += 1
        index = 0 
        for t in key_terms:
            term_index[t] = index
            index += 1
        #build feature vectors
        year_cluster_label = defaultdict(lambda :defaultdict(float))
        cluster_feature_vectors = []
        cluster_feature_index = {}
        cluster_feature = defaultdict(lambda :np.array([0.0 for i in range(len(x.authors))]))
        clusters = defaultdict(list)
        yindex = 0
        for y in year_terms:
            feature_vectors = []
            feature_index = {}
            feature_term = []
            index = 0
            for k in key_terms:
                if year_terms[y][k] > 0:
                    feature_index[k] = index
                    feature = np.array([0.0 for i in range(len(x.authors))])
                    for a in term_person[k]:
                        feature[person_index[a]] = 1
                    feature_vectors.append(feature)
                    feature_term.append(k)
                    index += 1
                    if index == 100:
                        break
            labels = None
            if len(feature_vectors) > 10:
                print "KMeans... ", y
                kmeans = KMeans(init='k-means++', n_clusters=5).fit(feature_vectors)
                print "KMeans finished"
                labels = kmeans.labels_
            else:
                labels = [0 for i in range(len(feature_vectors))]
            for i in range(len(labels)):
                l = labels[i]
                clusters[str(y)+"-"+str(l)].append(feature_term[i])
                cluster_feature[l] += feature
            for i in cluster_feature:
                cluster_feature_vectors.append(cluster_feature[i])
                cluster_feature_index[str(y)+"-"+str(labels[i])] = yindex
                yindex += 1 
            year_cluster_label[y] = clusters
        print "Global KMeans... "
        kmeans = KMeans(init='k-means++', n_clusters=5).fit(cluster_feature_vectors)
        print "Global KMeans finished"
        graph = {"nodes":[], "links":[]}
        index = 0
        #merge nodes
        year_global_clusters = defaultdict(lambda : defaultdict(list))
        year_global_clusters_terms = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
        year_global_clusters_sim_source = defaultdict(lambda : defaultdict(float))
        year_global_clusters_sim_target = defaultdict(lambda : defaultdict(float))
        year_global_clusters_index = {}
        #year_global_clusters_weight = defaultdict(dict)
        for key in cluster_feature_index:
            xx = key.split("-")
            y = int(xx[0])
            c = int(xx[1])
            l = kmeans.labels_[cluster_feature_index[key]]
            year_global_clusters[y][l].append(c)
            for k in clusters[key]:
                year_global_clusters_terms[y][l][k] += year_terms[y][k]
            #year_global_clusters_weight[y][l] = sum
        for y in year_terms:
            for c in year_global_clusters[y]:
                graph["nodes"].append({"key":[{"term":k, "w":int(year_terms[y][k])} 
                                              for k in year_global_clusters_terms[y][c]], 
                                       "name":sorted(year_global_clusters_terms[y][c], key=lambda e: year_terms[y][e], reverse=True)[0],
                                        "pos":y-min(year_terms.keys()), 
                                        "w":sum(year_global_clusters_terms[y][l].values()),
                                        "cluster":int(c)})
                year_global_clusters_index[str(y)+"-"+str(c)] = index
                index += 1
        for y in year_terms:
            for c1 in year_global_clusters[y]:
                for c2 in year_global_clusters[y-1]:
                    sim = len(set(year_global_clusters_terms[y][c1].keys()) 
                              - set(year_global_clusters_terms[y-1][c2].keys()))
                    if sim > 0:
                        year_global_clusters_sim_target[str(y)+"-"+str(c1)][str(y-1)+"-"+str(c2)] = sim
                        year_global_clusters_sim_source[str(y-1)+"-"+str(c2)][str(y)+"-"+str(c1)] = sim
        for key1 in year_global_clusters_sim_target:
            m1 = sum(year_global_clusters_sim_target[key1].values())
            for key2 in year_global_clusters_sim_target[key1]:
                m2 = sum(year_global_clusters_sim_source[key2].values())
                graph["links"].append({"source":int(year_global_clusters_index[key1]),
                            "target":int(year_global_clusters_index[key2]),
                            "w1":year_global_clusters_sim_target[key1][key2]/float(m1),
                            "w2":year_global_clusters_sim_target[key1][key2]/float(m2)})

        #for y in year_terms:
        #    for c in clusters:
        #        # cluster_year_weight[y][c] = cluster_year_weight[y-1][c]
        #        for k in clusters[c]:
        #            cluster_year_weight[y][c] += year_terms[y][k]
        #    z = max(cluster_year_weight[y].values())
        #    if z == 0:
        #        z = 1
        #    for c in clusters:
        #        cluster_year_weight[y][c] /= z
        #node_dict = {}
        #index = 0
        #for c in clusters:
        #    pre = None
        #    for y in cluster_year_weight:
        #        graph["nodes"].append({"name":",".join(clusters[c]), 
        #                               "pos":y-min(cluster_year_weight.keys()), 
        #                               "w":cluster_year_weight[y][c],
        #                               "cluster":int(c)})
        #        node_dict[str(y)+"-"+str(c)] = index
        #        if pre is not None:
        #            graph["links"].append({"source":pre,
        #                                    "target":index,
        #                                    "w1":1,
        #                                    "w2":1})
        #        pre = index
        #        index += 1
        return graph

    def query_terms_1(self, q):
        x = data_center.searchAuthors(q)
        pubs = []
        author_name_dict = {}
        year_publication = defaultdict(list)
        key_terms = defaultdict(int)
        year_terms = defaultdict(lambda: defaultdict(int))
        term_person = defaultdict(set)
        person_index = {}
        corpus = defaultdict(dict)
        #query authors and publications
        index = 0
        for a in x.authors:
            print "querying publications for", a.names
            result = data_center.getPublicationsByAuthorId([a.naid])
            print "found ", len(result.publications), "publications"
            person_index[a.naid] = index
            index += 1
            text = ""
            for p in result.publications:
                if p.year > 1990:
                    children_ids, parents_ids, authors, author_ids = extractPublication(p)
                    text += (p.title.lower() + " . " + p.abs.lower() +" . ")
                    corpus[p.year][p.id] = text
                    for i in range(len(author_ids)):
                        author_name_dict[author_ids[i]] = authors[i]
                    year_publication[p.year].append(p.id)
            terms = term_extractor.extractTerms(text)
            for t in terms:
                term_person[t].add(a.naid)
        for y in corpus:
            for text in corpus[y].values():
                for t in term_person:
                    if t in text:
                        key_terms[t] += 1
                        year_terms[y][t] += 1
        #build feature vectors
        feature_vectors = []
        feature_index = {}
        sorted_terms = sorted(key_terms.items(), key = lambda x: x[1], reverse=True)
        index = 0
        for k in sorted_terms:
            feature_index[k[0]] = index
            feature = [0.0 for i in range(len(x.authors))]
            for a in term_person[k[0]]:
                feature[person_index[a]] = 1
            feature_vectors.append(feature)
            index += 1
        kmeans = KMeans(init='k-means++', n_clusters=10).fit(feature_vectors)
        clusters = defaultdict(list)
        for i in range(len(kmeans.labels_)):
            clusters[kmeans.labels_[i]].append(sorted_terms[i][0])
        cluster_year_weight = defaultdict(lambda :defaultdict(float))
        for y in year_terms:
            for c in clusters:
                # cluster_year_weight[y][c] = cluster_year_weight[y-1][c]
                for k in clusters[c]:
                    cluster_year_weight[y][c] += year_terms[y][k]
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
        x = data_center.searchPublications(q)

    def query_community(self, q):
        import community
        x = data_center.searchAuthors(q)
        pubs = []
        author_name_dict = {}
        year_publication = defaultdict(list)
        key_terms = defaultdict(int)
        year_terms = defaultdict(lambda: defaultdict(int))
        document_terms = {}
        document_authors = {}
        networks = {}
        #query authors and publications
        for a in x.authors:
            result = data_center.getPublicationsByAuthorId([a.naid])
            for p in result.publications:
                if p.year > 1990:
                    children_ids, parents_ids, terms, authors, author_ids = self.extractPublication(p)
                    for i in range(len(author_ids)):
                        author_name_dict[author_ids[i]] = authors[i]
                    year_publication[p.year].append(p.id)
                    for t in terms:
                        key_terms[t.lower()] += 1
                        year_terms[p.year][t.lower()] += 1
                    document_terms[p.id] = terms
                    document_authors[p.id] = author_ids
        #build coauthor networks
        edge_dict = defaultdict(int)
        for y in year_publication:
            networks[y] = nx.Graph()
            for p in year_publication[y]:
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
    util: extracting publication result
    """
    def extractPublication(self, p):
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
    year_publication = defaultdict(list)
    key_terms = defaultdict(int)
    year_terms = defaultdict(lambda: defaultdict(int))
    term_person = defaultdict(set)
    person_index = {}
    document_terms = {}
    document_authors = {}
    networks = {}
    #query authors and publications
    index = 0
    for a in x.authors:
        print "querying publications for", a.names
        result = data_center.getPublicationsByAuthorId([a.naid])
        print "found ", len(result.publications), "publications"
        person_index[a.naid] = index
        for p in result.publications:
            if p.year > 1990:
                children_ids, parents_ids, terms, authors, author_ids = extractPublication(p)
                for i in range(len(author_ids)):
                    author_name_dict[author_ids[i]] = authors[i]
                year_publication[p.year].append(p.id)
                for t in terms:
                    key_terms[t.lower()] += 1
                    year_terms[p.year][t.lower()] += 1
                    term_person[t.lower()].add(a.naid)
                document_terms[p.id] = terms
                document_authors[p.id] = author_ids
        index += 1
    #build feature vectors
    feature_vectors = []
    feature_index = {}
    sorted_terms = sorted(key_terms.items(), key = lambda x: x[1], reverse=True)
    index = 0
    for k in sorted_terms[:100]:
        feature_index[k[0]] = index
        feature = [0.0 for i in range(len(x.authors))]
        for a in term_person[k[0]]:
            feature[person_index[a]] = 1
        feature_vectors.append(feature)
        index += 1
    kmeans = KMeans(init='k-means++', n_clusters=10).fit(feature_vectors)
    clusters = defaultdict(list)
    for i in range(len(kmeans.labels_)):
        clusters[kmeans.labels_[i]].append(sorted_terms[i][0])
    cluster_year_weight = defaultdict(lambda :defaultdict(float))
    for y in year_terms:
        for c in clusters:
            # cluster_year_weight[y][c] = cluster_year_weight[y-1][c]
            for k in clusters[c]:
                cluster_year_weight[y][c] += year_terms[y][k]
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
    pass

if __name__ == "__main__":
    main()