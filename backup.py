
class CommunityTrend(object):
    def __init__(self):
        pass

    def queryCommunity(self, q):
        pass

    def extractPublication(self, p):
        #extract key terms from abstract and title
        text = p.title.lower() + " . " + p.abs.lower()
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

    def getPersonByQuery(self, q):
        from community import Newman
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
                    year_publication[y].append(p.id)
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
        for y in networks:
            nx.connected_components(network[y])
        #build graph
        graph = {}
        graph["nodes"] = []
        graph["links"] = []
        start_year = min(year_publication.keys())
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


    def getTopic(self, query):
        x =  data_center.searchPublications(query)
        key_term_set = set()
        terms_frequence = defaultdict(int)
        year_terms_frequence = defaultdict(lambda: defaultdict(int))
        #extract key terms and citation network from publications
        for p in x.publications:
            if p.year <= 1980:
                continue
            children_ids, parents_ids, terms, authors, authors_ids = self.extractPublication(p)
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
        
class TopicTrend_new(object):
    def __init__(self):
        print "INIT TOPIC TREND"

    def load_topic_model(self):
        models = {}
        for y in range(1981, 2010):
            models[y] = gensim.models.ldamodel.LdaModel.load("lda-model-streamming-30-"+str(y))
        for y in models:
            models[y].show_topics(30)
        kmeans = pickle.load(open("kmeans"))
        self.models = models
        self.kmeans = kmeans

    def build_graph(self):
        labels = defaultdict(lambda :defaultdict(list))
        nodes = defaultdict(dict)
        graph = {"nodes":[], "links":[]}
        for y in models:
            for i in range(models[y].num_topics):
                labels[y][kmeans.labels_[item_dict[y][i]]].append(i)
        idx = 0
        for y in labels:
            for n in labels[y]:
                label = ""
                for i in labels[y][n]:
                    label+=models[y].print_topic(i)
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

    def cluster_topic_model(self):
        item_dict = defaultdict(dict)
        X = []
        index = 0
        for y in models:
            topics = models[y].state.sstats
            for i in range(len(topics)):
                X.append(topics[i])
                item_dict[y][i] = index
                index += 1
        kmeans = KMeans(init='k-means++', n_clusters=30).fit(X)
        dumps = open("kmeans", "w")
        import pickle
        pickle.dump(kmeans, dumps)
        dumps.close()

def build_topic_model():
    import logging, gensim, bz2
    import re
    from collections import defaultdict
    from sklearn.feature_extraction.text import CountVectorizer
    import codecs
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    f_out = codecs.open("topic_data", "w", "utf-8")
    #stoplist = 
    f = open("d:\\share\\abstracts.txt")
    flag = 0
    titles = []
    abstracts = []
    confs = []
    ids = []
    documents = []
    years = []
    authors = []
    content = None
    for line in f:
        if flag == 0:
            ids.append(int(line.strip()))
            flag += 1
        elif flag == 1:
            years.append(line.strip())
            flag += 1
        elif flag == 2:
            confs.append(line.strip())
            flag += 1
        elif flag == 3:
            authors.append(line.strip())
            flag += 1
        elif flag == 4:
            titles.append(line.strip())
            content = line
            flag += 1        
        elif flag == 5:
            abstracts.append(line.strip())
            content += line
            content.replace("\n"," ").lower()
            documents.append(content)
            flag = 0
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,3), max_df=0.95, min_df=10)#gram_range=(1,3)#gram_range=(1,3)
    X = vectorizer.fit_transform(documents)
    X = X.tolil();
    vocab = vectorizer.get_feature_names()
    for i in range(len(ids)):
        line = str(ids[i])+";"
        nz = X.getrow(i).nonzero()[1]
        line += "#".join(vocab[t] for t in nz)
        line += ";"
        x = authors[i].split(",")
        a = "#".join(x)
        line += a;
        line += ";"
        line += str(years[i])
        line += "0000\n"
        f_out.write(line)

def preprocess_topic_model():
    import logging, gensim, bz2
    import re
    from collections import defaultdict
    from sklearn.feature_extraction.text import CountVectorizer
    import codecs
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    f_out = codecs.open("topic_data", "w", "utf-8")
    #stoplist = 
    f = open("d:\\share\\abstracts.txt")
    flag = 0
    titles = []
    abstracts = []
    confs = []
    ids = []
    documents = []
    years = []
    authors = []
    content = None
    for line in f:
        if flag == 0:
            ids.append(int(line.strip()))
            flag += 1
        elif flag == 1:
            years.append(line.strip())
            flag += 1
        elif flag == 2:
            confs.append(line.strip())
            flag += 1
        elif flag == 3:
            authors.append(line.strip())
            flag += 1
        elif flag == 4:
            titles.append(line.strip())
            content = line
            flag += 1        
        elif flag == 5:
            abstracts.append(line.strip())
            content += line
            content.replace("\n"," ").lower()
            documents.append(content)
            flag = 0
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,3), max_df=0.95, min_df=10)#gram_range=(1,3)#gram_range=(1,3)
    X = vectorizer.fit_transform(documents)
    X = X.tolil();
    vocab = vectorizer.get_feature_names()
    for i in range(len(ids)):
        line = str(ids[i])+";"
        nz = X.getrow(i).nonzero()[1]
        line += "#".join(vocab[t] for t in nz)
        line += ";"
        x = authors[i].split(",")
        a = "#".join(x)
        line += a;
        line += ";"
        line += str(years[i])
        line += "0000\n"
        f_out.write(line)






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
    lda = None
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,3), max_df=0.95, min_df=3)
    titles = []
    abstracts = []
    confs = []
    ids = []
    documents = []
    years = []
    for y in range(1981, 2010):
        f = open("pubs\\"+str(y))
        flag = 0
        content = None
        for line in f:
            if flag == 0:
                years.append(y)
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
    X = vectorizer.fit_transform(documents)
    X = X.tolil();
    doc_dict = defaultdict(lambda : defaultdict(list))
    doc_ids = X.nonzero()[0]
    word_ids = X.nonzero()[1]
    for i in range(len(doc_ids)):
        doc_dict[years[doc_ids[i]]][doc_ids[i]].append((word_ids[i], 1))#X[doc_ids[i], word_ids[i]]))    
    id2word = {}
    for w in vectorizer.vocabulary_:
        id2word[vectorizer.vocabulary_[w]] = w
    #online update topic model (streamming)
    lda = None
    for y in doc_dict:
        if lda is None:
            lda = gensim.models.ldamodel.LdaModel(corpus=doc_dict[y].values(), id2word=id2word, 
                                        num_topics=30, update_every=1, chunksize=1, passes=1)   
        else:
            lda.update(corpus=doc_dict[y].values())
        lda.save("lda-model-streamming-30-"+str(y))
        
    #hdp = gensim.models.hdpmodel.HdpModel(corpus=doc_dict.values(), id2word=id2word, chunksize=1)


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

def cluster_topics():
    import gensim
    from sklearn.cluster import Ward, KMeans
    models = {}
    for y in range(1981, 2010):
        models[y] = gensim.models.ldamodel.LdaModel.load("lda-model-streamming-30-"+str(y))
    for y in models:
        models[y].show_topics(30)
    kmeans = KMeans(init='k-means++', n_clusters=30).fit(X)