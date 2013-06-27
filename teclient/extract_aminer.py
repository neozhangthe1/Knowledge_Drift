from teclient import TermExtractorClient
import pymongo

con = pymongo.Connection("10.1.1.110", 12345)
col = con["terms"]

pub_f = open("publication.txt")
abs_f = open("publication_ext.txt")
papers_title = {}
papers_abs = {}
for line in pub_f:
    x = line.strip("\n").split("\t")
    papers_title[x[0]] = x[1]

for line in abs_f:
    x = line.strip("\n").split("\t")
    papers_abs[x[0]] = x[1]

def dumpTerms(term_list):
    dump = open("terms.txt","w")
    for id in term_list:
        term_str = ""
        for term in term_list[id]:
            term_str+=term+":"+str(term_list[id][term])+"\t"
        dump.write("%s*%s\n"%(id,term_str))
    dump.close()

ext = TermExtractorClient()
terms = {}
titles = papers_title
abs = papers_abs
idx = 0
for id in titles:
    idx += 1
    if idx % 1000 == 0:
        print idx
    if idx < 1468000:
        continue
    text = titles[id]
    if abs.has_key(id):
        text += "\n"
        text += abs[id]
    if len(text) > 5:
        terms[id] = ext.extractTerms(text)
dumpTerms(terms)

    #item = {"_id":id}
    #item[
import pickle
title_dump = open("title_dump.pickle","w")
abs_dump = open("abs_dump.pickle", "w")
terms_dump = open("terms_dump.pickle", "w")
pickle.dump(terms, terms_dump)
pickle.dump(abs, abs_dump)
pickle.dump(titles, title_dump)
title_dump.close()
abs_dump.close()

terms_dump = open("terms_dump_all.pickle", "w")
pickle.dump(terms, terms_dump)
terms_dump.close()
    
terms = ext.extractTermsFromTitleAbs(papers_title, papers_abs)

ext.dumpTerms(terms)

