from teclient import TermExtractorClient
import pickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
extractor = TermExtractorClient()

def dumpTerms(term_list):
    with open("terms_dump.pickle", "wb") as terms_dump:
        pickle.dump(papers_abs, abs_dump)

def extract_aminer():
    terms = {}
    papers_title = {}
    papers_abs = {}
    idx = -1
    #read paper title
    with open("/home/yutao/Data/knowledge-drift/pub.txt") as pub_f:
        logging.info("reading paper title")
        for line in pub_f:
            x = line.strip("\n").split("\t")
            papers_title[x[0]] = x[1]
    #read paper abstracts
    with open("/home/yutao/Data/knowledge-drift/pub_ext.txt") as abs_f:    
        logging.info("reading paper abstract")
        for line in abs_f:
            x = line.strip("\n").split("\t")
            papers_abs[x[0]] = x[1]
    #concatenate titles and abstracts
    for i in papers_title:
        idx += 1
        if idx % 1000 == 0:
            logging.info("index: %s" % idx)
        text = papers_title[i]
        if papers_abs.has_key(i):
            text += "\n"
            text += papers_abs[i]
        if len(text) > 5:
            terms[i] = extractor.extractTerms(text)
    return papers_title, papers_abs, terms

if __name__ == "__main__":
    terms, papers_title, paper_abs = extract_aminer()
