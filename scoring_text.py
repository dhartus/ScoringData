import json
import math
from queue import PriorityQueue
from collections import Counter
from collections import deque

#  PreProcess Data - calculating IDF for all words in all reviews
idf = dict()  # Stores the idf of each word
reviews_with_word = dict()  # Stores the count of word in all the reviews. This is req for tf calculation
totalreviews = 0
total_words_inallreviews = 0
with open("review.json", "r") as json_file:
    totalreviews = sum(1 for line in json_file)

with open("review.json", "r") as json_file:
    for line in json_file:
        cur_review = json.loads(line)
        total_words_inallreviews += len(list(cur_review["review"].split()))
        unique_words = set(cur_review["review"].split())
        for word in unique_words:
            reviews_with_word[word] = reviews_with_word.get(word, 0) + 1

for word in reviews_with_word:
    idf[word] = math.log(totalreviews/reviews_with_word.get(word))

averagereviewlength = total_words_inallreviews/totalreviews


def calculate(listofwords, numberofreviews_to_display, scoring):

    # For each review in json file , calculate tf + idf score and store in queue
    q = PriorityQueue()
    with open("review.json", "r") as json_file:
        for line in json_file:
            cur_review = json.loads(line)

            # Calculate tf idf score for each word in list
            tfidf_score = 0  # Initialise tfidf score for current review
            docmagnitude = 0  # For cosine scoring , docmagnitude is sum of (tfidf of all words in current review) ** 2
            doc_vector = list()  # For cosine scoring , list of tfidf of all query words, in the current review
            query_vector = list()  # For cosine scoring , list of count of words in the query. e.g best best bbq -> [2 1]
            bm25score = 0
            length_of_current_review = 0
            wordsincur_review = list(cur_review["review"].split())
            if scoring == "bm25":
                length_of_current_review = len(wordsincur_review)
            if scoring == "cosine":
                for word in wordsincur_review:
                    tf = wordsincur_review.count(word)
                    tfidf = tf * idf[word]
                    docmagnitude += tfidf ** 2

            for word in listofwords:
                tf = wordsincur_review.count(word)
                tfidf = tf * idf[word]  # tfidf for current word in current review
                tfidf_score += tfidf    # tfidf score for all words in current review
                if scoring == "cosine":
                    query_vector.append(listofwords.count(word))
                    doc_vector.append(tfidf)
                if scoring == "bm25":
                    bm25score += getbm25(tf, idf[word], length_of_current_review)

            score = 0
            if scoring == "tfidf":
                score = tfidf_score
            elif scoring == "cosine":
                score = getcosine(doc_vector, query_vector, docmagnitude)
            elif scoring == "bm25":
                score = bm25score

            # Save top reviews in a queue
            if q.qsize() < numberofreviews_to_display:
                q.put((score, cur_review["review"]))

            else:
                minscore = q.get()
                if tfidf_score > minscore[0]:
                    q.put((score, cur_review["review"]))
                else:
                    q.put(minscore)
    return q


def getbm25(tf, idf, length_of_current_review):
    """
    k = 1.2
    b = 0.75
    bm25 = idf * tf * (1+k) / tf + (k * ( 1 - b + (b * (|D| / avgdl) ) ) )
    :param tf: tf score
    :param idf: idf score
    :param length_of_current_review: length
    :return:bm25
    """
    k = 1.2
    b = 0.75
    bm25 = (idf * (tf * (1+k))) / (tf + (k * (1 - b + (b * (length_of_current_review / averagereviewlength)))))
    return bm25


def getcosine(doc_vector, query_vector, docmagnitude):
    """
    Cosine(query, document) = DotProduct(query, document) / || query || * || doc ||
    :param doc_vector: For cosine scoring , list of tfidf of all query words, in the current review
    :param query_vector: For cosine scoring , list of count of words in the query. e.g best best bbq -> [2 1]
    :param docmagnitude: For cosine scoring , docmagnitude is sum of (tfidf of all words in current review) ** 2
    :return: Cosine value
    """
    if len(doc_vector) == 0 or len(query_vector) == 0:
        return 0

    dot_product = 0
    for i in range(len(doc_vector)):
        dot_product += doc_vector[i] * query_vector[i]

    mag_query = math.sqrt(sum(num**2 for num in query_vector))
    mag_doc = math.sqrt(docmagnitude)
    cos = dot_product / (mag_query * mag_doc)
    return cos


def score_documents(testquery: str, scoring: str, num_of_reviews_to_display: int):
    listofwords = testquery.split()
    top_reviews = calculate(listofwords, num_of_reviews_to_display, scoring)
    stack = deque()
    while not top_reviews.empty():
        stack.append(top_reviews.get())
    while stack:
        score, review = stack.pop()
        print(score)
        print(review)
        print("\n")

if __name__ == "__main__":
    # score_documents("best bbq", "tfidf", 10)
    # score_documents("kid fun and food", "tfidf", 10)

    # score_documents("best bbq", "cosine", 10)
    # score_documents("kid fun and food", "cosine", 10)

    # score_documents("best bbq", "bm25", 10)
    score_documents("kid fun and food", "bm25", 10)