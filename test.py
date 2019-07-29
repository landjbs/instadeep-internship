from questionAnswering.questionAnalysis import build_question_database
PATH = 'data/inData/natural_questions/v1.0/train'
x = build_question_database(PATH, n=2500, outPath='data/outData/paragraphFindingDf.sav')

# import re
# import numpy as np
# from scipy.spatial.distance import euclidean
# from tqdm import tqdm
# from keras.models import load_model
#
# from vectorizers.docVecs import vectorize_doc
#
# text = """ Facebook, Inc. is an American online social media and social networking service company based in Menlo Park, California. It was founded by Mark Zuckerberg, along with fellow Harvard College students and roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. It is considered one of the Big Four technology companies along with Amazon, Apple, and Google.[10][11]
# The founders initially limited the website's membership to Harvard students and subsequently Columbia, Stanford, and Yale students. Membership was eventually expanded to the remaining Ivy League schools, MIT, and higher education institutions in the Boston area, then various other universities, and lastly high school students. Since 2006, anyone who claims to be at least 13 years old has been allowed to become a registered user of Facebook, though this may vary depending on local laws. The name comes from the face book directories often given to American university students. Facebook held its initial public offering (IPO) in February 2012, valuing the company at $104 billion, the largest valuation to date for a newly listed public company. Facebook makes most of its revenue from advertisements that appear onscreen and in users' News Feeds.
# The Facebook service can be accessed from devices with Internet connectivity, such as personal computers, tablets and smartphones. After registering, users can create a customized profile revealing information about themselves. They can post text, photos and multimedia which is shared with any other users that have agreed to be their "friend". Users can also use various embedded apps, join common-interest groups, and receive notifications of their friends' activities. Facebook claimed that had more than 2.3 billion monthly active users as of December 2018.[12] However, it faces a big problem of fake accounts. It caught 3 billion fake accounts, but the ones it misses are the real problem.[13] Many critics questioned whether Facebook knows how many actual users it has.[14][15][13] Facebook is one of the world's most valuable companies.
# It receives prominent media coverage, including many controversies. These often involve user privacy (as with the Cambridge Analytica data scandal), political manipulation (as with the 2016 U.S. elections), psychological effects such as addiction and low self-esteem, and content that some users find objectionable, including fake news, conspiracy theories, and copyright infringement.[16] Facebook also does not remove false information from its pages, which brings continuous controversies.[17] Commentators have stated that Facebook helps to spread false information and fake news.[18][19][20][21]
# Facebook offers other products and services. It acquired Instagram, WhatsApp, Oculus, and GrokStyle[22] and independently developed Facebook Messenger, Facebook Watch, and Facebook Portal. """
#
#
# cleanedText = re.sub(r'\n', '', text)
# sentences = re.split(r'[.?!]', cleanedText)
#
# sentences = list(filter(lambda sent:sent!=" ", sentences))
#
# encodedSentences = {sentence:vectorize_doc(sentence)
#                     for sentence in tqdm(sentences)}
#
# paraModel = load_model('data/outData/models/paragraphAnswering.sav')
#
# while True:
#     search = input("Search: ")
#     searchVec = vectorize_doc(search)
#     calc_score = lambda sentVec : paraModel.predict(np.expand_dims(np.subtract(searchVec, sentVec), axis=0))
#     rankedList = [(calc_score(sentVec), sent)
#                     for sent, sentVec in encodedSentences.items()]
#     rankedList.sort(reverse=True)
#     print(rankedList[0][1])
