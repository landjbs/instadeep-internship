# from questionAnswering.answer import  answer_question

# while True:
#     text = input('Text: ')
#     question = input("Question: ")
#     print(answer_question(question, text))


from documentRetrieval.questionReading import read_question_dataset
PATH = 'data/inData/natural_questions/v1.0/train'
x = read_question_dataset(PATH, n=2500, outPath='data/outData/questionsDf2500.sav')


# from documentRetrieval.modelTraining import train_retrieval_model
# retrievalModel = train_retrieval_model(inPath='data/outData/questionsDf2500.sav',
#                                         numWrong=3,
#                                         outPath='data/outData/models/documentRetrievalModel2500.sav')
