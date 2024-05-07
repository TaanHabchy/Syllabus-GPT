import os
from transformers import pipeline
import PyPDF2

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def basics(text):

    subject = "(Philosophy, Calculus, Literary Tradition): What is the class subject?"
    teacher = "(prompt: Dr, Prof): Who is the teacher or proffessor?"
    location = "Where is the classroom located?"
    schedule = "(prompt: date): What is the class schedule?"

    question_answerer = pipeline(
        "question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

    mySubject = question_answerer(question=subject, context=text)
    myTeacher = question_answerer(question=teacher, context=text)
    myLocation = question_answerer(question=location, context=text)
    mySchedule = question_answerer(question=schedule, context=text)

    print(f"Subject: {mySubject['answer']} - {mySubject['score']}")
    print(f"Teacher: {myTeacher['answer']} - {myTeacher['score']}")
    print(f"Location: {myLocation['answer']} - {myLocation['score']}")
    print(f"Schedule: {mySchedule['answer']} - {mySchedule['score']}")


def make_schedule(text):

    assingments = "List of assingments"

    question_answerer = pipeline(
        "question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

    myAssingments = question_answerer(question=assingments, context=text)

    print(f"Assingments are: {myAssingments['answer']}")


text = ""
with open("syllabi/STS.pdf", 'r') as file:
    if 'txt' in file.name:
        text = file.read()
    elif 'pdf' in file.name:
        with open(file.name, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for i in range(len(reader.pages)):
                page = reader._get_page(i)
                text += page.extract_text()

    basics(text)
    make_schedule(text)
