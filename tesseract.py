import pytesseract
from transformers import pipeline
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
image = "testPNG/page0.png"

vqa = pipeline(model="impira/layoutlm-document-qa")

schedule = vqa(
    image=image,
    question="What is the class schedule?",
)
schedule[0]["score"] = round(schedule[0]["score"], 3)

teacher = vqa(
    image=image,
    question="Who is the teacher?"
)
teacher[0]["score"] = round(teacher[0]["score"], 3)

location = vqa(
    image=image,
    question="Where is the class located?"
)
location[0]["score"] = round(location[0]["score"], 3)


question = input("""What would you like to know? (simply type in the letter, e.g. a for teacher)
                 a. Who your teacher is
                 b. What your schedule is
                 c. Where your class is located
                 d. A list of your assingments
                 - To exit type 'quit'

""")

while True:
    if question == 'a':
        print(f"The teacher is {teacher[0]['answer']} ({teacher[0]['score']})")
        question = input()
    if question == 'b':
        print(
            f"The schedule is {schedule[0]['answer']} ({schedule[0]['score']})")
        question = input()
    if question == 'c':
        print(
            f"The class is located at {location[0]['answer']} ({location[0]['score']})")
        question = input()
    if question == 'd':
        print('Still working on gathering schedule data')
        question = input()
    if question == 'quit':
        break
    else:
        print("Incorrect input")
        question = input()
