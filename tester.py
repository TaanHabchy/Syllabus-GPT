from datasets import load_dataset
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import os
from transformers import pipeline
import PyPDF2
from transformers import AutoTokenizer
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


squad = load_dataset("squad", split="train[:5000]")
squad = squad.train_test_split(test_size=0.2)


tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


tokenized_squad = squad.map(
    preprocess_function, batched=True, remove_columns=squad["train"].column_names)

data_collator = DefaultDataCollator()

model = AutoModelForQuestionAnswering.from_pretrained(
    "distilbert/distilbert-base-uncased")

training_args = TrainingArguments(
    output_dir="my_awesome_qa_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()


# def basics(text):

#     subject = "(Philosophy, Calculus, Literary Tradition): What is the class subject?"
#     teacher = "(prompt: Dr, Prof): Who is the teacher or proffessor?"
#     location = "Where is the classroom located?"
#     schedule = "(prompt: date): What is the class schedule?"

#     """
#     Old Model: bert-large-uncased-whole-word-masking-finetuned-squad
#     """
#     question_answerer = pipeline(
#         "question-answering", model="my_awesome_qa_model")

#     mySubject = question_answerer(question=subject, context=text)
#     myTeacher = question_answerer(question=teacher, context=text)
#     myLocation = question_answerer(question=location, context=text)
#     mySchedule = question_answerer(question=schedule, context=text)

#     print(f"Subject: {mySubject['answer']} - {mySubject['score']}")
#     print(f"Teacher: {myTeacher['answer']} - {myTeacher['score']}")
#     print(f"Location: {myLocation['answer']} - {myLocation['score']}")
#     print(f"Schedule: {mySchedule['answer']} - {mySchedule['score']}")
#     questions = [q.strip() for q in examples["question"]]
#     inputs = tokenizer(
#         questions,
#         examples["context"],
#         max_length=384,
#         truncation="only_second",
#         return_offsets_mapping=True,
#         padding="max_length",
#     )

#     offset_mapping = inputs.pop("offset_mapping")
#     answers = examples["answers"]
#     start_positions = []
#     end_positions = []

#     for i, offset in enumerate(offset_mapping):
#         answer = answers[i]
#         start_char = answer["answer_start"][0]
#         end_char = answer["answer_start"][0] + len(answer["text"][0])
#         sequence_ids = inputs.sequence_ids(i)

#         # Find the start and end of the context
#         idx = 0
#         while sequence_ids[idx] != 1:
#             idx += 1
#         context_start = idx
#         while sequence_ids[idx] == 1:
#             idx += 1
#         context_end = idx - 1

#         # If the answer is not fully inside the context, label it (0, 0)
#         if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
#             start_positions.append(0)
#             end_positions.append(0)
#         else:
#             # Otherwise it's the start and end token positions
#             idx = context_start
#             while idx <= context_end and offset[idx][0] <= start_char:
#                 idx += 1
#             start_positions.append(idx - 1)

#             idx = context_end
#             while idx >= context_start and offset[idx][1] >= end_char:
#                 idx -= 1
#             end_positions.append(idx + 1)

#     inputs["start_positions"] = start_positions
#     inputs["end_positions"] = end_positions
#     return inputs


# text = ""
# with open("syllabi/test.txt", 'r') as file:
#     if 'txt' in file.name:
#         text = file.read()
#     elif 'pdf' in file.name:
#         with open(file.name, 'rb') as file:
#             reader = PyPDF2.PdfReader(file)
#             for i in range(len(reader.pages)):
#                 page = reader._get_page(i)
#                 text += page.extract_text()

#     basics(text)
#     # make_schedule(text)
