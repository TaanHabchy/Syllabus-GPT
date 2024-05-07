import PyPDF2
import re
import formats.classes as classes
import formats.dates as formats


def basic_info(text):
    # Regular expressions for getting the subject, pattern, and classes
    subject_pattern = r"\b(?:" + "|".join(map(re.escape,
                                              classes.college_majors)) + r")\b"
    prof_pattern = r"(Dr.|Prof|Doctor|Professor|Instructor|Teacher)\s+(\w+\s+\w+)"
    class_times = r'(MWF|TR|M|W|F)\s+([\s\S]*?)(?=\n|\s{2,})'

    # try/except so program doesn't crash if we don't find one of these parameters
    try:
        subject = re.search(subject_pattern, text, flags=re.IGNORECASE).group()
    except:
        subject = 'Unknown subject'
    try:
        proffessor = re.search(prof_pattern, text, flags=re.IGNORECASE).group()
    except AttributeError:
        proffessor = "unknown"
    try:
        schedule = re.search(class_times, text, flags=re.IGNORECASE).group()
    except:
        schedule = "unknown"

    print(f"'{subject}' is taught by {proffessor} every {schedule} \n")


def make_schedule(text):

    # matches either two main date formats
    date_pattern = f"{formats.f1}|{formats.f2}|{formats.f3}"  # |{formats.f3}"
    # Find all occurrences of dates in the text using the regex pattern
    dates = re.findall(date_pattern, text)
    print(dates)
    # Create an array to store classes
    class_assignments = []
    # Iterate over the extracted dates
    for i, date_str in enumerate(dates):

        # Find the next date, if available
        next_index = i + 1

        # goes from the end of the date(so beginning of assingment), to the beginning of next date
        if next_index < len(dates):
            assignment = text[text.find(
                date_str) + len(date_str):text.find(dates[next_index])]
        # finds last assingment
        else:
            temp = ''
            for index in range(text.find(dates[-1]) + len(dates[-1]), len(text)):
                temp += text[index]
            assignment = temp

        assignment = assignment.strip()

        # Create a class containing the date object and its corresponding assignment
        class_assignment = {'date': date_str, 'assignment': assignment}
        # Append the class to the array
        class_assignments.append(class_assignment)

    # Print the array of classes
    if len(class_assignments) > 0:
        print('The schedule is as follows: ')

    for class_assignment in class_assignments:
        spaces = 15 - len(class_assignment['date'])
        mid = ""
        for i in range(spaces):
            mid += " "
        print(
            f"{class_assignment['date']}:{mid}{class_assignment['assignment']}")


with open("syllabi/test.txt", 'r') as file:
    if 'txt' in file.name:
        text = file.read()
    elif 'pdf' in file.name:
        with open(file.name, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for i in range(len(reader.pages)):
                page = reader._get_page(i)
                text += page.extract_text()
    basic_info(text)
    make_schedule(text)
    file.close()
