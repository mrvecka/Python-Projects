students = []

def get_students_titlecase():
    students_titlecase = []
    for student in students:
        students_titlecase = student.title()
    return students_titlecase


def print_student_titlecase():
    students_titlecase = get_students_titlecase()
    print(students_titlecase)


def add_student(name, student_id = 332):
    student = {"name": name, "student_id": student_id}
    students.append(student)

#variable number of parameters
def var_args(name, *args):
    print(name)
    print(args)

#key value parameters
def var_kwargs(name, **kwargs):
    print(name)
    print(kwargs["description"], kwargs["module"])

def print_all_students():
    for st in students:
        print(st["name"],st["student_id"])
        
inputExit = "Y"
while inputExit != "N":
    print("\n")
    name = input("Enter student Name:")
    student_id = input("Enter student ID:")
    add_student(name,student_id)
    print("Studem",name,"has been added to program!")
    inputExit = input("Type y if you want to continue otherwise type n")
    inputExit = inputExit.upper()


print_all_students()




#add_student("Mark", student_id=15)
#var_args("Mark", "python",3, None, True)

#var_kwargs("Mark", description="python",module=3,subscriber= None, helpful=True)




