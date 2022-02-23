class Person(object):
    def __init__(self, firstName, lastName):    
        self.firstName = firstName
        self.lastName = lastName
        
    def __str__(self):
        return "%s %s" % (self.firstName, self.lastName)

        
class Student(Person):  
    def __init__(self ,firstName, lastName, subject):
        super(Student, self).__init__(firstName,lastName)
        self.subject = subject
        
    def printNameSubject(self) :
        print(super(Student, self).__str__() + ', ' + self.subject)
        
    
class Teacher(Person):
    def __init__(self ,firstName, lastName, courseName):
        super(Teacher, self).__init__(firstName,lastName)
        self.courseName = courseName
        
    def printNameSubject(self) :
        print(super(Teacher, self).__str__() + ', ' + self.courseName)