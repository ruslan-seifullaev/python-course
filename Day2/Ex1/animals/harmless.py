class Fish:
    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Cod', 'Notothenia', 'Sardines']
        
    def printMembers(self):
        print('Printing members of the Dangerous Fish class')
        for member in self.members:
            print('\t%s ' % member)

class Birds:
    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Dove', 'Electus', 'Parrot']
        
    def printMembers(self):
        print('Printing members of the Dangerous Birds class')
        for member in self.members:
            print('\t%s ' % member)