class Fish:
    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Puffer', 'Piranha', 'Stonefish']
        
    def printMembers(self):
        print('Printing members of the Dangerous Fish class')
        for member in self.members:
            print('\t%s ' % member)

class Birds:
    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Barred Owl', 'Ostrich', 'Emu']
        
    def printMembers(self):
        print('Printing members of the Dangerous Birds class')
        for member in self.members:
            print('\t%s ' % member)