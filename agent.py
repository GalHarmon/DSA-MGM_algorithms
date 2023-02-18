class Agent:
    # agent constructor
    def __init__(self, id, assignment):
        self.id = id
        self.assignment = assignment # for initial iteration
        self.assignment_val = 0 # for initial iteration
        self.neighbors = {}
        self.alternative_best_assin = assignment
        self.alternative_best_assin_value = 0

    # returns agent id
    def get_id(self):
        return self.id

    # saves neighbors and sets constraints_table
    def set_neighbor(self, neighbor, constraints_table):
        self.neighbors[neighbor] = constraints_table

    # returns dictionary of neighbors with constraints table
    def get_neighbors(self):
        return self.neighbors

    # sets current assignment
    def set_curr_assig(self, assignment):
        self.assignment = assignment

    def get_curr_assig(self):
        return self.assignment

    # sets previous assignment
    def set_alternative_best_assin(self, alternative_best_assin, alternative_best_assin_value):
        self.alternative_best_assin = alternative_best_assin
        self.alternative_best_assin_value = alternative_best_assin_value

    # gets alternative best assignment
    def get_alternative_best_assin(self):
        return self.alternative_best_assin

    # gets alternative best assignment value
    def get_alternative_best_assin_value(self):
        return self.alternative_best_assin_value

    def set_assignment_val(self, assignment_val):
        self.assignment_val = assignment_val

    # gets assignment val
    def get_assignment_val(self):
        return self.assignment_val
