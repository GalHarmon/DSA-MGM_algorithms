class Message:
    # Message constructor
    def __init__(self, from_who, to_who, value):
        self.from_who = from_who
        self.to_who = to_who
        self.value = value

    # returns the receives id
    def get_to_who(self):
        return self.to_who.get_id()

    # returns the sender id
    def get_from_who(self):
        return self.from_who.get_id()

    # returns the sender id
    def get_value(self):
        return self.value