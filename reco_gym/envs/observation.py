class Observation:
    def __init__(self, context, sessions):
        self.current_context = context
        self.current_sessions = sessions

    def context(self):
        return self.current_context

    def sessions(self):
        return self.current_sessions
