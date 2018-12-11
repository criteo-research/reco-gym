class Context:

    def time(self):
        pass

    def user(self):
        pass


class DefaultContext(Context):

    def __init__(self, current_time, current_user_id):
        super(DefaultContext, self).__init__()
        self.current_time = current_time
        self.current_user_id = current_user_id

    def time(self):
        return self.current_time

    def user(self):
        return self.current_user_id
