class Session(list):
    """Abstract Session class"""

    def to_strings(self, user_id, session_id):
        """represent session as list of strings (one per event)"""
        user_id, session_id = str(user_id), str(session_id)
        session_type = self.get_type()
        strings = []
        for event, product in self:
            columns = [user_id, session_type, session_id, event, str(product)]
            strings.append(','.join(columns))
        return strings


class Organic_Session(Session):

    def next(self, product):
        event = 'pageview'

        self.append((event, product))

    def get_type(self):
        return 'organic'

    def get_views(self):
        return [p for e, p in self if e == 'pageview']
