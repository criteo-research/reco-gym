import pandas as pd
import numpy as np

from recogym import Configuration, build_agent_init
from recogym.agents import OrganicUserEventCounterAgent, organic_user_count_args

dump_agent_args = {
    'agent': build_agent_init(
        'OrganicUserCount',
        OrganicUserEventCounterAgent,
        {**organic_user_count_args}
    )
}


class DumpAgent(OrganicUserEventCounterAgent):
    """
    Dump Agent

    This is the Agent that dumps all its `train' and `act' functions.
    It used mostly for debugging purposes.
    """

    def __init__(self, config=Configuration(dump_agent_args)):
        super(DumpAgent, self).__init__(config)
        self.previous_action = None

        self.data = {
            'case': [],
            't': [],
            'u': [],
            'z': [],
            'v': [],
            'a': [],
            'c': [],
            'ps': [],
            'ps-a': [],
            'done': [],
        }

    def _dump(self, case, observation, action, reward, done):
        def _dump_organic():
            for session in observation.sessions():
                self.data['case'].append(case)
                self.data['t'].append(session['t'])
                self.data['u'].append(session['u'])
                self.data['z'].append('organic')
                self.data['v'].append(session['v'])
                self.data['a'].append(None)
                self.data['c'].append(None)
                self.data['ps'].append(None)
                self.data['ps-a'].append(None)
                self.data['done'].append(done)

        def _dump_bandit():
            if action:
                self.data['case'].append(case)
                self.data['t'].append(action['t'])
                self.data['u'].append(action['u'])
                self.data['z'].append('bandit')
                self.data['v'].append(None)
                self.data['a'].append(action['a'])
                self.data['c'].append(reward)
                self.data['ps'].append(action['ps'])
                self.data['ps-a'].append(action['ps-a'])
                self.data['done'].append(done)

        if case == 'A':
            _dump_bandit()
            _dump_organic()
        else:
            _dump_organic()
            _dump_bandit()

    def train(self, observation, action, reward, done=False):
        self._dump('T', observation, action, reward, done)
        self.config.agent.train(observation, action, reward, done)

    def act(self, observation, reward, done):
        """Make a recommendation"""
        self._dump('A', observation, self.previous_action, reward, done)
        if done:
            return None
        else:
            action = self.config.agent.act(observation, reward, done)
            self.previous_action = action
            return action

    def reset(self):
        super().reset()
        self.config.agent.reset()
        self.previous_action = None

    def dump(self):
        self.data['t'] = np.array(self.data['t'], dtype=np.float32)
        self.data['u'] = pd.array(self.data['u'], dtype=pd.UInt16Dtype())
        self.data['v'] = pd.array(self.data['v'], dtype=pd.UInt16Dtype())
        self.data['a'] = pd.array(self.data['a'], dtype=pd.UInt16Dtype())
        self.data['c'] = np.array(self.data['c'], dtype=np.float32)
        return pd.DataFrame().from_dict(self.data)


agent = build_agent_init('DumpAgent', DumpAgent, {**dump_agent_args})
