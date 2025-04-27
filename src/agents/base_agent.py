class BaseAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, observation):
        """
        Given an observation, select an action.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement select_action method.")