class LiveRiskManager:
    def __init__(self, config=None):
        self.config = config

    def check_order(self, order):
        return True

    def update_position(self, position):
        pass
