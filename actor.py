

class Actor:
    def __init__(self, input_dim, output_dim, lr=0.005, use_gpu=False):
        from net import Net
        self.brain = Net(input_dim=input_dim, output_dim=output_dim, lr=lr, use_gpu=use_gpu)

    def predict_one(self, x):
        return self.brain.predict_one(x)

    def get_weights(self):
        return self.brain.get_weights()

    def set_weights(self, weights):
        self.brain.set_weights(weights)