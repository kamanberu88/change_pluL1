class LayerActivations:
    features=None

    def __init__(self,model,layer_num):
        self.hook=model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self,module,input,output):
        self.features=output.cpu()

    def remove(self):
        self.hook.remove()
