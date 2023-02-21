from lamorel import BaseModuleFunction

class TwoLayersMLPModuleFn(BaseModuleFunction):
    def __init__(self, model_type, n_outputs):
        super().__init__()
        self._model_type = model_type
        self._n_outputs = n_outputs

    def initialize(self):
        '''
        Use this method to initialize your module operations.
        - self.llm_config gives the configuration of the LLM (e.g. useful to know the size of representations)
        - self.device gives you access to the main device (e.g. GPU) the LLM is using
        '''
        llm_hidden_size = self.llm_config.to_dict()[self.llm_config.attribute_map['hidden_size']]
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, 128),
            torch.nn.Linear(128, 128),
            torch.nn.Linear(128, self._n_outputs),
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_context, **kwargs):
        '''
        Perform your operations here.
        - forward_outputs gives access the output of the computations performed by the LLM (e.g. representations of each layer)
        - minibatch gives access to the input data (i.e. a prompt and multiple candidates) given to the LLM
        - tokenized_context gives access to the prompt used
        '''
        # Get the last layer's representation from the token right after the prompt
        if self._model_type == "causal": # adapt to the Transformers API differing between Encoder-Decoder and Decoder-only models
            model_head = forward_outputs['hidden_states'][0][0, len(tokenized_context["input_ids"])-1, :]
        else:
            model_head = forward_outputs['encoder_last_hidden_state'][0, len(tokenized_context["input_ids"]) - 1, :]
        
        # Give representation to our MLP
        output = self.mlp(model_head)
        return output