from lamorel import BaseUpdater

class CERLUpdater(BaseUpdater):
    """
    Cross Entropy + PPO Loss Updater
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = 20 # kwargs.max_length
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = torch.nn.CrossEntropyLoss()
        if not hasattr(self, 'optimizer'):
            # You can train:
            # training the LLM. TODO : only train the adapter
            self.optimizer = torch.optim.Adam(self._llm_module._LLM_model.parameters())
        self.loss = []

    def forward(self, input)
        """
        go through the llm, and calculate the loss
        """

        # Use the computational graph with gradient
        # 1. Only the LLM's scoring module
        
        self._llm_module(input, require_grad=True)
        # output = self._llm_module(['text','__score'], contexts=contexts, require_grad=True)
        
        # Stack outputs to batch loss computation
        stacked_output = torch.stack([_o['__score'] for _o in output]).to('cpu')
        text_output = [o['text'] for o in output]
        
        # Compute loss with the labels corresponding to the current batch
        self.loss.append(self.loss_fn(stacked_output, kwargs["labels"][_current_batch_ids, :]))
        return text_output, stacked_output, self.loss[-1]

    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        """
        for now this one is only doing cross entropy loss, and on the whole model
        """
        
        # Compute gradients and update graph
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Return anything you want using a dictionary
        return {"loss": loss}