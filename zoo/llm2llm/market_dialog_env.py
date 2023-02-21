from data import FruitDataset

class market_dialog_env:
    def __init__(self, args, lm_server):
        """
        :param args: Arguments
            args.max_turn: Maximum number of turns
            args.llm_server: LLM server
            args.context: context, the available agent-specific information in string format (or should be tokenized?)
        """
        # Load the dataset
        dataset = FruitDataset("./data/mindless_dataset_randomized_train.txt")
        # dataset = FruitDataset(args.dataset_path)
        # Load the dataloader
        self.dataloader = iter(DataLoader(dataset, batch_size=32, shuffle=True))
        self.max_turn = args.max_turn # should be 2
        self.llm_server = lm_server # Caller(args.lamorel_args_2) if we're using different models
        self.obs = None
        self.dataloader = iter(args.dataloader)

    def reset(self):
        """
        Reset the environment
        :return: Initial observation
        """
        self.turn = 0
        self.epoch_data = self.dataloader.next()
        self.question = self.epoch_data[0]
        self.answer = self.epoch_data[-1]
        self.obs = self.question + self.epoch_data[self.turn + 1]
        return self.obs
    
    def step(self, action):
        """
        Step the environment
        :param action: Action to take
        :return: (observation, reward, done, info)
        """
        self.turn += 1
        if self.turn != self.max_turn:
            output = self.llm_server.generate([action + self.epoch_data[self.turn + 1]])
            self.obs = output[0][0]["text"] + self.question
            done = False
            reward = 0
        else:
            reward = 1 if self.answer == action else 0
            done = True
        return self.obs, reward, done, {}
