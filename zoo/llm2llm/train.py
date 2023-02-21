import hydra
from lamorel import Caller, lamorel_init
lamorel_init()

class LLMAgent:
    def __init__(self, lm_server):
        self._lm_server = lm_server

    def generate_goal(self, initial_obs):
        promt_suffix = "\nThis is an example of what I could do here:"
        prompt = initial_obs + promt_suffix
        result = self._lm_server.generate(contexts=[prompt],
                                          max_new_tokens=15,
                                          do_sample=True,
                                          temperature=1,
                                          top_p=0.70,
                                          top_k=0)

        goal = result[0][0]["text"]
        return goal

    def play(self, obs, possible_actions):
        prompt_prefix = "\nYou choose to "
        output = self._lm_server.generate([prompt_prefix + obs])
        action_idx = np.argmax(scores[0]).item()
        return possible_actions[action_idx]

@hydra.main(config_path='/homedtcl/mmahaut/projects/lamorel/zoo/llm2llm/configs/', config_name='local_gpu_config')
def main(config_args):
    lm_server = Caller(config_args.lamorel_args)
    # Do whatever you want with your LLM
    print(lm_server.forward(['I have 10 peaches, how many peaches do I have ?', 'You have 10 peaches, how many peaches do you have ?', 'Once upon a time there was a boy who']))
    lm_server.close()

def main2(config_args):
    # setup llm server with lamorel
    lm_server = Caller(config_args.lamorel_args,
                    custom_updater_class=TestUpdater)    
    # TODO : setup adapter instead of complete retraining
    # setup the environment
    env = market_dialog_env(config_args.rl_script_args, lm_server)
    
    for i in len(env.dataloader):
        obs = env.reset()
        done = False
        cumulated_reward = 0
        while not done:
            print(obs)
            action = lm_server.generate(obs)
            print(action)
            obs, reward, done, info = env.step(action)
            cumulated_reward += reward
        print("Episode reward: ", cumulated_reward)
        # train the llm
        output = lm_server.update(
                    inputs=[obs],
                    labels=torch.tensor([0, 1], dtype=torch.float32),
                )
        losses = [r["loss"] for r in result] # one loss returned per LLM
    lm_server.close()
if __name__ == '__main__':
    main()
