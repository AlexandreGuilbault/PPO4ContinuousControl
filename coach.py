import torch
import numpy as np

import os
import time

class Coach():
    def __init__(self, env, brain_name, save_directory):
        
        self.env = env
        self.save_directory = save_directory
        self.brain_name = brain_name
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
    
    def train(self, agent, num_episodes, max_steps, log_interval, save_interval):

        agent.set_train(True)
        all_avg_rewards = []
        cum_avg_rewards = []
        num_steps = 0
        
        start_time = time.time()
        for i_episode in range(1, num_episodes+1):
            
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            
            episode_rewards = 0
            for _ in range(max_steps):

                observations = env_info.vector_observations
                with torch.no_grad():
                    actions, values, actions_log_prob = agent.act(observations)

                env_info = self.env.step(actions.squeeze().detach().cpu().numpy())[self.brain_name]

                rewards = torch.FloatTensor(env_info.rewards).unsqueeze(-1).to(agent.device)
                dones = np.asarray(env_info.local_done)  

                agent.memorize(observations, actions, actions_log_prob, values, rewards)

                episode_rewards += rewards.squeeze().detach().cpu().numpy()

                if sum(dones) > 0: break

            num_steps += agent.get_num_steps()
            value_loss, action_loss, entropy_loss, overall_loss = agent.learn()

            avg_episode_reward = episode_rewards.mean()
            all_avg_rewards.append(avg_episode_reward)
            
            cum_avg_reward = np.array(all_avg_rewards[-100:]).mean()
            cum_avg_rewards.append(cum_avg_reward)
            

            completion = (i_episode)/(num_episodes)
            elapsed_time = time.time() - start_time

            em, es = divmod(elapsed_time, 60)
            eh, em = divmod(em, 60)    


            print('\rSteps: {:6.0f} | Episode: {:4.0f}/{} | Cum.Avg.Reward: {:3.3f} | Epis.Avg.Reward: {:3.3f} | Elaps.Time: {:.0f}h {:02.0f}m {:02.0f}s'.format(num_steps, i_episode, num_episodes, cum_avg_reward, avg_episode_reward, eh, em, es), end="")
            if i_episode % log_interval == 0: print()
            if i_episode % save_interval == 0: agent.save(self.save_directory,'PPO_Episode_{}.pth'.format(i_episode))
        
        return all_avg_rewards, cum_avg_rewards
    
    def watch(self, agent):
        
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        scores = np.zeros(agent.get_num_actors())

        total_steps = 0
        while True:
            observations = env_info.vector_observations
            actions = agent.act(observations, deploy=True)
            env_info = self.env.step(actions)[self.brain_name]
            total_steps += 1

            rewards = env_info.rewards
            dones = env_info.local_done
            scores += env_info.rewards

            if np.any(dones): break

        return total_steps, np.mean(scores)
   
        