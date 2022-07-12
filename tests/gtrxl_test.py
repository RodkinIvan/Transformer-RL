import os
import torch
import numpy as np
from VMPO import Memory, VMPO, PPO
from hps import set_up_hyperparams
import gym_deepmindlab
import gym


def main():
    H, logprint = set_up_hyperparams()

    env = gym.make('DeepmindLabNavMazeRandomGoal01-v0',
                   observations=['RGB_INTERLEAVED'],
                   width=64,
                   height=64,
                   )
    H.img_size = 64
    H.device = 'cuda:' + H.gpu if H.gpu is not None else 'cpu'

    memory = Memory()
    if H.model == 'vmpo':
        agent = VMPO(H)
    elif H.model == 'ppo':
        agent = PPO(H)

    # Logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # Training loop
    for i_episode in range(1, H.max_episodes + 1):
        img = env.reset()
        for t in range(H.max_timesteps):
            timestep += 1

            # Running policy_old:
            action = agent.policy_old.act(t, img, memory)
            new_img, reward, done, info = env.step(H.action_list[action].astype(np.intc))


            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            running_reward += reward

            img = new_img
            if done:
                break

        # Update if its time
        if timestep > H.update_timestep:
            agent.update(memory)
            memory.clear_memory()
            timestep = 0

        avg_length += t

        # Logging
        if i_episode % H.log_interval == 0:
            avg_length = int(avg_length / H.log_interval)
            running_reward = int((running_reward / H.log_interval))

            logprint(model=H.desc, type='tr_loss', episodes=i_episode,
                     **{'avg_length': avg_length, 'running_reward': running_reward})
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
