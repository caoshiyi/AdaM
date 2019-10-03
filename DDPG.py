from Environment import MetaEnvironment
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU

# import timeit

OU = OU()  # Ornstein-Uhlenbeck Process


def playGame(train_indicator=0):  # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.00001  # Learning rate for Actor
    LRC = 0.0001  # Lerning rate for Critic

    server_number = 5
    # node_number = 18
    hot_node_number = 150
    action_dim = hot_node_number  # Number of servers
    state_dim = hot_node_number * (server_number + 1 + 10)  # 1000 node * 10 features
    # baseline = 4e-05 #load&locality of baselines

    np.random.seed(500)

    # vision = False

    EXPLORE = 100000.
    episode_count = 100
    max_steps = 100000
    line_number = 1000
    step_number = 35
    # reward = 0
    done = False
    step = 0
    epsilon = 1
    # indicator = 0

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    # Generate a MDS environment
    env = MetaEnvironment(server_number)

    # Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("model/actormodel-" + str(server_number) + ".h5")
        critic.model.load_weights("model/criticmodel-" + str(server_number) + ".h5")
        actor.target_model.load_weights("model/actormodel-" + str(server_number) + ".h5")
        critic.target_model.load_weights("model/criticmodel-" + str(server_number) + ".h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("Experiment Start.")

    f = open("query.txt")
    queryList = []
    for line in f.readlines():
        line = line.strip()
        queryList.append(line)
    f.close()

    sumLoc = 0
    sumLod = 0
    lossList = []
    mdsLoadList = [[] for x in range(server_number)]

    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        # if np.mod(i, 3) == 0:
        # ob = env.reset(relaunch=True)   #relaunch every 3 episode because of the memory leak error
        # else:
        # ob = env.reset()

        traceList = queryList[0:line_number]  # Reset
        s_t = env.state(traceList)  # Get State from env

        localityList = []
        loadList = []

        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            # add noise
            a_t_original = actor.model.predict(s_t)
            for k in range(action_dim):
                noise_t[0][k] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][k], 0.0, 0.60, 0.30)

            for m in range(action_dim):
                a_t[0][m] = a_t_original[0][m]  # + noise_t[0][m]

            migration = env.take_actions(a_t[0])
            print("migration", migration)

            tracelist = queryList[(j + 1) * line_number:(j + 2) * line_number]
            s_t1 = env.state(tracelist)  # Update state from env
            # r_t = 0.5*env.locality() + 50*env.load() - baseline  
            # print("gagaga", 1e5*env.locality() + 1e7*env.load())
            # 1.5, 3, 2
            x = 1e5 * env.locality() + 1e7 * env.load() - 1.5 * migration
            # x = 1e5*env.locality() + 1.5 * 1e7*env.load()
            # r_t = 1.0 / (1.0 + np.exp(-(x/50)))
            r_t = x

            if j == step_number:
                done = True
            else:
                done = False

            buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer

            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])
            states = states.reshape(len(batch), -1)
            new_states = new_states.reshape(len(batch), -1)
            actions = actions.reshape(len(batch), -1)

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            # print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss, "Locality", env.locality(), "Load", env.load())
            print("Episode", i, "Step", step, "Reward", r_t, "Loss", loss, "Locality", env.locality(), "Load",
                  env.load())

            lossList.append(loss)
            localityList.append(env.locality())
            loadList.append(env.load())
            for index in range(server_number):
                mdsLoadList[index].append(env.loadList[index])

            step += 1
            if done:
                break

        curLocalitySum = sum(localityList)
        curLoadSum = sum(loadList)

        # f = open('' + str(server_number) + '.txt', 'w')
        # f.write(','.join(map(str, lossList)))
        # f.close()

        # f = open('anglecut-mdsload-' + str(server_number) + '.txt', 'w')
        # for i in range(server_number):
        #     f.write(','.join(map(str, mdsLoadList[i])))
        #     f.write('\n')
        # f.close()
        # print("写入成功")

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("model/actormodel-" + str(server_number) + ".h5", overwrite=True)
                with open("model/actormodel-" + str(server_number) + ".json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("model/criticmodel-" + str(server_number) + ".h5", overwrite=True)
                with open("model/criticmodel-" + str(server_number) + ".json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        # print("Final Locality:", env.final_locality(), "Final Load Balancing:", env.final_load())     
        # env.clear()
        print("")

    # env.end()
    print("Finish.")


if __name__ == "__main__":
    playGame(1)
