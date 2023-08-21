from matplotlib import pyplot as plt
import pickle
import numpy as np


def visualize(results_path):
    # Load results
    with open(results_path, 'rb') as fp:
        fp.seek(0)
        results = pickle.load(fp)

    # Average Damage
    # print(sum(results['damage']) / len(results['scatter_density']))
    # print(sum(results['scatter_density']) / len(results['scatter_density']))

    # Network reward
    plt.figure()
    plt.title("UnitAI2D-Training Progress - Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Network Reward")

    x = np.arange(0, len(results['reward']))
    plt.plot(x, results['reward'], label="Reward")

    # Network loss
    plt.figure()
    plt.title("UnitAI2D-Training Progress - Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Network Loss")

    x = np.arange(0, len(results['loss']))
    plt.plot(x, results['loss'], label="Loss")

    plt.show()


if __name__ == "__main__":
    visualize('results/experiment_4_train.pt')
