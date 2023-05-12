from matplotlib import pyplot as plt
import pickle
import numpy as np


def visualize(results_path):
    # Load results
    with open('results/120523_no_death_ranged.pt', 'rb') as fp:
        fp.seek(0)
        results = pickle.load(fp)

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


if __name__ == "__main__":
    visualize()
