import numpy as np
import matplotlib.pyplot as plt
from wsgiref.handlers import CGIHandler

repeat = 1
NUMBER_OF_ALG1_TASK = 3
NUMBER_OF_ALG2_TASK = 1
ALG1 = "MFEA"
ALG2 = "MFEAII"
problem="ionosphere"
def load(algorithm):
    results = []
    for seed in range(1, repeat + 1):
        stats = np.load('results/%s/%s/%d.npy' % (problem, algorithm, seed))
        results.append(stats)
    return np.array(results)



def main():
    config = get_config('config.yaml')
    conn = create_connection(config)
    cur = conn.cursor()
    cur.execute('SELECT TABLE iteration ADD COLUMN rmp VARCHAR(128);')

    results1 = load("mfea_result")
    results2 = load("mfeaii_result")
    fig, axes = plt.subplots(2, 3)
    axes = axes.flatten()

    for k in range(NUMBER_OF_ALG1_TASK):
        ax = axes[k]

        # ARS
        result = results1[:, :, k]

        mu = np.mean(result, axis = 0)
        sigma = np.std(result, axis = 0)
        x = np.arange(result.shape[1])

        line1, = ax.plot(x, mu, color = "blue")
        ax.fill_between(x, mu + sigma, mu - sigma, color = "blue", alpha = 0.3)

        # MFARSRR
        # result = results2[:, :, k]
        result = results2[:, :, 0]

        mu = np.mean(result, axis = 0)
        sigma = np.std(result, axis = 0)
        x = np.arange(result.shape[1])

        line2, = ax.plot(x, mu, color = "orange")
        ax.fill_between(x, mu + sigma, mu - sigma, color = "orange", alpha = 0.3)

        # Legend
        ax.grid()
        ax.legend((line1, line2), (ALG1, ALG2))
    plt.show()
    plt.savefig("mean_and_std.eps", format = "eps")


if __name__ == "__main__":
    main()
