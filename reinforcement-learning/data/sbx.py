import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['text.color'] = 'black'

def sbx_crossover(p1, p2, sbxdi):
    D = p1.shape[0]
    cf = np.empty([D])
    u = np.random.rand(D)        

    cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (sbxdi + 1)))
    cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (sbxdi + 1)))

    c1 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
    c2 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)

    c1 = np.clip(c1, 0, 1)
    c2 = np.clip(c2, 0, 1)

    return c1, c2

def main():
    p1 = np.array([0.2, 0.8])
    p2 = np.array([0.8, 0.2])
    np.random.seed(0)

    sbxdis = [1, 2, 5, 10]

    for i in range(4):
        plt.subplot(2, 2, i+1)
        sbxdi = sbxdis[i]

        x = []
        y = []
        for i in range(100):
            c1, c2 = sbx_crossover(p1, p2, sbxdi)
            x.append(c1[0])
            x.append(c2[0])
            y.append(c1[1])
            y.append(c2[1])

        plt.scatter(x, y, s=[10 for i in range(100)], label='Offspring')
        plt.xlim((-0.2, 1.2))
        plt.ylim((-0.2, 1.2))


        x = [0.2, 0.8]
        y = [0.8, 0.2]
        plt.scatter(x, y, s=[40 for i in range(100)], label='Parent')

        plt.legend()
        plt.title('SBXDI=%d' % sbxdi)
    plt.show()


if __name__ == '__main__':
    main()