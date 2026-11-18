import numpy as np
import matplotlib.pyplot as plt
from tools.inpainting_ch import CH2D

def circle_mask(img, cx, cy, r):
    shape = img.shape
    y, x = np.ogrid[:shape[0], :shape[1]]
    return (x - cx)**2 + (y - cy)**2 < r**2

def gen_bars(N, n_bars):
    img = np.ones([N, N])
    nodes_per_bar = N // n_bars

    for i in range(1, n_bars, 2):
        img[nodes_per_bar*i:nodes_per_bar*(i+1), :] = -1

    return img

def gen_checkerboard(N, n_squares):
    img = np.ones([N, N])
    nodes_per_bar = N // n_squares

    for i in range(1, n_squares, 2):
        for j in range(1, n_squares, 2):
            img[nodes_per_bar*i:nodes_per_bar*(i+1), nodes_per_bar*j:nodes_per_bar*(j+1)] = -1
            img[nodes_per_bar*(i-1):nodes_per_bar*(i), nodes_per_bar*(j-1):nodes_per_bar*(j)] = -1

    return img

def gen_damage(img, pos=[0.5, 0.5], rad=0.1, val=0, damage_type='square'):
    N = img.shape[0]

    x = int(pos[0] * N)
    y = int(pos[1] * N)

    r = int(rad * N) - 1

    if damage_type == 'square':
        img[x - r: x + r, y - r: y + r] = val
    elif damage_type == 'circle':
        img[circle_mask(img, x, y, r)] = val
    return img



if __name__ == "__main__":

    imgs = []
    N = 256
    img = gen_checkerboard(N, 2)
    # img = gen_bars(N, 4)
    imgs.append(img.copy())
    

    img = gen_damage(img, pos=[0.5, 0.5], rad=0.1, damage_type='square')
    # plt.imshow(img, cmap='binary', vmax=1, vmin=-1)
    # plt.show()

    imgs.append(img)

    solver = CH2D(img, eps=0.1, dt=1e-7)
    for i in range(400):
        solver.step()
    imgs.append(solver.Un[:N**2].reshape((N, N)))

    solver.set_eps(0.005)
    for i in range(400):
        solver.step()

    solver.set_eps(0.0005)
    for i in range(4000):
        solver.step()
    imgs.append(solver.Un[:N**2].reshape((N, N)))


    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    titles = [
        "Original Image",
        "Damaged Image",
        "Blurred Image",
        "Restored Image"
    ]
    axs = axs.flatten()

    for i, img in enumerate(imgs):
        ax = axs[i]
        im = ax.imshow(img, extent=[0, 1, 0, 1], origin='lower', vmin=-1.0, vmax=1.0, cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(titles[i], fontweight='bold')

    plt.show()