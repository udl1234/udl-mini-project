import numpy as np
from torchvision import datasets, transforms
from utils.misc import transform_flatten


def permute_mnist(n_tasks=10, seed=0):
    tasks = []
    for i in range(n_tasks):
        np.random.seed(seed + i)
        perm_inds = list(range(28 * 28))
        if i > 0:
            np.random.shuffle(perm_inds)

        def permute_image(perm_inds):
            def permute(img):
                return img[perm_inds]

            return permute

        t = transforms.Compose(
            [transforms.ToTensor(), transform_flatten, permute_image(perm_inds)]
        )

        mnist_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=t
        )
        mnist_test = datasets.MNIST(
            root="./data", train=False, download=True, transform=t
        )

        tasks.append((mnist_train, mnist_test))

    return tasks
