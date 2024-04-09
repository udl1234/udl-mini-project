from torchvision import datasets, transforms

from utils.misc import transform_flatten


def split_mnist(n_tasks=5):
    assert n_tasks <= 5, "Split MNIST supports at most 5 tasks"

    tasks = []

    sets_0 = [0, 2, 4, 6, 8]
    sets_1 = [1, 3, 5, 7, 9]

    for i in range(n_tasks):

        def filter_dataset(dataset, digit0, digit1):

            filter_idx = (dataset.targets == digit0) | (dataset.targets == digit1)
            filtered_data = dataset.data[filter_idx]
            filtered_labels = dataset.targets[filter_idx]
            binary_labels = (filtered_labels == digit1).long()

            dataset.data = filtered_data
            dataset.targets = binary_labels

        t = transforms.Compose([transforms.ToTensor(), transform_flatten])

        mnist_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=t
        )
        mnist_test = datasets.MNIST(
            root="./data", train=False, download=True, transform=t
        )

        filter_dataset(mnist_train, sets_0[i], sets_1[i])
        filter_dataset(mnist_test, sets_0[i], sets_1[i])

        tasks.append((mnist_train, mnist_test))

    return tasks
