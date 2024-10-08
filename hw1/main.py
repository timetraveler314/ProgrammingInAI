from net_def import *


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()  # move back to cpu
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Test the network with a batch (4 images) of test images
def test_network(net):
    dataiter = iter(testloader)
    images, labels = next(dataiter)[0].to(device), next(dataiter)[1].to(device)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))


def test_total_accuracy(net):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct // total


def test_class_accuracy(net):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    class_accuracy = {classname: 0.0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        class_accuracy[classname] = 100 * float(correct_count) / total_pred[classname]

    return class_accuracy


def train_over_multiple_momentum(momentum_values):
    momentum_loss_map = {}

    for momentum in momentum_values:
        print(f'Training the network with momentum = {momentum}...')
        net = Net().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=momentum)
        loss_list = train(net, criterion, optimizer)
        momentum_loss_map[momentum] = loss_list
        print(f'Training with momentum = {momentum} is done.')

    return momentum_loss_map


if __name__ == '__main__':
    # Plot the training loss curve for different momentum values
    momentum_loss_map = train_over_multiple_momentum([0.0, 0.3, 0.5, 0.9, 0.99])

    plt.figure(figsize=(10, 6))
    for momentum, loss_list in momentum_loss_map.items():
        plt.plot(loss_list, label=f'Momentum = {momentum}', color=np.random.rand(3))

    plt.xlabel('Mini-batch (x2000)')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve of LeNet on CIFAR-10 for Different Momentum Values')
    plt.legend()
    plt.show()

    print('Training loss curves for different momentum values:', momentum_loss_map)

    # Below are the code for training the network and testing for accuracy

    train_flag = True

    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if train_flag:
        loss_list = train(net, criterion, optimizer, save=True)
        print('Loss list:', loss_list)

        plt.figure(figsize=(10, 6))
        plt.plot(loss_list, label='Loss', color='blue', linestyle='dashed')
        plt.xlabel('Epoch, Mini-batch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve of LeNet on CIFAR-10')
        plt.legend()
        plt.show()
    else:
        net.load_state_dict(torch.load(PATH, weights_only=True))

    test_network(net)

    total_acc = test_total_accuracy(net)
    print(f'Accuracy of the network on the 10000 test images: {total_acc} %')

    class_acc = test_class_accuracy(net)
    for classname, acc in class_acc.items():
        print(f'Accuracy of {classname:5s}: {acc:.2f} %')
