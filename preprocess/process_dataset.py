import torch


def computeMeanStd(images: torch.Tensor):

    batch, channel = images.shape[0], images.shape[1]
    mean = [0 for i in range(channel)]
    std = [0 for i in range(channel)]
    for i in range(batch):
        for j in range(channel):
            
            mean[j] += images[i, j, :, :].mean().item()
            std[j] += images[i, j, :, :].std().item()

    return mean, std



if __name__ == "__main__":
    image = torch.randn((3,3,32,32))
    print(computeMeanStd(image))