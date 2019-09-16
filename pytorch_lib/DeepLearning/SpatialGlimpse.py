import torch
import torch.nn as nn


class SpatialGlimpse(nn.Module):

    def __init__(self, retina, scale, depth, channels=False, flatten=True):
        """
        Layer defining spatial  glimpses
        :param retina: size of the retina, it is assumed as a square
        :param scale: factor by which the retina grows with depth
        :param depth: number of stacked growing patches
        :param channels: input image is using more than one channel
        :param flatten: return as flatted array instead of image patch
        """
        super(SpatialGlimpse, self).__init__()

        self.retina = retina - retina % 2
        self.scale = scale
        self.depth = depth
        self.flatten = flatten

        self.hard_tanh = nn.Hardtanh()
        self.pad_width = int(self.retina / 2) * scale ** (depth - 1)
        self.padding = nn.ZeroPad2d(self.pad_width)
        if channels:
            self.padding = nn.ConstantPad3d(self.pad_width, 0.)
        self.scaler = nn.AdaptiveAvgPool2d((retina, retina))

        print('pad_width:', self.pad_width)
        print('scale:', self.scale)
        print('depth:', self.depth)
        print('retina:', self.retina, '\n')

    def forward(self, X, loc):
        """
        cropping the image
        :param X: batch of images
        :param loc: batch of locations
        :return: croped images
        """
        with torch.no_grad():
            loc = self.hard_tanh(loc)
            x_shape = torch.tensor(X.shape[1:], dtype=torch.float)

            # padding should be done outside of the learning algorithm
            X = self.padding(X)

            # calculate the index for cropping
            index = torch.floor(self.pad_width + (loc + 1.) * torch.tensor(x_shape) / 2)
            res = None

            for x, l in zip(X, index):
                r = self.retina // 2
                l0, l1 = int(l[0].item()), int(l[1].item())

                # crop the image
                x_crop = x[l0 - r: l0 + r, l1 - r: l1 + r].unsqueeze(0)
                for i in range(1, self.depth):
                    r *= self.scale
                    x_tmp = self.scaler(x[l0 - r: l0 + r, l1 - r: l1 + r].unsqueeze(0))
                    x_crop = torch.cat((x_crop, x_tmp))

                res = torch.cat((res, x_crop)) if res is not None else x_crop

        if self.flatten:
            res = res.reshape(X.shape[0], -1)
        else:
            res = res.view(-1, self.depth, self.retina, self.retina)
        return res