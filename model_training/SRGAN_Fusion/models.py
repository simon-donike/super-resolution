import torch
from torch import nn
import torchvision
import math


class ConvolutionalBlock(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channe;s
        :param kernel_size: kernel size
        :param stride: stride
        :param batch_norm: include a BN layer?
        :param activation: Type of activation; None if none
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # A container that will hold the layers in this convolutional block
        layers = list()

        # A convolutional layer
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # A batch normalization (BN) layer, if wanted
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # An activation layer, if wanted
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # Put together the convolutional block as a sequence of the layers in this container
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation.

        :param input: input images, a tensor of size (N, in_channels, w, h)
        :return: output images, a tensor of size (N, out_channels, w, h)
        """
        #if input.shape[0]!=8:
        #    return(input)
        output = self.conv_block(input)  # (N, out_channels, w, h)

        return output


class SubPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :param kernel_size: kernel size of the convolution
        :param n_channels: number of input and output channels
        :param scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        Forward propagation.

        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class ResidualBlock(nn.Module):
    """
    A residual block, comprising two convolutional blocks with a residual connection across them.
    """

    def __init__(self, n_channels=64,kernel_size=3 ):
        """
        :param kernel_size: kernel size
        :param n_channels: number of input and output channels (same because the input must be added to the output)
        """
        super(ResidualBlock, self).__init__()

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # The second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        """
        Forward propagation.

        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: output images, a tensor of size (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        #if input.shape[0] !=8:
        #    return input
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output

class RecuversiveNet(nn.Module):
# FROM HRResNet to fuse temporal images
    def __init__(self):
        '''
        Args:
            config : dict, configuration file
        '''
        
        config= {
              "alpha_residual": False,
              "in_channels": 64,
              "num_layers" : 4,
              "kernel_size": 3}
        
        
        
        super(RecuversiveNet, self).__init__()

        self.input_channels = config["in_channels"]
        self.num_layers = config["num_layers"]
        self.alpha_residual = config["alpha_residual"]
        kernel_size = config["kernel_size"]
        padding = kernel_size // 2

        self.fuse = nn.Sequential(
            ResidualBlock(2 * self.input_channels, kernel_size),
            nn.Conv2d(in_channels=2 * self.input_channels, out_channels=self.input_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.PReLU())

    def forward(self, x, alphas):
        '''
        Fuses hidden states recursively.
        Args:
            x : tensor (B, L, C, W, H), hidden states
            alphas : tensor (B, L, 1, 1, 1), boolean indicator (0 if padded low-res view, 1 otherwise)
        Returns:
            out: tensor (B, C, W, H), fused hidden state
        '''
        
        batch_size, nviews, channels, width, heigth = x.shape
        parity = nviews % 2
        half_len = nviews // 2
        
        while half_len > 0:
            alice = x[:, :half_len] # first half hidden states (B, L/2, C, W, H)
            bob = x[:, half_len:nviews - parity] # second half hidden states (B, L/2, C, W, H)
            bob = torch.flip(bob, [1])

            alice_and_bob = torch.cat([alice, bob], 2)  # concat hidden states accross channels (B, L/2, 2*C, W, H)
            alice_and_bob = alice_and_bob.view(-1, 2 * channels, width, heigth)
            #print(alice_and_bob.shape)
            x = self.fuse(alice_and_bob)
            x = x.view(batch_size, half_len, channels, width, heigth)  # new hidden states (B, L/2, C, W, H)

            if self.alpha_residual: # skip connect padded views (alphas_bob = 0)
                alphas_alice = alphas[:, :half_len]
                alphas_bob = alphas[:, half_len:nviews - parity]
                alphas_bob = torch.flip(alphas_bob, [1])
                x = alice + alphas_bob * x
                alphas = alphas_alice

            nviews = half_len
            parity = nviews % 2
            half_len = nviews // 2

        return torch.mean(x, 1)
    
    
    
class SRResNet(nn.Module):
    """
    The SRResNet, as defined in the paper.
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        :param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        :param small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
        :param n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
        :param n_blocks: number of residual blocks
        :param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
        """
        super(SRResNet, self).__init__()

        # Scaling factor must be 2, 4, or 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "The scaling factor must be 2, 4, or 8!"

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        # A sequence of n_blocks residual blocks, each containing a skip-connection across the block
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

        # Another convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        # Upscaling is done by sub-pixel convolution, with each such block upscaling by a factor of 2
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])

        # The last convolutional block
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')
        
        self.fuse = RecuversiveNet()

    def forward(self, lr_imgs):
        """
        Forward prop.

        :param lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
        :return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        """
        # settings
        batch_size=2
        sen2_amount = 4
        
        # current input size b,l*c,w,h
        """ reshape to b*l,c,w,h """
        #print("LR shape before",lr_imgs.shape)
        if lr_imgs.shape != torch.Size([batch_size, 12, 75, 75]):
            print("ERROR! input shape to SRResNet is not 2,12,75,75")
            print(lr_imgs.shape)

        # iterate over batch sizes
        ls_tensors = []
        for i in range(batch_size):
            # get batch
            _ = lr_imgs[i]
            # extract individual images
            for j in range(sen2_amount):
                j = j*3
                __ = _[j:j+3]
                ls_tensors.append(__)
        # rearrange tensors in desired shape, stacking by images
        lr_imgs = torch.stack(ls_tensors)
        """ now b*l,c,w,h """
        #print("LR shape after",lr_imgs.shape)
        if lr_imgs.shape != torch.Size([8, 3, 75, 75]):
            print("ERROR! after initial reshape")
            print(lr_imgs.shape)
        
        _ = []
        for i in lr_imgs:
            # encoder, use Srresnet encoder below
            output = self.conv_block1(i.unsqueeze(0))  # (N, 3, w, h)
            residual = output  # (N, n_channels, w, h)
            output = self.residual_blocks(output)  # (N, n_channels, w, h)
            output = self.conv_block2(output)  # (N, n_channels, w, h)
            output = output + residual  # (N, n_channels, w, h)  - old
            # output size: b*l,c encoded,w,h
            _.append(output[0])
        output = torch.stack(_)
        #print("NEW",output.shape)
        
        # reshape to tensor (B, L, C encoded, W, H)
        # output here isnt consistent
        #print("output shape after encoding",output.shape)
        ls_tensors = []
        for i in range(batch_size):
            _ = output[i*batch_size:(i*batch_size) + (sen2_amount)]
            # now individual batches
            ls_ind = []
            for j in _:
                ls_ind.append(j)
            #print(torch.stack(ls_ind).shape)
            ls_tensors.append(torch.stack(ls_ind))
        output = torch.stack(ls_tensors)
        #print("output shape after reshaping",output.shape)
        if output.shape != torch.Size([2, 4, 64, 75, 75]):
            print("ERROR! after output reshaped")
            print(output.shape)
        
        #print("shape before fusing",output.shape)
        # input : tensor (B, L, C, W, H), hidden states
        recursive_layer = self.fuse(output, alphas="placeholder") 
        # output : tensor (B, C, W, H), fused hidden state
        #print("shape after fusing",recursive_layer.shape)
        
        # back to BxCxWxH
        output = self.subpixel_convolutional_blocks(recursive_layer)  # (N, c encoded, w * scaling factor, h * scaling factor)
        # 
        #print("shape before SR",output.shape)
        sr_imgs = self.conv_block3(output)  # (N, 3, w * scaling factor, h * scaling factor)
        #print("shape after SR",sr_imgs.shape)
        # here is nx64

        return sr_imgs


    
class Generator(nn.Module):
    """
    The generator in the SRGAN, as defined in the paper. Architecture identical to the SRResNet.
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        :param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        :param small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
        :param n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
        :param n_blocks: number of residual blocks
        :param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
        """
        super(Generator, self).__init__()

        # The generator is simply an SRResNet, as above
        self.net = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                            n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)

    def initialize_with_srresnet(self, srresnet_checkpoint):
        """
        Initialize with weights from a trained SRResNet.

        :param srresnet_checkpoint: checkpoint filepath
        """
        srresnet = torch.load(srresnet_checkpoint)['model']
        self.net.load_state_dict(srresnet.state_dict())

        print("\nLoaded weights from pre-trained SRResNet.\n")

    def forward(self, lr_imgs):
        """
        Forward prop.

        :param lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
        :return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        """
        sr_imgs = self.net(lr_imgs)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return sr_imgs


class Discriminator(nn.Module):
    """
    The discriminator in the SRGAN, as defined in the paper.
    """

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
        """
        :param kernel_size: kernel size in all convolutional blocks
        :param n_channels: number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
        :param n_blocks: number of convolutional blocks
        :param fc_size: size of the first fully connected layer
        """
        super(Discriminator, self).__init__()

        in_channels = 3

        # A series of convolutional blocks
        # The first, third, fifth (and so on) convolutional blocks increase the number of channels but retain image size
        # The second, fourth, sixth (and so on) convolutional blocks retain the same number of channels but halve image size
        # The first convolutional block is unique because it does not employ batch normalization
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation='LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # An adaptive pool layer that resizes it to a standard size
        # For the default input size of 96 and 8 convolutional blocks, this will have no effect
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(1024, 1)

        # Don't need a sigmoid layer because the sigmoid operation is performed by PyTorch's nn.BCEWithLogitsLoss()

    def forward(self, imgs):
        """
        Forward propagation.

        :param imgs: high-resolution or super-resolution images which must be classified as such, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: a score (logit) for whether it is a high-resolution image, a tensor of size (N)
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit


class TruncatedVGG19(nn.Module):
    """
    A truncated VGG19 network, such that its output is the 'feature map obtained by the j-th convolution (after activation)
    before the i-th maxpooling layer within the VGG19 network', as defined in the paper.

    Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss.
    """

    def __init__(self, i, j):
        """
        :param i: the index i in the definition above
        :param j: the index j in the definition above
        """
        super(TruncatedVGG19, self).__init__()

        # Load the pre-trained VGG19 available in torchvision
        vgg19 = torchvision.models.vgg19(pretrained=True)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # Iterate through the convolutional section ("features") of the VGG19
        for layer in vgg19.features.children():
            truncate_at += 1

            # Count the number of maxpool layers and the convolutional layers after each maxpool
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # Break if we reach the jth convolution after the (i - 1)th maxpool
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # Check if conditions were satisfied
        assert maxpool_counter == i - 1 and conv_counter == j, "One or both of i=%d and j=%d are not valid choices for the VGG19!" % (
            i, j)

        # Truncate to the jth convolution (+ activation) before the ith maxpool layer
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):
        """
        Forward propagation
        :param input: high-resolution or super-resolution images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: the specified VGG19 feature map, a tensor of size (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output
