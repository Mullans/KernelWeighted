# Names of the post-bottleneck layers for the UNet model
POST_BOTTLENECK_LAYERS = [
    'decoder1.dec1conv2',
    'decoder1.dec1conv1',
    'upconv1',
    'decoder2.dec2conv2',
    'decoder2.dec2conv1',
    'upconv2',
    'decoder3.dec3conv2',
    'decoder3.dec3conv1',
    'upconv3',
    'decoder4.dec4conv2',
    'decoder4.dec4conv1',
    'upconv4'
]
BOTTLENECK = ['bottleneck.bottleneckconv2']
LAST_CONV = ['decoder1.dec1conv2']
