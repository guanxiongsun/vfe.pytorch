# The pixel level on the ChannelMapper's 512-channel output instead of the
# backbone's 2048-channel map: the RPN and the RoI head still read the
# enhanced map, at a sixteenth of the attention's projection cost.
_base_ = ['./mamba_pix_r101_dc5_3x.py']

model = dict(pixel=dict(in_channels=512, position='neck'))
