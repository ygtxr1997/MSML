from backbones.osb.unet import unet


def _prepare_fm(self, fm_op, fm_layers):
    self.heights = [64, 32, 16, 8]
    self.f_channels = [48, 96, 192, 128]
    self.s_channels = [18, 18, 18, 18]

    fm_ops = []
    for i in range(4):
        if i in fm_layers:
            fm_ops.append(fm_op(
                height=self.heights[i],
                width=self.heights[i],
                channel_f=self.f_channels[i]
            ))
        else:
            fm_ops.append(fm_op())
    self.fm_ops = fm_ops