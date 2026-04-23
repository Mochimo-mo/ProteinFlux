import torch


def get_offsets(ref_frame, rigids):
    """Compute relative rigid transforms from a reference frame to all frames."""
    B, T, L = rigids.shape
    if T > 500000:
        offsets1 = ref_frame.invert().compose(rigids[:, :500000]).to_tensor_7()
        offsets2 = ref_frame.invert().compose(rigids[:, 500000:]).to_tensor_7()
        return torch.cat([offsets1, offsets2], 1)
    else:
        return ref_frame.invert().compose(rigids).to_tensor_7()
