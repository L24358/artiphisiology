import torch

# def get_output_shape(model):
#     for i, layer in enumerate(model.features):
#         layer.kernel_size

def get_output_shape(model, image_dim):
    out = model(torch.rand(*(image_dim)))
    import pdb; pdb.set_trace()
    return model(torch.rand(*(image_dim))).data.shape

if __name__ == "__main__":
    import spectools.models.models as mdl
    model = mdl.get_vgg16()
    print(get_output_shape(model, (3, 227, 227)))