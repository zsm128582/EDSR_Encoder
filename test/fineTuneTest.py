import torch
from models.classifitor import Classifitor
from models.myEncoder import myEncoder
from timm.models.layers import trunc_normal_
    # args:
    #     encoder_spec:
    #     name: edsr-baseline
    #     args:
    #         no_upsampling: true
    #     width: 256
    #     blocks: 16
# encoder_spec:
#       name: edsr-baseline
#       args:
#         no_upsampling: 


def interpolate_pos_embed(model , checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]

        # 这里需要改成实际输入的点数
        pixelNum = model.width ** 2
        # extra就是cls token的个数，就是总的token树剪掉像素点的个数，剩下的就是cls token的数量
        num_extra_tokens = model.pos_embed.shape[-2] - pixelNum
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(pixelNum ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed 


def test():
        
    encoder_spec = {}
    encoder_spec["name"] = "edsr-baseline"

    encoder_spec["args"] = {
        "no_upsampling" : True
    }
    # # model = Classifitor(encoder_spec)
    # model = myEncoder(encoder_spec)
    # print(model)

    model = Classifitor(encoder_spec,width=256)

    checkPointPath = "result/finetuneTest/epoch-last.pth"
    checkpoint = torch.load(checkPointPath,map_location='cpu')
    checkpoint_model = checkpoint['model']['sd']
    state_dict = model.state_dict()
    decoderKeys = []
    for key in checkpoint_model.keys():
        if(key.startswith('decoder')): 
            decoderKeys.append(key)
    for key in decoderKeys:
        del checkpoint_model[key]
    interpolate_pos_embed(model , checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    
    trunc_normal_(model.head.weight, std=2e-5)
    # 为什么这里只是初始化了一下head 的权重呢？
    pass

if __name__ == "__main__":
    test()