import torch
from networks.drbn import DRBN

def load_drbn_model(ckpt_path, device=None):
    """
    載入 DRBN 模型與指定權重檔

    :param ckpt_path: 權重檔案路徑
    :param device: 'cuda' or 'cpu'
    :return: 已載入權重的 DRBN 模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRBN().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model
