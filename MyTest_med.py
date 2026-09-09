import os
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from tqdm import tqdm

from lib.pranet import PraNet_V2
from utils.dataloader import test_dataset


# ---------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------

TEST_SIZE = 352

DATASETS = [
    'CVC-300',
    'CVC-ClinicDB',
    'Kvasir',
    'ETIS-LaribPolypDB'
]

CHECKPOINT = './snapshots/PraNet-V2/RES-V2.pth'
RESULTS_DIR = './results/PraNet-V2'


# ---------------------------------------------------------
# Modeli yükle
# ---------------------------------------------------------

def load_model():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print('========================================')
    print('PraNet-V2 başlatılıyor...')
    print('========================================')
    print('Cihaz:', device)

    model = PraNet_V2(num_class=1)

    print('Model oluşturuldu.')

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint,
        strict=False
    )

    print('RES-V2.pth yüklendi.')

    model = model.to(device)
    model.eval()

    print('MODEL HAZIR.')
    print()

    return model, device


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

def test_dataset_model(model, device, dataset_name):

    data_path = './data/TestDataset/{}/'.format(dataset_name)

    image_root = os.path.join(data_path, 'images')
    gt_root = os.path.join(data_path, 'masks')

    save_path = os.path.join(
        RESULTS_DIR,
        dataset_name
    )

    os.makedirs(save_path, exist_ok=True)

    print()
    print('========================================')
    print('Dataset:', dataset_name)
    print('========================================')

    test_loader = test_dataset(
        image_root + '/',
        gt_root + '/',
        TEST_SIZE
    )

    print('Görüntü sayısı:', test_loader.size)

    with torch.no_grad():

        for i in tqdm(
            range(test_loader.size),
            desc=dataset_name
        ):

            image, gt, name = test_loader.load_data()

            gt = np.asarray(
                gt,
                np.float32
            )

            gt /= (gt.max() + 1e-8)

            image = image.to(device)

            # PraNet-V2 çıktıları
            (
                res2,
                res3,
                res4,
                res5,
                res2_bg,
                res3_bg,
                res4_bg,
                res5_bg
            ) = model(image)

            # Çok seviyeli çıktıları birleştir
            res = res2 + res3 + res4 + res5

            # Orijinal görüntü boyutuna getir
            res = F.interpolate(
                res,
                size=gt.shape,
                mode='bilinear',
                align_corners=False
            )

            # Sigmoid
            res = res.sigmoid()

            # CPU -> NumPy
            res = res.detach().cpu().numpy().squeeze()

            # 0-1 arasına normalize et
            res = (
                res - res.min()
            ) / (
                res.max() - res.min() + 1e-8
            )

            # 0-255
            res_uint8 = (
                res * 255
            ).astype(np.uint8)

            # Kaydet
            imageio.imwrite(
                os.path.join(
                    save_path,
                    name
                ),
                res_uint8
            )

    print()
    print('TAMAMLANDI:', dataset_name)
    print('Sonuç klasörü:', save_path)


# ---------------------------------------------------------
# Ana program
# ---------------------------------------------------------

if __name__ == '__main__':

    model, device = load_model()

    for dataset_name in DATASETS:

        test_dataset_model(
            model,
            device,
            dataset_name
        )

    print()
    print('========================================')
    print('BÜTÜN TESTLER TAMAMLANDI')
    print('========================================')
    print()
    print('Sonuçlar:')
    print(RESULTS_DIR)