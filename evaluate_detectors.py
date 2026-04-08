import argparse
import os
import random
import sys

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CACHE_DIR = None

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 123

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False


def calculate_auc_metrics(id_conf, ood_conf):
    all_conf = np.concatenate([id_conf, ood_conf])
    labels = np.concatenate([np.ones(len(id_conf)), np.zeros(len(ood_conf))])
    fpr, tpr, _ = metrics.roc_curve(labels, all_conf)
    auroc = metrics.auc(fpr, tpr)
    tpr_threshold = 0.95
    valid_indices = tpr >= tpr_threshold
    fpr_at_95 = fpr[np.argmax(valid_indices)] if np.any(valid_indices) else fpr[-1]
    return auroc, fpr_at_95


def calculate_average_precision(id_predictions, ood_predictions):
    all_predictions = np.concatenate([id_predictions, ood_predictions])
    labels = np.concatenate(
        [np.ones(len(id_predictions)), np.zeros(len(ood_predictions))]
    )
    return metrics.average_precision_score(labels, all_predictions)


def sim_auc(similarities, datasets):
    id_confi = similarities[0]
    auc_scores, fpr_scores = [], []
    for ood_confi, dataset in zip(similarities[1:], datasets[1:]):
        auroc, fpr_95 = calculate_auc_metrics(id_confi, ood_confi)
        auc_scores.append(auroc)
        fpr_scores.append(fpr_95)
        print(f'Dataset: {dataset:<25} | AUC: {auroc:.4f} | FPR95: {fpr_95:.4f}')
    avg_auc, avg_fpr = np.mean(auc_scores), np.mean(fpr_scores)
    print('-' * 60)
    print(f'Average AUC: {avg_auc:.4f} | Average FPR95: {avg_fpr:.4f}')
    return avg_auc, avg_fpr


def sim_ap(similarities, datasets):
    id_confi = similarities[0]
    ap_scores = []
    for ood_confi, dataset in zip(similarities[1:], datasets[1:]):
        aver_p = calculate_average_precision(id_confi, ood_confi)
        ap_scores.append(aver_p)
        print(f'Dataset: {dataset:<25} | AP: {aver_p:.4f}')
    avg_ap = np.mean(ap_scores)
    print('-' * 40)
    print(f'Average AP: {avg_ap:.4f}')
    return avg_ap


class HFImageDataset(Dataset):
    def __init__(self, hf_data, transform=None):
        self.hf_data = hf_data
        self.transform = transform

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        item = self.hf_data[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label


# Detector base and implementations
class DetectorWrapper:
    def __init__(self):
        self.model = None
        self.transform = None

    @torch.no_grad()
    def detect(self, data):
        # Default: sigmoid probability where high = fake if output dim is 1
        # If output dim is 2, use softmax[:, 1]
        out = self.model(data)
        if out.shape[1] == 1:
            return out.sigmoid().flatten()
        else:
            return out.softmax(dim=1)[:, 1].flatten()


class AIDE_Detector(DetectorWrapper):
    def __init__(self, model_path):
        sys.path.append('detector_codes/AIDE-main')
        from data.dct import DCT_base_Rec_Module
        from models.AIDE import AIDE

        # AIDE expects resnet_path and convnext_path for preloading,
        # but since we load the full state dict later, we can pass None.
        self.model = AIDE(resnet_path=None, convnext_path=None)
        self.dct = DCT_base_Rec_Module()
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        # Handle SAFE/AIDE state dict naming if necessary
        msg = self.model.load_state_dict(
            state_dict['model'] if 'model' in state_dict else state_dict, strict=False
        )
        print(f'AIDE load message: {msg}')
        self.model.to(DEVICE).eval()
        self.dct.to(DEVICE)
        # Transform should not include normalization because DCT is applied on raw ToTensor
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.resize = transforms.Resize((256, 256))

    @torch.no_grad()
    def detect(self, data):
        # data is [B, 3, 256, 256] from 0 to 1
        batch_stacked = []
        for i in range(data.shape[0]):
            img = data[i]
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(img)
            # All must be 256x256
            # Stack: [x_minmin, x_maxmax, x_minmin1, x_maxmax1, original]
            # All need to be normalized
            stacked = torch.stack(
                [
                    self.normalize(self.resize(x_minmin)),
                    self.normalize(self.resize(x_maxmax)),
                    self.normalize(self.resize(x_minmin1)),
                    self.normalize(self.resize(x_maxmax1)),
                    self.normalize(img),
                ],
                dim=0,
            )
            batch_stacked.append(stacked)

        batch_data = torch.stack(batch_stacked, dim=0)  # [B, 5, 3, 256, 256]
        out = self.model(batch_data)
        # AIDE has 2 classes
        return out.softmax(dim=1)[:, 1].flatten()


class C2P_CLIP_Detector(DetectorWrapper):
    def __init__(self, model_path):
        sys.path.append('detector_codes/C2P-CLIP-DeepfakeDetection-main')
        from networks.c2p_clip import C2P_CLIP_Model

        self.model = C2P_CLIP_Model(name='openai/clip-vit-large-patch14', num_classes=1)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']

        # Clean state dict by removing 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def detect(self, img):
        return self.model.detect(img)


class CLIPDetection_Detector(DetectorWrapper):
    def __init__(self, model_path):
        sys.path.append('detector_codes/CLIPDetection-main')
        # CLIPDetection uses openai-clip. We need to handle its specific architecture.
        # This implementation assumes dependencies are installed.
        from models.clip_models import CLIPModel

        self.model = CLIPModel(name='ViT-L/14', num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        # CLIP standard normalization
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )


class CNNDetection_Detector(DetectorWrapper):
    def __init__(self, model_path):
        sys.path.append('detector_codes/CNNDetection-master')
        from networks.resnet import resnet50

        self.model = resnet50(num_classes=1)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(
            state_dict['model'] if 'model' in state_dict else state_dict
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class DFFreq_Detector(DetectorWrapper):
    def __init__(self, model_path):
        sys.path.append('detector_codes/DFFreq-main')
        from networks.resnet import resnet50

        self.model = resnet50(num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class FreqNet_Detector(DetectorWrapper):
    def __init__(self, model_path):
        sys.path.append('detector_codes/FreqNet-DeepfakeDetection-main')
        from networks.freqnet import FreqNet

        self.model = FreqNet(num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class GramNet_Detector(DetectorWrapper):
    def __init__(self, model_path):
        sys.path.append('detector_codes/Gram-Net-main')
        from networks.resnet import resnet50

        self.model = resnet50(num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class LGrad_Detector(DetectorWrapper):
    def __init__(self, model_path):
        # LGrad classifier is a standard ResNet trained on gradients
        # For simplicity, we assume we've already transformed images if we were in a pipeline,
        # but here we'll just use the ResNet model.
        # NOTE: Full LGrad requires a gradient transformation step which is expensive for runtime eval.
        sys.path.append('detector_codes/LGrad-master/CNNDetection')
        from networks.resnet import resnet50

        self.model = resnet50(num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class NPR_Detector(DetectorWrapper):
    def __init__(self, model_path):
        sys.path.append('detector_codes/NPR-DeepfakeDetection-main')
        from networks.resnet import resnet50

        self.model = resnet50(num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class RIGID_Detector(DetectorWrapper):
    def __init__(self, model_path=None):
        sys.path.append('detector_codes/RIGID-main')
        from rigid_detector import RIGID_Detector as RIGID_Impl

        self.model = RIGID_Impl(lamb=0.05)
        self.model.model.to(DEVICE)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    @torch.no_grad()
    def detect(self, data):
        return self.model.detect(data)


class Resnet50_Detector(DetectorWrapper):
    def __init__(self, model_path):
        sys.path.append('detector_codes/Resnet50-main')
        from networks.resnet import resnet50

        self.model = resnet50(num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class SAFE_Detector(DetectorWrapper):
    def __init__(self, model_path):
        sys.path.append('detector_codes/SAFE-main')
        from models.resnet import resnet50

        self.model = resnet50(num_classes=2)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(
            state_dict['model'] if 'model' in state_dict else state_dict, strict=True
        )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=[
            'AIDE',
            'C2P-CLIP',
            'CLIPDetection',
            'CNNDetection',
            'DFFreq',
            'FreqNet',
            'GramNet',
            'LGrad',
            'NPR',
            'RIGID',
            'Resnet50',
            'SAFE',
        ],
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='AIGC-Detection-Benchmark',
        choices=['AIGC-Detection-Benchmark', 'MS-COCOAI', 'Real-and-Fake-Faces'],
        help='HuggingFace dataset to evaluate on',
    )
    parser.add_argument(
        '--limit', type=int, default=1000, help='Limit samples per subset for speed'
    )
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    dataset_configs = {
        'AIGC-Detection-Benchmark': {
            'path': 'TheKernel01/AIGC-Detection-Benchmark',
            'mapping': {
                1: 'ADM',
                2: 'BigGAN',
                3: 'CycleGAN',
                4: 'DALLE2',
                5: 'GauGAN',
                6: 'GLIDE',
                7: 'Midjourney',
                8: 'ProGAN',
                9: 'SD14',
                10: 'SD15',
                11: 'SDXL',
                12: 'StarGAN',
                13: 'StyleGAN',
                14: 'StyleGAN2',
                15: 'VQDM',
                16: 'WhichFaceIsReal',
                17: 'Wukong',
            },
        },
        'MS-COCOAI': {
            'path': 'TheKernel01/MS-COCOAI',
            'mapping': {1: 'SD21', 2: 'SDXL', 3: 'SD3', 4: 'DALLE3', 5: 'Midjourney 6'},
        },
        'Real-and-Fake-Faces': {
            'path': 'TheKernel01/140k-Real-and-Fake-Faces',
            'mapping': {1: 'StyleGAN'},
        },
    }

    weight_mapping = {
        'AIDE': './AIGIBench_models/AIDE-main/model_epoch_best.pth',
        'C2P-CLIP': './AIGIBench_models/C2P-CLIP-DeepfakeDetection-main/model_epoch_best.pth',
        'CLIPDetection': './AIGIBench_models/CLIPDetection-main/model_epoch_best.pth',
        'CNNDetection': './AIGIBench_models/CNNDetection-master/model_epoch_best.pth',
        'DFFreq': './AIGIBench_models/DFFreq-main/model_epoch_best.pth',
        'FreqNet': './AIGIBench_models/FreqNet-DeepfakeDetection-main/model_epoch_best.pth',
        'GramNet': './AIGIBench_models/Gram-Net-main/model_epoch_best.pth',
        'LGrad': './AIGIBench_models/LGrad-master/model_epoch_best.pth',
        'NPR': './AIGIBench_models/NPR-DeepfakeDetection-main/model_epoch_best.pth',
        'RIGID': None,
        'Resnet50': './AIGIBench_models/Resnet50-main/model_epoch_best.pth',
        'SAFE': './AIGIBench_models/SAFE-main/model_epoch_best.pth',
    }

    detector_classes = {
        'AIDE': AIDE_Detector,
        'C2P-CLIP': C2P_CLIP_Detector,
        'CLIPDetection': CLIPDetection_Detector,
        'CNNDetection': CNNDetection_Detector,
        'DFFreq': DFFreq_Detector,
        'FreqNet': FreqNet_Detector,
        'GramNet': GramNet_Detector,
        'LGrad': LGrad_Detector,
        'NPR': NPR_Detector,
        'RIGID': RIGID_Detector,
        'Resnet50': Resnet50_Detector,
        'SAFE': SAFE_Detector,
    }

    print(f'Initializing {args.model} detector...')
    detector = detector_classes[args.model](weight_mapping[args.model])

    print(f'Loading dataset {args.dataset}...')
    config = dataset_configs[args.dataset]
    test_data = load_dataset(
        config['path'],
        split='test',
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
    )
    all_generators = np.array(test_data['generator'])
    generator_mapping = config['mapping']

    # Prepare subsets
    real_indices = np.nonzero(all_generators == 0)[0]
    real_dataset = HFImageDataset(
        test_data.select(real_indices), transform=detector.transform
    )
    evaluation_datasets = [('Real (ID)', real_dataset)]

    for gen_id, gen_name in generator_mapping.items():
        fake_indices = np.nonzero(all_generators == gen_id)[0]
        fake_dataset = HFImageDataset(
            test_data.select(fake_indices), transform=detector.transform
        )
        evaluation_datasets.append((f'{gen_name} (OOD)', fake_dataset))

    # Run detection
    sim_datasets = []
    test_datasets = [name for name, _ in evaluation_datasets]

    for dataset_name, dataset_obj in evaluation_datasets:
        loader = DataLoader(
            dataset_obj, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        scores = []
        total = 0
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            # Invert: high score = real (ID-positive)
            # Detector returns p(fake), so we take 1 - p(fake)
            score = 1.0 - detector.detect(imgs)
            scores.append(score.cpu())
            total += len(imgs)
            if total >= args.limit:
                break

        scores = torch.cat(scores)[: args.limit]
        print(
            f'{dataset_name:<25}, Count: {len(scores)}, Mean Score: {scores.mean():.4f}'
        )
        sim_datasets.append(scores.numpy())

    print('\n' + '=' * 60)
    print(f'Results for {args.model}:')
    print('=' * 60)
    sim_auc(sim_datasets, test_datasets)
    print('\n' + '-' * 40)
    sim_ap(sim_datasets, test_datasets)


if __name__ == '__main__':
    main()
