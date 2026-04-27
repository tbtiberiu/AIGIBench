import argparse
import importlib
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
from tqdm import tqdm

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
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True


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


def calculate_accuracy(id_conf, ood_conf, use_optimal=False):
    """Calculates accuracy either using optimal threshold or fixed 0.5 threshold.
    Returns (total_accuracy, real_accuracy)"""
    if use_optimal:
        all_conf = np.concatenate([id_conf, ood_conf])
        labels = np.concatenate([np.ones(len(id_conf)), np.zeros(len(ood_conf))])

        fpr, tpr, thresholds = metrics.roc_curve(labels, all_conf)

        # Number of positive (ID) and negative (OOD) samples
        P = len(id_conf)
        N = len(ood_conf)

        # Accuracy = (TP + TN) / (P + N)
        accuracies = (tpr * P + (1 - fpr) * N) / (P + N)
        best_idx = np.argmax(accuracies)

        return accuracies[best_idx], tpr[best_idx]
    else:
        # Use fixed 0.5 threshold
        r_acc = (id_conf >= 0.5).mean()
        f_acc = (ood_conf < 0.5).mean()

        P = len(id_conf)
        N = len(ood_conf)
        total_acc = (r_acc * P + f_acc * N) / (P + N)
        return total_acc, r_acc


def print_table_header():
    print('\n' + '=' * 95)
    print(
        f'{"Dataset":<25} | {"Similarity":<10} | {"Accuracy":<10} | {"AUC":<10} | {"AP":<10} | {"FPR95":<10}'
    )
    print('-' * 95)


def print_legend(use_optimal_threshold=False):
    print('\nLegend:')
    print(
        '- Similarity: The average detector score indicating the predicted probability of the image being Real (ID).'
    )
    if use_optimal_threshold:
        print(
            '- Accuracy: The maximum possible classification accuracy achieved by choosing an optimal threshold.'
        )
        print('  (Calculated pairwise between Real and the specific Generated dataset)')
    else:
        print(
            '- Accuracy: The percentage of correctly classified images using a 0.5 threshold.'
        )
        print(
            '  (For Real: score >= 0.5 is correct; For Generated: score < 0.5 is correct)'
        )
    print('- AUC: Area Under the Receiver Operating Characteristic Curve (ROC AUC).')
    print('- AP: Average Precision, summarizing the precision-recall curve.')
    print('- FPR95: False Positive Rate when the True Positive Rate (TPR) is at 95%.')


def print_evaluation_results(similarities, datasets, use_optimal_threshold=False):
    id_confi = similarities[0]
    id_name = datasets[0]

    # Pre-calculate metrics to get average Real accuracy
    ood_results = []
    id_acc_scores = []

    for ood_confi, dataset_name in zip(similarities[1:], datasets[1:]):
        auroc, fpr_95 = calculate_auc_metrics(id_confi, ood_confi)
        aver_p = calculate_average_precision(id_confi, ood_confi)
        acc, r_acc = calculate_accuracy(
            id_confi, ood_confi, use_optimal=use_optimal_threshold
        )
        sim = ood_confi.mean()

        ood_results.append(
            {
                'name': dataset_name,
                'sim': sim,
                'acc': acc,
                'auc': auroc,
                'ap': aver_p,
                'fpr': fpr_95,
            }
        )
        id_acc_scores.append(r_acc)

    avg_id_acc = np.mean(id_acc_scores) if id_acc_scores else 0.0

    print_table_header()

    # Real Section
    id_sim = id_confi.mean()
    print(
        f'{id_name:<25} | {id_sim:<10.4f} | {avg_id_acc:<10.4f} | {"-":<10} | {"-":<10} | {"-":<10}'
    )
    print(
        f'{"Average Real":<25} | {id_sim:<10.4f} | {avg_id_acc:<10.4f} | {"-":<10} | {"-":<10} | {"-":<10}'
    )
    print('-' * 95)

    # Generated Section
    auc_scores, ap_scores, fpr_scores, sim_scores, acc_scores = [], [], [], [], []

    for res in ood_results:
        print(
            f'{res["name"]:<25} | {res["sim"]:<10.4f} | {res["acc"]:<10.4f} | {res["auc"]:<10.4f} | {res["ap"]:<10.4f} | {res["fpr"]:<10.4f}'
        )
        sim_scores.append(res['sim'])
        acc_scores.append(res['acc'])
        auc_scores.append(res['auc'])
        ap_scores.append(res['ap'])
        fpr_scores.append(res['fpr'])

    avg_sim = np.mean(sim_scores)
    avg_acc = np.mean(acc_scores)
    avg_auc = np.mean(auc_scores)
    avg_ap = np.mean(ap_scores)
    avg_fpr = np.mean(fpr_scores)

    print('-' * 95)
    print(
        f'{"Average Generated":<25} | {avg_sim:<10.4f} | {avg_acc:<10.4f} | {avg_auc:<10.4f} | {avg_ap:<10.4f} | {avg_fpr:<10.4f}'
    )
    print('=' * 95)


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


class DetectorWrapper:
    def __init__(self):
        self.model = None
        self.transform = None
        self.use_optimal_threshold = False

    @torch.inference_mode()
    def detect(self, data):
        # Default: sigmoid probability where high = fake if output dim is 1
        # If output dim is 2, use softmax[:, 1]
        out = self.model(data)
        if out.shape[1] == 1:
            return out.sigmoid().flatten()
        else:
            return out.softmax(dim=1)[:, 1].flatten()

    def _setup_path(self, path):
        """Append path to sys.path and clear related cached modules to avoid collisions."""
        if path not in sys.path:
            sys.path.insert(0, path)
        # Clear modules that might conflict (e.g., 'networks', 'models', 'utils', 'model')
        # Use more surgical matching to avoid deleting things like 'dataclasses'
        conflicting = ('networks', 'models', 'utils', 'data', 'model', 'util')
        to_delete = [
            m
            for m in list(sys.modules.keys())
            if m in conflicting or any(m.startswith(c + '.') for c in conflicting)
        ]
        for m in to_delete:
            del sys.modules[m]


class AIDE_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/AIDE-main')
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

    @torch.inference_mode()
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
        super().__init__()
        self._setup_path('detector_codes/C2P-CLIP-DeepfakeDetection-main')
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


class C2P_DINOv2_Detector(DetectorWrapper):
    def __init__(self, model_path=None):
        super().__init__()
        self._setup_path('detector_codes/C2P-DINOv2-main')
        from model import C2P_DINOv2_Model

        self.model = C2P_DINOv2_Model()
        if model_path is not None:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(
                state_dict['model_state_dict']
                if 'model_state_dict' in state_dict
                else state_dict,
                strict=False,
            )
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def detect(self, img):
        return self.model.detect(img)


class C2P_DINOv3_Detector(DetectorWrapper):
    def __init__(self, model_path=None):
        super().__init__()
        self._setup_path('detector_codes/C2P-DINOv3-main')
        from model import C2P_DINOv3_Model

        checkpoint = None
        checkpoint_args = {}
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                checkpoint_args = checkpoint.get('args', {}) or {}

        model_kwargs = {
            'lora_r': checkpoint_args.get('lora_r', 16),
            'lora_alpha': checkpoint_args.get('lora_alpha', 32),
            'lora_dropout': checkpoint_args.get('lora_dropout', 0.5),
            'unfreeze_last_blocks': checkpoint_args.get('unfreeze_last_blocks', 0),
            'image_size': checkpoint_args.get('image_size', 256),
            'forensic_dim': checkpoint_args.get('forensic_dim', 256),
        }
        lora_target_modules = checkpoint_args.get('lora_target_modules')
        if isinstance(lora_target_modules, str):
            model_kwargs['lora_target_modules'] = [
                module.strip()
                for module in lora_target_modules.split(',')
                if module.strip()
            ]
        elif lora_target_modules:
            model_kwargs['lora_target_modules'] = lora_target_modules

        self.model = C2P_DINOv3_Model(**model_kwargs)
        if checkpoint is not None:
            self.model.load_state_dict(
                checkpoint['model_state_dict']
                if 'model_state_dict' in checkpoint
                else checkpoint,
                strict=False,
            )
        self.model.to(DEVICE).eval()

        size = model_kwargs['image_size']
        resize_size = max(int(round(size * 1.15)), size)
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def detect(self, img):
        return self.model.detect(img)


class CLIPDetection_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/CLIPDetection-main')
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
        super().__init__()
        self._setup_path('detector_codes/CNNDetection-master')
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
        super().__init__()
        self._setup_path('detector_codes/DFFreq-main')
        import networks.resnet as resnet_module

        importlib.reload(resnet_module)
        self.model = resnet_module.resnet50(num_classes=1)

        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state_dict)
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


class Effort_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/Effort-AIGI-Detection')
        from models.clip_models import ClipModel

        opt = argparse.Namespace(use_svd=True)
        self.model = ClipModel(
            name='openai/clip-vit-large-patch14', opt=opt, num_classes=1
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)
        )
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


class FreqNet_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/FreqNet-DeepfakeDetection-main')
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
        super().__init__()
        self._setup_path('detector_codes/Gram-Net-main')
        import networks.resnet as resnet_module

        importlib.reload(resnet_module)
        # GramNet weights are ResNet-18
        self.model = resnet_module.resnet18(num_classes=1)

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
        super().__init__()
        # LGrad weights match standard ResNet-50
        self._setup_path('detector_codes/LGrad-master/CNNDetection')
        import networks.resnet as resnet_module

        importlib.reload(resnet_module)
        self.model = resnet_module.resnet50(num_classes=1)

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
        super().__init__()
        self._setup_path('detector_codes/NPR-DeepfakeDetection-main')
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
        super().__init__()
        self.use_optimal_threshold = True
        self._setup_path('detector_codes/RIGID-main')
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

    @torch.inference_mode()
    def detect(self, data):
        return self.model.detect(data)


class Resnet50_Detector(DetectorWrapper):
    def __init__(self, model_path):
        super().__init__()
        self._setup_path('detector_codes/Resnet50-main')
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
        super().__init__()
        self._setup_path('detector_codes/SAFE-main')
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
            'C2P-DINOv2',
            'C2P-DINOv3',
            'CLIPDetection',
            'CNNDetection',
            'DFFreq',
            'Effort',
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
        choices=['AIGC-Detection-Benchmark', 'MS-COCOAI', '140k-Real-and-Fake-Faces'],
        help='HuggingFace dataset to evaluate on',
    )
    parser.add_argument(
        '--limit', type=int, default=1000, help='Limit samples per subset for speed'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16, help='Batch size for evaluation'
    )
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
        '140k-Real-and-Fake-Faces': {
            'path': 'TheKernel01/140k-Real-and-Fake-Faces',
            'mapping': {1: 'StyleGAN'},
        },
    }

    weight_mapping = {
        'AIDE': './AIGIBench_models/AIDE-main/model_epoch_best.pth',
        'C2P-CLIP': './AIGIBench_models/C2P-CLIP-DeepfakeDetection-main/model_epoch_best.pth',
        'C2P-DINOv2': './AIGIBench_models/C2P-DINOv2-main/model_epoch_best.pth',
        'C2P-DINOv3': './AIGIBench_models/C2P-DINOv3-main/model_epoch_best.pth',
        'CLIPDetection': './AIGIBench_models/CLIPDetection-main/model_epoch_best.pth',
        'CNNDetection': './AIGIBench_models/CNNDetection-master/model_epoch_best.pth',
        'DFFreq': './AIGIBench_models/DFFreq-main/model_epoch_best.pth',
        'Effort': './AIGIBench_models/Effort-AIGI-Detection/model_epoch_best.pth',
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
        'C2P-DINOv2': C2P_DINOv2_Detector,
        'C2P-DINOv3': C2P_DINOv3_Detector,
        'CLIPDetection': CLIPDetection_Detector,
        'CNNDetection': CNNDetection_Detector,
        'DFFreq': DFFreq_Detector,
        'Effort': Effort_Detector,
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
            dataset_obj,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        scores = []
        total = 0

        # Calculate expected number of batches based on samples limit
        total_batches = (
            min(len(dataset_obj), args.limit) + args.batch_size - 1
        ) // args.batch_size
        pbar = tqdm(
            loader, total=total_batches, desc=f'Evaluating {dataset_name}', leave=False
        )

        for i, (imgs, _) in enumerate(pbar):
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
            f'{dataset_name:<25}, Count: {len(scores)}, Similarity: {scores.mean():.4f}'
        )
        sim_datasets.append(scores.numpy())

    print('\n' + '=' * 95)
    print(f'Results for {args.model} on {args.dataset}:')
    print('=' * 95)
    print_evaluation_results(
        sim_datasets,
        test_datasets,
        use_optimal_threshold=detector.use_optimal_threshold,
    )
    print_legend(use_optimal_threshold=detector.use_optimal_threshold)


if __name__ == '__main__':
    main()
