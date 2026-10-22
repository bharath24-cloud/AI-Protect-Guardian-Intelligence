import sys
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SurrogateVisionModel(nn.Module):
    """
    Modular vision model abstraction for surrogate model training and adversarial cloaking.
    Supports ResNet-50, ConvNeXt-Tiny, and ViT-B/16.
    Provides access to deep intermediate embeddings before the classification head.
    """
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 20,
        pretrained: bool = True,
        embedding_dim: Optional[int] = None,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone_flag = freeze_backbone

        self.backbone, self.feature_dim = self._build_backbone()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Optional projection head for embedding space normalization
        self.embedding_dim = embedding_dim or self.feature_dim
        if self.embedding_dim != self.feature_dim:
            self.embedding_proj = nn.Linear(self.feature_dim, self.embedding_dim)
        else:
            self.embedding_proj = nn.Identity()

        # Final classification head
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.embedding_dim, self.num_classes)
            )
        else:
            self.classifier = None

    def _build_backbone(self) -> Tuple[nn.Module, int]:
        name = self.model_name
        weights = "DEFAULT" if self.pretrained else None

        if "resnet50" in name:
            base = models.resnet50(weights=weights)
            feature_dim = base.fc.in_features
            base.fc = nn.Identity()
            return base, feature_dim

        elif "resnet18" in name:
            base = models.resnet18(weights=weights)
            feature_dim = base.fc.in_features
            base.fc = nn.Identity()
            return base, feature_dim

        elif "convnext" in name:
            base = models.convnext_tiny(weights=weights)
            feature_dim = base.classifier[2].in_features
            base.classifier[2] = nn.Identity()
            return base, feature_dim

        elif "clip" in name:
            try:
                from transformers import CLIPVisionModelWithProjection
                base = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
                feature_dim = getattr(base.config, "projection_dim", 512)
                return base, feature_dim
            except Exception as e:
                print(f"[Warning] Could not load HuggingFace CLIP ({e}), falling back to torchvision ViT-B/16.")
                base = models.vit_b_16(weights=weights)
                feature_dim = base.heads.head.in_features
                base.heads.head = nn.Identity()
                return base, feature_dim

        elif "vit" in name:
            base = models.vit_b_16(weights=weights)
            feature_dim = base.heads.head.in_features
            base.heads.head = nn.Identity()
            return base, feature_dim

        else:
            raise ValueError(
                f"Unsupported model architecture '{name}'. "
                f"Choose from: 'resnet50', 'resnet18', 'convnext_tiny', 'vit_b_16'."
            )

    def freeze_backbone(self):
        """Freezes backbone weights, keeping classification head trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.freeze_backbone_flag = True

    def unfreeze_last_blocks(self):
        """Unfreezes the deepest residual block (e.g. layer4 in ResNet-50) or deep stages."""
        if hasattr(self.backbone, "layer4"):
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            print("[Transfer Learning] Unfroze ResNet-50 layer4 blocks.")
        elif hasattr(self.backbone, "features") and len(self.backbone.features) > 0:
            for param in self.backbone.features[-2:].parameters():
                param.requires_grad = True
            print("[Transfer Learning] Unfroze deep backbone stages.")
        elif hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layers"):
            for param in self.backbone.encoder.layers[-2:].parameters():
                param.requires_grad = True
            print("[Transfer Learning] Unfroze last 2 transformer encoder blocks.")

    def unfreeze_all(self):
        """Unfreezes all layers for full end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        self.freeze_backbone_flag = False
        print("[Transfer Learning] Unfroze all backbone and head parameters.")

    def get_param_counts(self) -> Tuple[int, int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return total_params, trainable_params, frozen_params

    def configure_parameter_groups(
        self,
        mode: str = "full_finetuning",
        head_lr: float = 1e-4,
        layer4_lr: float = 5e-5,
        base_lr: float = 1e-5,
        weight_decay: float = 1e-4
    ) -> List[Dict[str, Any]]:
        """
        Configures model freezing and differential learning rate parameter groups:
          - 'frozen_backbone' (Exp A): Only classifier head is trainable.
          - 'finetune_layer4' (Exp B): Head + layer4 trainable, earlier backbone frozen.
          - 'full_finetuning' (Exp C): All layers trainable with differential LRs (head > layer4 > earlier).
        """
        mode = mode.lower()
        param_groups = []

        if mode == "frozen_backbone":
            self.freeze_backbone()
            # Unfreeze classifier and projection head
            for param in self.classifier.parameters():
                param.requires_grad = True
            for param in self.embedding_proj.parameters():
                param.requires_grad = True

            head_params = list(self.classifier.parameters()) + list(self.embedding_proj.parameters())
            param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
            print(f"[Transfer Learning] Mode: FROZEN BACKBONE | Head LR: {head_lr:.1e} | Backbone: FROZEN")

        elif mode in ("finetune_layer4", "layer4_only"):
            self.freeze_backbone()
            self.unfreeze_last_blocks()
            for param in self.classifier.parameters():
                param.requires_grad = True
            for param in self.embedding_proj.parameters():
                param.requires_grad = True

            head_params = list(self.classifier.parameters()) + list(self.embedding_proj.parameters())
            if hasattr(self.backbone, "layer4"):
                deep_params = list(self.backbone.layer4.parameters())
            elif hasattr(self.backbone, "features") and len(self.backbone.features) > 0:
                deep_params = list(self.backbone.features[-2:].parameters())
            elif hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layers"):
                deep_params = list(self.backbone.encoder.layers[-2:].parameters())
            else:
                deep_params = []

            param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
            if deep_params:
                param_groups.append({"params": deep_params, "lr": layer4_lr, "weight_decay": weight_decay})
            print(f"[Transfer Learning] Mode: FINETUNE LAYER4/DEEP BLOCKS | Head LR: {head_lr:.1e} | Deep LR: {layer4_lr:.1e} | Earlier: FROZEN")

        else:  # full_finetuning
            self.unfreeze_all()
            head_params = list(self.classifier.parameters()) + list(self.embedding_proj.parameters())
            
            if hasattr(self.backbone, "layer4"):
                layer4_params = list(self.backbone.layer4.parameters())
                earlier_params = [
                    p for n, p in self.backbone.named_parameters()
                    if not n.startswith("layer4")
                ]
                param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
                param_groups.append({"params": layer4_params, "lr": layer4_lr, "weight_decay": weight_decay})
                param_groups.append({"params": earlier_params, "lr": base_lr, "weight_decay": weight_decay})
                print(f"[Transfer Learning] Mode: FULL FINE-TUNING | Head LR: {head_lr:.1e} | Layer4 LR: {layer4_lr:.1e} | Earlier LR: {base_lr:.1e}")
            elif hasattr(self.backbone, "features") and len(self.backbone.features) > 0:
                deep_params = list(self.backbone.features[-2:].parameters())
                deep_param_ids = set(id(p) for p in deep_params)
                earlier_params = [p for p in self.backbone.parameters() if id(p) not in deep_param_ids]
                param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
                param_groups.append({"params": deep_params, "lr": layer4_lr, "weight_decay": weight_decay})
                param_groups.append({"params": earlier_params, "lr": base_lr, "weight_decay": weight_decay})
                print(f"[Transfer Learning] Mode: FULL FINE-TUNING | Head LR: {head_lr:.1e} | Deep LR: {layer4_lr:.1e} | Earlier LR: {base_lr:.1e}")
            else:
                param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
                param_groups.append({"params": self.backbone.parameters(), "lr": base_lr, "weight_decay": weight_decay})
                print(f"[Transfer Learning] Mode: FULL FINE-TUNING | Head LR: {head_lr:.1e} | Backbone LR: {base_lr:.1e}")

        total, trainable, frozen = self.get_param_counts()
        print(f"  Trainable Parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%) | Frozen: {frozen:,}")
        return param_groups

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts raw backbone features before the embedding projection."""
        if "clip" in self.model_name and hasattr(self.backbone, "vision_model"):
            outputs = self.backbone(pixel_values=x)
            return outputs.image_embeds
        return self.backbone(x)

    def get_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Extracts L2-normalized deep feature vector.
        Essential for adversarial feature-space disruption and transferability analysis.
        """
        features = self.extract_features(x)
        embeddings = self.embedding_proj(features)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass producing classification logits and optional deep feature embeddings.
        """
        features = self.extract_features(x)
        embeddings = self.embedding_proj(features)

        if self.classifier is not None:
            logits = self.classifier(embeddings)
        else:
            logits = embeddings

        if return_embedding:
            norm_embeddings = F.normalize(embeddings, p=2, dim=-1)
            return logits, norm_embeddings

        return logits

    def print_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print("\n" + "=" * 50)
        print(f"Model Architecture    : {self.model_name.upper()}")
        print(f"Pretrained Weights    : {'ImageNet-1k (DEFAULT)' if self.pretrained else 'Random Init'}")
        print(f"Number of parameters  : {total_params:,}")
        print(f"Trainable parameters  : {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"Frozen parameters     : {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print("=" * 50 + "\n")


def get_model(
    model_name: str = "resnet50",
    num_classes: int = 20,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: Optional[torch.device] = None
) -> SurrogateVisionModel:
    """Convenience factory function."""
    model = SurrogateVisionModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Models] Testing architectures on {device}...")

    for arch in ["resnet50", "convnext_tiny", "vit_b_16"]:
        try:
            m = get_model(arch, num_classes=10, pretrained=False, device=device)
            m.print_summary()
            dummy_input = torch.randn(2, 3, 224, 224, device=device)
            logits, embs = m(dummy_input, return_embedding=True)
            print(f"  [OK]  {arch:<15}: Logits={list(logits.shape)}, Embeddings={list(embs.shape)}")
        except Exception as e:
            print(f"  [ERR] {arch:<15}: Error {e}")
