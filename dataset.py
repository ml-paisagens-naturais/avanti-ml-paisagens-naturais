import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import cv2
from PIL import Image
import numpy as np
import logging
import random
import os
from typing import Tuple, Dict, List, Optional, Union
from dataset_utils import CachedDataset


logger = logging.getLogger("landscape_classifier")

class IntelImageDataset(Dataset):
    """
    Dataset para classificação de imagens Intel.
    Suporta técnicas avançadas de data augmentation como Mixup e CutMix.
    
    Attributes:
        dataframe: DataFrame pandas com caminhos de imagem e rótulos
        transform: Transformações PyTorch a serem aplicadas às imagens
        apply_mixup: Booleano indicando se a técnica Mixup deve ser aplicada
        mixup_alpha: Parâmetro alpha para a distribuição beta do Mixup
        cutmix_prob: Probabilidade de aplicar CutMix a uma imagem
        cutmix_alpha: Parâmetro alpha para a distribuição beta do CutMix
        balance_classes: Booleano indicando se o balanceamento de classes deve ser aplicado
        class_weights: Pesos para cada classe (para balanceamento)
        sample_weights: Pesos para cada amostra (para balanceamento)
        classes: Lista de nomes de classes únicas
        class_to_idx: Dicionário mapeando nomes de classes para índices
    """

    def __init__(
        self, 
        dataframe, 
        transform=None, 
        apply_mixup=False, 
        mixup_alpha=1.0,
        cutmix_prob=0.0, 
        cutmix_alpha=1.0,
        balance_classes=False,
        return_one_hot=False,
        verify_images=False,
        model_type='cnn'
    ):
        """
        Inicializa o dataset.
        
        Args:
            dataframe: DataFrame pandas com colunas 'image_path', 'label' e 'corrupted'
            transform: Transformações PyTorch a serem aplicadas às imagens
            apply_mixup: Se True, aplica Mixup com probabilidade interna
            mixup_alpha: Parâmetro alpha para distribuição beta do Mixup
            cutmix_prob: Probabilidade de aplicar CutMix a uma imagem
            cutmix_alpha: Parâmetro alpha para distribuição beta do CutMix
            balance_classes: Se True, calcula pesos para balanceamento de classes
            return_one_hot: Se True, retorna rótulos em formato one-hot
            verify_images: Se True, verifica todas as imagens antes do treinamento
        """
        # Filtrar imagens corrompidas
        self.df = dataframe[~dataframe['corrupted']].reset_index(drop=True)
        self.transform = transform
        self.apply_mixup = apply_mixup
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob
        self.cutmix_alpha = cutmix_alpha
        self.balance_classes = balance_classes
        self.return_one_hot = return_one_hot
        self.bad_images = []  # Lista para armazenar imagens problemáticas
        
        self.model_type = model_type

        # Verificar se existem imagens
        if len(self.df) == 0:
            raise ValueError("Dataset vazio após filtrar imagens corrompidas")
        
        # Mapear rótulos de texto para números
        self.classes = sorted(self.df['label'].unique())
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Calcular pesos para balanceamento se solicitado
        if balance_classes:
            class_counts = self.df['label'].value_counts().to_dict()
            max_count = max(class_counts.values())
            self.class_weights = {cls: max_count/count for cls, count in class_counts.items()}
            self.sample_weights = [self.class_weights[self.df.iloc[i]['label']] for i in range(len(self.df))]
            logger.info(f"Pesos de classes calculados para balanceamento: {self.class_weights}")
        else:
            self.class_weights = None
            self.sample_weights = None
        
        logger.info(f"Dataset criado com {len(self.df)} imagens e {len(self.classes)} classes")
        logger.info(f"Mapeamento de classes: {self.class_to_idx}")
        
        # Registrar configurações de augmentation
        aug_config = []
        if self.apply_mixup:
            aug_config.append(f"Mixup (alpha={mixup_alpha})")
        if self.cutmix_prob > 0:
            aug_config.append(f"CutMix (prob={cutmix_prob}, alpha={cutmix_alpha})")
        if aug_config:
            logger.info(f"Configurações de augmentation: {', '.join(aug_config)}")
            
        # Verificar imagens se solicitado
        if verify_images:
            self._verify_all_images()
    
    def _verify_all_images(self):
        """
        Verifica todas as imagens do dataset para identificar problemas antes do treinamento.
        """
        logger.info("Verificando integridade de todas as imagens...")
        bad_images = []
        
        for idx in range(len(self.df)):
            img_path = self.df.iloc[idx]['image_path']
            try:
                with Image.open(img_path) as img:
                    # Verificar se a imagem pode ser convertida para RGB
                    img = img.convert('RGB')
                    
                    # Verificar dimensões mínimas
                    width, height = img.size
                    if width < 32 or height < 32:
                        bad_images.append((img_path, f"Dimensões muito pequenas: {width}x{height}"))
                        continue
                    
                    # Tentar aplicar transformações
                    if self.transform:
                        try:
                            transformed = self.transform(img)
                            if transformed.dim() != 3 or any(s == 0 for s in transformed.shape):
                                bad_images.append((img_path, f"Transformação resultou em tensor inválido: {transformed.shape}"))
                        except Exception as e:
                            bad_images.append((img_path, f"Erro na transformação: {str(e)}"))
            
            except Exception as e:
                bad_images.append((img_path, f"Erro ao abrir imagem: {str(e)}"))
        
        # Registrar resultados da verificação
        if bad_images:
            self.bad_images = bad_images
            logger.warning(f"Encontradas {len(bad_images)} imagens problemáticas durante a verificação.")
            for img_path, reason in bad_images[:10]:  # Mostrar os primeiros 10 exemplos
                logger.warning(f"  - {img_path}: {reason}")
            if len(bad_images) > 10:
                logger.warning(f"  ... e mais {len(bad_images) - 10} imagens (veja o arquivo de log para detalhes completos)")
                
            # Salvar lista completa em um arquivo de log
            with open("bad_images.log", "w") as f:
                for img_path, reason in bad_images:
                    f.write(f"{img_path}: {reason}\n")
                    
            logger.info("Lista completa de imagens problemáticas salva em 'bad_images.log'")
        else:
            logger.info("Todas as imagens foram verificadas e estão íntegras.")
    
    def __len__(self):
        """Retorna o número de amostras no dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Obtém um item do dataset pelo índice.
        
        Args:
            idx: Índice da amostra desejada.
            
        Returns:
            tuple: (imagem, rótulo) onde a imagem é um tensor e o rótulo é um índice ou vetor one-hot.
        """
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label']
        label_idx = self.class_to_idx[label]
        
        try:
            # Carregar a imagem
            image = Image.open(img_path).convert('RGB')
            
            # Aplicar transformações se fornecidas
            if self.transform:
                image = self.transform(image)
            
            # Verificar validade da imagem transformada
            if image.dim() != 3 or any(s == 0 for s in image.shape):
                raise ValueError(f"Imagem inválida após transformação: {image.shape}")
            
            # Preparar rótulo (índice ou one-hot)
            if self.return_one_hot:
                label_tensor = torch.zeros(self.num_classes)
                # Usar o índice diretamente, sem conversão para int
                label_tensor[label_idx] = 1.0
            else:
                # Não converter explicitamente para int para evitar erros com tensores
                label_tensor = label_idx
            
            # Aplicar CutMix com probabilidade especificada
            if self.cutmix_prob > 0 and random.random() < self.cutmix_prob:
                result = self._apply_cutmix(image, label_tensor)
                # Se for Swin e retornar tupla de 4 elementos, mantenha assim
                if self.model_type in ['swin', 'swin_transformer'] and len(result) == 4:
                    return result
                else:
                    return result  # imagem misturada, rótulo misturado
    
            # Aplicar Mixup com probabilidade interna
            if self.apply_mixup and self.mixup_alpha > 0 and random.random() < 0.5:
                return self._apply_mixup(image, label_tensor)
            
            return image, label_tensor
            
        except Exception as e:
            logger.error(f"Erro ao carregar imagem {img_path}: {str(e)}")
            
            # Criar uma imagem alternativa em caso de erro
            # Usar ruído gaussiano para ser mais realista do que zeros
            if 'image' in locals() and isinstance(image, torch.Tensor):
                # Conhecemos a forma esperada
                dummy_image = torch.randn_like(image) * 0.1 + 0.5
                dummy_image = torch.clamp(dummy_image, 0, 1)
            else:
                # Imagem nem sequer foi carregada, usar forma padrão
                dummy_image = torch.randn(3, 224, 224) * 0.1 + 0.5
                dummy_image = torch.clamp(dummy_image, 0, 1)
            
            if self.return_one_hot:
                dummy_label = torch.zeros(self.num_classes)
                dummy_label[label_idx] = 1.0
                return dummy_image, dummy_label
            else:
                return dummy_image, label_idx
    
    def _safely_convert_to_one_hot(self, label):
        """
        Converte um rótulo para formato one-hot de forma segura, independente do tipo.
        
        Args:
            label: Rótulo a ser convertido (pode ser int, float, tensor, etc.)
            
        Returns:
            torch.Tensor: Rótulo em formato one-hot
        """
        if isinstance(label, torch.Tensor):
            if label.dim() > 0 and label.size(0) == self.num_classes:
                # Já está em formato one-hot
                return label
            # É um tensor escalar, extrair o valor
            label_value = label.item()
        else:
            # É um valor Python normal
            label_value = label
        
        # Criar tensor one-hot    
        label_one_hot = torch.zeros(self.num_classes)
        label_one_hot[int(label_value)] = 1.0
        return label_one_hot
    
    def _apply_mixup(self, image, label):
        """
        Aplica a técnica Mixup em uma imagem, adaptada ao tipo de modelo.
    
        Args:
            image: Tensor da imagem
            label: Índice do rótulo ou tensor one-hot
        
        Returns:
            tuple: Para modelos padrão: (imagem misturada, rótulo misturado)
                Para modelos Swin: (imagem misturada, rótulo original)
        """
        # Gerar lambda da distribuição beta
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
    
        # Selecionar outra imagem aleatória para misturar
        rand_idx = random.randint(0, len(self) - 1)
        rand_img, rand_label = self[rand_idx]
    
        # Misturar imagens
        mixed_img = lam * image + (1 - lam) * rand_img
    
        # Tratamento diferenciado por tipo de modelo
        if hasattr(self, 'model_type') and self.model_type.lower() in ['swin', 'swin_transformer']:
            # Para Swin Transformer, retornamos os rótulos originais sem modificação
            return mixed_img, label
        else:
            # Para outros modelos (CNN, DeiT), usamos a abordagem original
            # Converter rótulos para one-hot de forma segura
            label_one_hot = self._safely_convert_to_one_hot(label)
            rand_label_one_hot = self._safely_convert_to_one_hot(rand_label)
        
            # Misturar rótulos
            mixed_label = lam * label_one_hot + (1 - lam) * rand_label_one_hot
        
            return mixed_img, mixed_label

    def _apply_cutmix(self, image, label):
        """
        Aplica a técnica CutMix em uma imagem, adaptada ao tipo de modelo.
    
        Args:
            image: Tensor da imagem
            label: Índice do rótulo ou tensor one-hot
        
        Returns:
            tuple: Para modelos padrão: (imagem com corte, rótulo misturado)
                Para modelos Swin: (imagem com corte, rótulo original)
        """
        # Selecionar outra imagem aleatória para misturar
        rand_idx = random.randint(0, len(self) - 1)
        rand_img, rand_label = self[rand_idx]
    
        # Gerar lambda da distribuição beta
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
    
        # Gerar coordenadas do retângulo de corte
        img_h, img_w = image.shape[1], image.shape[2]
        cut_h = int(img_h * np.sqrt(1.0 - lam))
        cut_w = int(img_w * np.sqrt(1.0 - lam))
    
        # Gerar ponto central aleatório
        cx = np.random.randint(img_w)
        cy = np.random.randint(img_h)
    
        # Obter limites do corte
        bbx1 = np.clip(cx - cut_w // 2, 0, img_w)
        bby1 = np.clip(cy - cut_h // 2, 0, img_h)
        bbx2 = np.clip(cx + cut_w // 2, 0, img_w)
        bby2 = np.clip(cy + cut_h // 2, 0, img_h)
    
        # Calcular a área efetiva do corte
        actual_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_h * img_w))
    
        # Aplicar o corte
        mixed_img = image.clone()
        mixed_img[:, bby1:bby2, bbx1:bbx2] = rand_img[:, bby1:bby2, bbx1:bbx2]
    
        # Tratamento diferenciado por tipo de modelo
        if hasattr(self, 'model_type') and self.model_type.lower() in ['swin', 'swin_transformer']:
            # Para Swin Transformer, retornamos os rótulos originais sem modificação
            return mixed_img, label
        else:
            # Para outros modelos, usar a abordagem original
            # Converter rótulos para one-hot de forma segura
            label_one_hot = self._safely_convert_to_one_hot(label)
            rand_label_one_hot = self._safely_convert_to_one_hot(rand_label)
        
            # Mixar rótulos de acordo com lambda real
            mixed_label = actual_lam * label_one_hot + (1 - actual_lam) * rand_label_one_hot
        
            return mixed_img, mixed_label
    
    def get_sampler(self):
        """
        Retorna um WeightedRandomSampler para balanceamento de classes.
        
        Returns:
            WeightedRandomSampler ou None se balance_classes=False
        """
        if not self.balance_classes or self.sample_weights is None:
            return None
            
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True
        )
        
    def get_class_weights_tensor(self, device=None):
        """
        Retorna um tensor de pesos de classe para uso com funções de perda ponderadas.
        
        Args:
            device: Dispositivo onde o tensor será alocado
            
        Returns:
            torch.Tensor: Tensor de pesos de classe ou None se balance_classes=False
        """
        if not self.balance_classes or self.class_weights is None:
            return None
            
        weights = torch.zeros(self.num_classes)
        for cls, idx in self.class_to_idx.items():
            weights[idx] = self.class_weights[cls]
            
        if device:
            weights = weights.to(device)
            
        return weights
        
    def get_bad_images(self):
        """
        Retorna a lista de imagens problemáticas identificadas.
        
        Returns:
            list: Lista de tuplas (caminho_da_imagem, razão_do_problema)
        """
        return self.bad_images