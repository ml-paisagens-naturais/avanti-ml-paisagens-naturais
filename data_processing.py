"""
Módulo para processamento de dados e augmentation no projeto de classificação de paisagens.

Este módulo contém funções para processamento de imagens, criação de DataFrames,
visualizações de dados e implementações de técnicas de data augmentation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import imagehash
import torch
import torchvision
from torchvision import transforms
import glob
import logging
import random
from pathlib import Path
import warnings
from torch.utils.tensorboard import SummaryWriter

# Obter logger
logger = logging.getLogger("landscape_classifier")

def create_df(data_dir):
    """
    Cria um DataFrame com metadados das imagens
    
    Args:
        data_dir: Diretório contendo as imagens organizadas em pastas por classe
        
    Returns:
        tuple: (DataFrame com metadados, lista de imagens corrompidas)
    """
    logger.info(f"Iniciando processamento do diretório: {data_dir}")
    
    data = []
    corrupted_count = 0
    corrupted_images = []
    
    # Percorre as classes (pastas) no diretório
    class_counts = {}
    try:
        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        logger.info(f"Classes encontradas: {classes}")
        
        for class_name in classes:
            class_dir = os.path.join(data_dir, class_name)
            logger.info(f"Processando classe: {class_name}")
            class_image_count = 0
            
            # Percorre as imagens na pasta da classe
            image_files = glob.glob(os.path.join(class_dir, "*.jpg"))
            logger.info(f"Encontradas {len(image_files)} imagens na classe {class_name}")
            
            for img_path in image_files:
                img_corrupted = False
                img_hash = None
                width, height, channels = None, None, None
                img_format = None
                
                try:
                    # Tenta abrir a imagem para verificar corrupção
                    img = cv2.imread(img_path)
                    if img is None:
                        img_corrupted = True
                        corrupted_count += 1
                        corrupted_images.append(img_path)
                        logger.warning(f"Imagem corrompida: {img_path}")
                    else:
                        # Obtém dimensões e canais
                        height, width, channels = img.shape
                        
                        # Obtém formato da imagem
                        img_format = os.path.splitext(img_path)[1][1:].lower()
                        
                        # Calcula o hash da imagem usando o perceptual hash
                        pil_img = Image.open(img_path)
                        img_hash = str(imagehash.phash(pil_img))
                        class_image_count += 1
                        
                except Exception as e:
                    img_corrupted = True
                    corrupted_count += 1
                    corrupted_images.append(img_path)
                    logger.error(f"Erro ao processar {img_path}: {str(e)}")
                
                # Adiciona os metadados ao dataframe
                data.append({
                    'image_path': img_path,
                    'label': class_name,
                    'image_format': img_format,
                    'width': width,
                    'height': height,
                    'channels': channels,
                    'image_hash': img_hash,
                    'corrupted': img_corrupted
                })
            
            class_counts[class_name] = class_image_count
            logger.info(f"Classe {class_name}: {class_image_count} imagens processadas")
        
        df = pd.DataFrame(data)
        
        # Logar estatísticas
        logger.info(f"Total de imagens processadas: {len(df)}")
        logger.info(f"Total de imagens corrompidas: {corrupted_count}")
        
        # Verificar balanceamento
        class_distribution = df['label'].value_counts()
        if len(class_distribution) > 1:
            imbalance_ratio = class_distribution.max() / class_distribution.min()
            if imbalance_ratio > 1.5:
                logger.warning(f"Dataset desbalanceado. Razão max/min: {imbalance_ratio:.2f}")
                logger.warning(f"Distribuição detalhada: {class_distribution.to_dict()}")
            else:
                logger.info(f"Dataset razoavelmente balanceado. Razão max/min: {imbalance_ratio:.2f}")
        
        return df, corrupted_images
    except Exception as e:
        logger.critical(f"Erro crítico ao processar diretório {data_dir}", exc_info=True)
        raise

def show_random_images(df, num_images=6, save_path=None):
    """
    Mostra e salva imagens aleatórias do dataset, uma de cada classe
    
    Args:
        df: DataFrame contendo metadados das imagens
        num_images: Número de imagens para mostrar
        save_path: Caminho para salvar a figura (opcional)
        
    Returns:
        bool: True se a visualização foi gerada com sucesso
    """
    try:
        logger.info(f"Gerando visualização de imagens para todas as classes")
        plt.figure(figsize=(15, 10))
        
        # Obter lista de classes únicas
        classes = sorted(df['label'].unique())
        logger.info(f"Classes para visualização: {classes}")
        
        # Garantir que num_images seja pelo menos igual ao número de classes
        num_images = max(num_images, len(classes))
        
        # Inicializar DataFrame vazio para armazenar as amostras
        sample_df = pd.DataFrame()
        
        # Selecionar uma imagem de cada classe
        for cls in classes:
            # Filtrar imagens não corrompidas da classe atual
            cls_images = df[(df['label'] == cls) & (~df['corrupted'])]
            # Selecionar uma imagem aleatória desta classe
            if not cls_images.empty:
                cls_sample = cls_images.sample(1)
                # Adicionar ao DataFrame de amostras
                sample_df = pd.concat([sample_df, cls_sample])
        
        # Se precisar de mais imagens (para completar num_images), selecione aleatoriamente
        if len(sample_df) < num_images:
            # Excluir as imagens já selecionadas
            remaining_df = df[~df.index.isin(sample_df.index) & (~df['corrupted'])]
            # Selecionar imagens adicionais aleatoriamente
            additional_samples = remaining_df.sample(num_images - len(sample_df))
            # Adicionar ao DataFrame de amostras
            sample_df = pd.concat([sample_df, additional_samples])
        
        # Limitar ao número desejado (caso tenha mais classes que num_images)
        sample_df = sample_df.head(num_images)
        
        for i, (idx, row) in enumerate(sample_df.iterrows()):
            try:
                img = cv2.imread(row['image_path'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter de BGR para RGB
                
                plt.subplot(1, num_images, i+1)
                plt.imshow(img)
                plt.title(f"Classe: {row['label']}")
                plt.axis('off')
                
                # Adicionar ao TensorBoard se disponível
                try:
                    # Verificar se existe uma instância do TensorBoard no escopo
                    if 'tb_writer' in globals():
                        tb_writer.add_image(f"Sample Images/Class_{row['label']}", 
                                          img.transpose(2, 0, 1)/255.0,  # Normalizar e converter para formato CHW
                                          global_step=i)
                except Exception as tb_err:
                    logger.debug(f"Não foi possível adicionar imagem ao TensorBoard: {str(tb_err)}")
                    
            except Exception as e:
                logger.error(f"Erro ao processar imagem {row['image_path']}", exc_info=True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Visualização salva em: {save_path}")
        
        return True
    except Exception as e:
        logger.error("Erro ao gerar visualização de imagens", exc_info=True)
        return False

def mixup_data(x, y, alpha=1.0, model_type='cnn'):
    """
    Realiza mixup de batches de dados, adaptado ao tipo de modelo
    
    Args:
        x: Tensor de imagens
        y: Tensor de rótulos
        alpha: Parâmetro para a distribuição beta
        model_type: Tipo de modelo ('cnn', 'vit', 'swin', etc.)
        
    Returns:
        tuple: Dependendo do tipo de modelo:
               - Para 'cnn'/'deit': (inputs mixados, labels originais, labels mixados, lambda)
               - Para 'swin': (inputs mixados, labels originais)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    # Para Swin Transformer, retornar apenas os rótulos originais (sem mixup nos rótulos)
    if model_type.lower() in ['swin', 'swin_transformer']:
        return mixed_x, y
    
    # Para CNNs e outros modelos que funcionam com mixup tradicional
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b=None, lam=None, model_type='cnn'):
    """
    Calcula a perda para dados com mixup, adaptado ao tipo de modelo
    
    Args:
        criterion: Função de perda
        pred: Previsões do modelo
        y_a: Rótulos originais
        y_b: Rótulos permutados
        lam: Fator de mistura
        model_type: Tipo de modelo ('cnn', 'vit', 'swin', etc.)
        
    Returns:
        Perda calculada
    """
    # Para Swin Transformer, aplicar critério diretamente
    if model_type.lower() in ['swin', 'swin_transformer']:
        return criterion(pred, y_a)
    
    # Tratamento específico para modelos ViT/DeiT
    if model_type.lower() in ['vit', 'deit']:
        # Para modelos ViT, verificar dimensões e ajustar conforme necessário
        if isinstance(y_a, torch.Tensor):
            if y_a.dim() > 1:
                y_a = y_a.squeeze()
            y_a = y_a.long()
        
        if y_b is not None and isinstance(y_b, torch.Tensor):
            if y_b.dim() > 1:
                y_b = y_b.squeeze()
            y_b = y_b.long()
        
        # Se y_b for None ou não for um tensor, retornar apenas a perda para y_a
        if y_b is None or not isinstance(y_b, torch.Tensor):
            return criterion(pred, y_a)
    else:
        # Para CNNs e outros modelos, garantir formato 1D
        if isinstance(y_a, torch.Tensor):
            y_a = y_a.view(-1).long()
        
        if y_b is not None and isinstance(y_b, torch.Tensor):
            y_b = y_b.view(-1).long()
    
    # Verificar se temos todos os dados necessários para aplicar mixup
    if y_b is None or lam is None:
        return criterion(pred, y_a)
    
    # Aplicar mixup tradicional
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def create_augmentation_transforms(use_randaugment=True, model_type=None):
    """
    Cria transformações com técnicas modernas de augmentation, otimizadas por tipo de modelo
    
    Args:
        use_randaugment: Boolean indicando se deve usar RandAugment
        model_type: Tipo de modelo ('cnn' ou 'vit')
        
    Returns:
        transforms.Compose: Composição de transformações
    """
    logger.info(f"Criando transformações com RandAugment={use_randaugment} para model_type={model_type}")
    
    # Transformações base
    transforms_list = []
    
    # Transformações específicas para ViT - seguindo práticas de DeiT e outros papers
    if model_type == 'vit':
        # ViTs se beneficiam de RandomResizedCrop com proporções mais extremas
        transforms_list.append(transforms.RandomResizedCrop(
            (224, 224), 
            scale=(0.08, 1.0),  # Mais agressivo que o padrão (0.08, 1.0)
            ratio=(0.75, 1.3333)  # Mais agressivo que o padrão (0.75, 1.33)
        ))
        transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        # Adicionar RandAugment com parâmetros mais fortes para ViT
        if use_randaugment:
            transforms_list.append(transforms.RandAugment(num_ops=4, magnitude=9))  # Mais ops e magnitude maior
            logger.info("RandAugment adicionado com parâmetros otimizados para ViT")
        
        # Aumento de Cor mais agressivo para ViT
        transforms_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
        
        # Primeiro converter para tensor e normalizar
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        # Adicionar RandomErasing APÓS converter para tensor
        transforms_list.append(transforms.RandomErasing(p=0.25))
        
    else:
        # Transformações padrão para CNN
        transforms_list.append(transforms.Resize((224, 224)))
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.RandomRotation(10))
        
        # Adicionar RandAugment se solicitado
        if use_randaugment:
            transforms_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
            logger.info("RandAugment adicionado às transformações de treinamento")
        
        # Adicionar transformações básicas
        transforms_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1))
        
        # Transformações finais comuns para CNN
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(transforms_list)

def verify_dataset_cache(dataset):
    """
    Verifica o status do cache do dataset
    
    Args:
        dataset: Dataset para verificar
        
    Returns:
        bool: True se o dataset tiver cache
    """
    from dataset_utils import CachedDataset
    
    if isinstance(dataset, CachedDataset):
        logger.info(f"Dataset com cache: {len(dataset.cache)} itens em cache de {len(dataset)}")
        return True
    else:
        logger.info(f"Dataset sem cache: {len(dataset)} itens")
        return False