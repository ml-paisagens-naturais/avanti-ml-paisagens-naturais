#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline principal para classificação de paisagens naturais com múltiplas arquiteturas de deep learning.

Este script implementa um pipeline completo e modular para classificação de imagens de paisagens naturais 
usando diferentes arquiteturas: ResNet50, EfficientNet, Vision Transformer (ViT) e MobileNet.

Autor: Rafael Albuquerque
Data: 04/04/2025
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import time
import logging
import json
import gc
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Importações de módulos do projeto
import utils
import data_processing
import dataset
import dataset_utils
import models
import training
import visualization
import optimization
import model_compression
import ensemble
import landscape_enhancements
from dataset_utils import custom_collate_fn
from torchvision import models as tv_models

# Configuração de flags para controle granular do pipeline
# Flags para controlar o fluxo do pipeline
RUN_DATA_EXPLORATION = True    # Análise exploratória dos dados
RUN_DUPLICATE_DETECTION = True # Detecção de imagens duplicadas
RUN_DATASET_PREP = True        # Preparação e carregamento dos datasets
RUN_CROSS_VALIDATION = True   # Validação cruzada para verificar robustez (opcional)
RUN_PROFILING = True           # NEW: Executar profiling de memória/performance

# Flags para otimização de hiperparâmetros (por modelo)
RUN_RESNET_OPTIMIZATION = True      # Otimização de hiperparâmetros para ResNet
RUN_EFFICIENTNET_OPTIMIZATION = True # Otimização de hiperparâmetros para EfficientNet
RUN_MOBILENET_OPTIMIZATION = True    # Otimização de hiperparâmetros para MobileNet
RUN_VIT_OPTIMIZATION = True          # Otimização de hiperparâmetros para ViT

# Flags para treinamento individual por modelo
RUN_RESNET_TRAINING = True     # Treinamento do ResNet
RUN_EFFICIENTNET_TRAINING = True # Treinamento do EfficientNet
RUN_MOBILENET_TRAINING = True   # Treinamento seguro do MobileNet
RUN_VIT_TRAINING = True         # Treinamento do Vision Transformer

# Flags para avaliação e visualização
RUN_MODEL_EVALUATION = True    # Avaliação detalhada dos modelos
RUN_VISUALIZATION = True       # Visualizações de interpretabilidade (Grad-CAM, mapas de atenção)

# Flags para otimização de modelos e ensemble
RUN_COMPRESSION = True        # Compressão de modelos (quantização, pruning)
RUN_ENSEMBLE = True            # Criação e avaliação do ensemble
RUN_EXPORT = True             # Exportação dos modelos para ONNX
RUN_FINAL_REPORT = True        # Geração de relatório final

# Flags específicas para cada modelo
USE_RESNET = True
USE_EFFICIENTNET = True
USE_MOBILENET = True
USE_VIT = True

def main():
    """
    Função principal que orquestra todo o pipeline de classificação de imagens de paisagens.
    
    Esta função executa sequencialmente as etapas controladas pelas flags:
    1. Configuração do ambiente e inicialização
    2. Análise exploratória dos dados
    3. Preparação e carregamento dos datasets
    4. Otimização de hiperparâmetros (por modelo)
    5. Treinamento dos modelos
    6. Avaliação detalhada
    7. Visualizações para interpretabilidade
    8. Compressão de modelos
    9. Ensemble de modelos
    10. Exportação para formatos de implantação
    11. Relatório final e conclusões
    """
    # Configuração inicial
    print("=== INICIANDO PIPELINE DE CLASSIFICAÇÃO DE PAISAGENS NATURAIS ===")
    
    # Inicialização do timestamp para identificação única da execução
    EXPERIMENT_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Inicializar o logger
    logger = utils.setup_logger(log_dir=f"logs/experiment_{EXPERIMENT_ID}")
    logger.info(f"Iniciando experimento {EXPERIMENT_ID}")
    
    # Criar diretórios necessários
    utils.ensure_directories()
    logger.info("Diretórios do projeto verificados/criados")
    
    # Inicializar TensorBoard
    writer, log_dir = utils.setup_tensorboard(
        base_dir=f"tensorboard_logs/experiment_{EXPERIMENT_ID}", 
        logger=logger
    )
    logger.info(f"TensorBoard inicializado em {log_dir}")
    
    # Carregar configurações do arquivo YAML
    config = utils.load_config('config.yaml')
    logger.info("Configurações carregadas do arquivo")
    
    # Definir constantes a partir da configuração
    SEED = config['training']['seed']
    DATASET_PATH = config['dataset']['path']
    IMG_SIZE = config['dataset']['img_size']
    BATCH_SIZE = config['dataset']['batch_size']
    NUM_EPOCHS = config['training']['epochs']
    LR = config['training']['learning_rate']
    GAMMA = config['training']['gamma']
    STEP_SIZE = config['training']['step_size']
    K_FOLDS = config['training'].get('k_folds', 5)
    
    # Verificar disponibilidade de GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dispositivo: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memória total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"Memória disponível: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
        
        # Inicializar uso de memória
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Memória inicial: {torch.cuda.memory_allocated() / (1024**2):.2f} MB alocada, {torch.cuda.memory_reserved() / (1024**2):.2f} MB reservada")
    
    # Configurações para reprodutibilidade
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
    
    # Identificar subdiretórios de treinamento e teste
    train_dir = os.path.join(DATASET_PATH, 'seg_train', 'seg_train')
    test_dir = os.path.join(DATASET_PATH, 'seg_test', 'seg_test')
    pred_dir = os.path.join(DATASET_PATH, 'seg_pred', 'seg_pred')
    
    # Verificar diretórios
    if not os.path.exists(train_dir):
        train_dir = os.path.join(DATASET_PATH, 'train')
    if not os.path.exists(test_dir):
        test_dir = os.path.join(DATASET_PATH, 'test')
    if not os.path.exists(pred_dir) and os.path.exists(os.path.join(DATASET_PATH, 'pred')):
        pred_dir = os.path.join(DATASET_PATH, 'pred')
    
    logger.info(f"Diretórios de dados:")
    logger.info(f"- Treinamento: {train_dir}")
    logger.info(f"- Teste: {test_dir}")
    logger.info(f"- Predição: {pred_dir}")
    
    try:
        #######################################################
        # ETAPA 1: ANÁLISE EXPLORATÓRIA DE DADOS
        #######################################################
        
        if RUN_DATA_EXPLORATION:
            logger.info("=== ETAPA 1: ANÁLISE EXPLORATÓRIA DOS DADOS ===")
            
            # Liberar memória antes da análise exploratória
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada antes da análise exploratória")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            
            # Criar DataFrames com metadados das imagens
            logger.info("Carregando datasets...")
            df_train, train_corrupted = data_processing.create_df(train_dir)
            df_test, test_corrupted = data_processing.create_df(test_dir)
            
            print(f"Dataset de treinamento: {len(df_train)} imagens")
            print(f"Dataset de teste: {len(df_test)} imagens")
            
            # Detectar imagens duplicadas no conjunto de treinamento
            if RUN_DUPLICATE_DETECTION:
                logger.info("Executando detecção de imagens duplicadas...")
                try:
                    # Liberar memória antes de detecção de duplicatas
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info("Memória GPU liberada antes da detecção de duplicatas")
                    
                    threshold = 3  # Limiar para similaridade de imagens
                    duplicates_df = landscape_enhancements.detect_duplicate_images(df_train, hash_diff_threshold=threshold)
                    
                    if duplicates_df is not None and len(duplicates_df) > 0:
                        # Visualizar duplicatas
                        visualization_success = landscape_enhancements.visualize_duplicates(
                            duplicates_df, 
                            num_groups=3, 
                            output_dir="results/duplicates"
                        )
                        
                        # Filtrar dataset para remover duplicatas
                        df_train_filtered = landscape_enhancements.filter_duplicates(df_train, duplicates_df, strategy='best_quality')
                        
                        # Atualizar DataFrame de treinamento
                        original_count = len(df_train)
                        df_train = df_train_filtered
                        removed_count = original_count - len(df_train)
                        
                        print(f"\nDetecção de duplicatas concluída:")
                        print(f"- Imagens originais: {original_count}")
                        print(f"- Duplicatas encontradas: {len(duplicates_df.drop_duplicates(subset='image_path'))}")
                        print(f"- Imagens removidas: {removed_count}")
                        print(f"- Imagens após remoção: {len(df_train)}")
                        print(f"- Redução do dataset: {removed_count/original_count*100:.2f}%")
                        
                        # Adicionar estatísticas ao TensorBoard
                        writer.add_text("Dataset/Duplicates", f"Imagens duplicadas encontradas: {len(duplicates_df.drop_duplicates(subset='image_path'))}")
                        writer.add_text("Dataset/Duplicates", f"Imagens removidas: {removed_count}")
                        writer.add_text("Dataset/Duplicates", f"Redução do dataset: {removed_count/original_count*100:.2f}%")
                        
                        # Salvar relatório de duplicatas
                        duplicates_df.to_csv("results/duplicate_images.csv", index=False)
                        logger.info(f"Relatório de duplicatas salvo em results/duplicate_images.csv")
                    else:
                        logger.info("Nenhuma imagem duplicada encontrada no conjunto de treinamento.")
                except Exception as e:
                    logger.error(f"Erro durante detecção de duplicatas: {str(e)}")
                    logger.warning("Continuando com o dataset original sem remoção de duplicatas.")
                
                # Liberar memória após detecção de duplicatas
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após detecção de duplicatas")
            
            # Verificar a distribuição de classes
            train_class_dist = df_train['label'].value_counts()
            test_class_dist = df_test['label'].value_counts()
            
            print("\nDistribuição de classes (Treinamento):")
            print(train_class_dist)
            
            print("\nDistribuição de classes (Teste):")
            print(test_class_dist)
            
            # Adicionar estatísticas ao TensorBoard
            writer.add_text("Dataset/Info", f"Total imagens treinamento: {len(df_train)}")
            writer.add_text("Dataset/Info", f"Total imagens teste: {len(df_test)}")
            writer.add_text("Dataset/Info", f"Classes: {list(train_class_dist.index)}")
            
            # Verificar imagens corrompidas
            if train_corrupted:
                logger.warning(f"Encontradas {len(train_corrupted)} imagens corrompidas no conjunto de treinamento")
            if test_corrupted:
                logger.warning(f"Encontradas {len(test_corrupted)} imagens corrompidas no conjunto de teste")
            
            # Mostrar exemplos de cada classe
            data_processing.show_random_images(df_train, num_images=6, save_path="results/sample_images.png")
            
            # Visualizar distribuição por classe
            plt.figure(figsize=(12, 6))
            sns.barplot(x=train_class_dist.index, y=train_class_dist.values)
            plt.title("Distribuição de Classes - Treinamento")
            plt.ylabel("Número de imagens")
            plt.xlabel("Classe")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("results/class_distribution.png")
            plt.close()
            
            # Analisar dimensões e formatos das imagens
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.histplot(df_train['width'])
            plt.title("Distribuição de Larguras")
            plt.subplot(1, 2, 2)
            sns.histplot(df_train['height'])
            plt.title("Distribuição de Alturas")
            plt.tight_layout()
            plt.savefig("results/image_dimensions.png")
            plt.close()
            
            # Registrar visualizações no TensorBoard
            figure = plt.figure(figsize=(12, 6))
            sns.barplot(x=train_class_dist.index, y=train_class_dist.values)
            plt.title("Distribuição de Classes")
            writer.add_figure("Dataset/class_distribution", figure, 0)
            plt.close()
            
            # Liberar memória após visualizações
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada após visualizações")
        else:
            # Mesmo pulando exploração, precisamos carregar os datasets
            logger.info("Carregando datasets diretamente (pulando exploração detalhada)...")
            df_train, _ = data_processing.create_df(train_dir)
            df_test, _ = data_processing.create_df(test_dir)

        #######################################################
        # ETAPA 2: PREPARAÇÃO DE DATASETS E TRANSFORMAÇÕES
        #######################################################
        
        if RUN_DATASET_PREP:
            logger.info("=== ETAPA 2: PREPARAÇÃO DE DATASETS E TRANSFORMAÇÕES ===")
            
            # Liberar memória antes de preparação de datasets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada antes da preparação de datasets")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            
            # Criar transformações específicas para cada modelo
            import torchvision.transforms as transforms
            from torchvision.transforms import InterpolationMode
            
            # Transformações para ResNet e EfficientNet (CNNs padrão)
            cnn_transforms = data_processing.create_augmentation_transforms(use_randaugment=True, model_type='cnn')
            
            # Transformações específicas para ViT
            vit_transforms = data_processing.create_augmentation_transforms(use_randaugment=True, model_type='vit')
            
            # Transformações para MobileNet (mais leve)
            mobile_transforms = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Transformações para validação/teste (sem augmentation)
            val_transforms = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Criar datasets com balanceamento de classes e técnicas avançadas
            logger.info("Criando datasets para treinamento...")
            
            train_dataset_resnet = dataset.IntelImageDataset(
                df_train, 
                transform=cnn_transforms, 
                apply_mixup=True, 
                mixup_alpha=0.2,
                cutmix_prob=0.1, 
                balance_classes=True,
                model_type='cnn'  # Especificar CNN
            )

            train_dataset_efficientnet = dataset.IntelImageDataset(
                df_train, 
                transform=cnn_transforms, 
                apply_mixup=True, 
                mixup_alpha=0.2,
                cutmix_prob=0.1, 
                balance_classes=True,
                model_type='cnn'  # Especificar CNN para EfficientNet
            )

            train_dataset_mobilenet = dataset.IntelImageDataset(
                df_train, 
                transform=mobile_transforms, 
                apply_mixup=False,  # MobileNet pode ser mais instável com Mixup
                cutmix_prob=0.0, 
                balance_classes=True,
                model_type='cnn'  # Especificar CNN para MobileNet
            )

            train_dataset_vit = dataset.IntelImageDataset(
                df_train, 
                transform=vit_transforms, 
                apply_mixup=True, 
                mixup_alpha=0.3,
                cutmix_prob=0.3,  # ViT se beneficia mais de CutMix
                balance_classes=True,
                model_type='swin'  # Especificar Swin para evitar o erro com mixup
            )

            # Dataset de validação/teste (mesmo para todos os modelos)
            test_dataset = dataset.IntelImageDataset(
                df_test, 
                transform=val_transforms,
                apply_mixup=False,
                balance_classes=False,
                model_type='cnn'  # O padrão, mas não importa já que mixup está desativado
            )

            def validate_dataset(dataset):
                invalid_count = 0
                for i in range(len(dataset)):
                    try:
                        item = dataset[i]
                        if not isinstance(item, tuple) or len(item) != 2:
                            invalid_count += 1
                            logger.warning(f"Item {i} tem formato inválido: {type(item)}")
                    except Exception as e:
                        invalid_count += 1
                        logger.warning(f"Erro ao acessar item {i}: {str(e)}")
                logger.info(f"Validação de dataset: {invalid_count} itens inválidos de {len(dataset)}")
                return invalid_count

            # Execute a validação dos datasets
            logger.info("Validando dataset de treinamento ResNet...")
            validate_dataset(train_dataset_resnet)
            logger.info("Validando dataset de treinamento EfficientNet...")
            validate_dataset(train_dataset_efficientnet)
            logger.info("Validando dataset de treinamento MobileNet...")
            validate_dataset(train_dataset_mobilenet)
            logger.info("Validando dataset de treinamento ViT...")
            validate_dataset(train_dataset_vit)
            logger.info("Validando dataset de teste...")
            validate_dataset(test_dataset)
                        
            # Implementar cache para datasets
            logger.info("Aplicando cache aos datasets para melhor desempenho...")
            cache_size = 1000  # Número de imagens no cache
            
            cached_train_dataset_resnet = dataset_utils.CachedDataset(train_dataset_resnet, cache_size=cache_size)
            cached_train_dataset_efficientnet = dataset_utils.CachedDataset(train_dataset_efficientnet, cache_size=cache_size)
            cached_train_dataset_mobilenet = dataset_utils.CachedDataset(train_dataset_mobilenet, cache_size=cache_size)
            cached_train_dataset_vit = dataset_utils.CachedDataset(train_dataset_vit, cache_size=cache_size)
            cached_test_dataset = dataset_utils.CachedDataset(test_dataset, cache_size=min(cache_size, len(test_dataset)))
            
            print(f"Datasets criados com cache de {cache_size} imagens")
            
            # Preparar dataloaders
            train_loader_resnet = DataLoader(
                cached_train_dataset_resnet, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                num_workers=4,
                pin_memory=True if device.type == 'cuda' else False,
                collate_fn=custom_collate_fn
            )
            
            train_loader_efficientnet = DataLoader(
                cached_train_dataset_efficientnet, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                num_workers=4,
                pin_memory=True if device.type == 'cuda' else False,
                collate_fn=custom_collate_fn
            )
            
            train_loader_mobilenet = DataLoader(
                cached_train_dataset_mobilenet, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                num_workers=4,
                pin_memory=True if device.type == 'cuda' else False,
                collate_fn=custom_collate_fn
            )
            
            train_loader_vit = DataLoader(
                cached_train_dataset_vit, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                num_workers=4,
                pin_memory=True if device.type == 'cuda' else False,
                multiprocessing_context='spawn',
                collate_fn=custom_collate_fn
            )
            
            test_loader = DataLoader(
                cached_test_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                num_workers=4,
                pin_memory=True if device.type == 'cuda' else False,
                collate_fn=custom_collate_fn
            )
            
            # Lista de classes
            classes = sorted(df_train['label'].unique())
            print(f"Classes: {classes}")
            logger.info(f"Classes no dataset: {classes}")
            
            # Liberar memória após preparação de datasets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada após preparação de datasets")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        else:
            logger.warning("ATENÇÃO: A flag RUN_DATASET_PREP está desativada. Isso pode causar erros.")

        #######################################################
        # ETAPA 3: VALIDAÇÃO CRUZADA (OPCIONAL)
        #######################################################
        
        if RUN_CROSS_VALIDATION:
            logger.info("=== ETAPA 3: VALIDAÇÃO CRUZADA ===")
            
            # Liberar memória antes de validação cruzada
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada antes da validação cruzada")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            
            # Função para criar modelo para validação cruzada
            def create_model_for_cv():
                """Cria um modelo ResNet para validação cruzada"""
                return models.create_model_with_best_params({
                    'freeze_layers': 6,
                    'n_layers': 2,
                    'fc_size_0': 512,
                    'dropout_rate': 0.3
                }, model_name='resnet50')
            
            # Realizar validação cruzada
            cross_val_mean, cross_val_std, fold_results = training.perform_cross_validation(
                model_fn=create_model_for_cv,
                dataframe=df_train,
                transform=cnn_transforms,
                n_folds=K_FOLDS,
                n_epochs=5,  # Menos épocas para validação cruzada
                batch_size=BATCH_SIZE
            )
            
            logger.info(f"Validação cruzada completa: {cross_val_mean:.4f} ± {cross_val_std:.4f}")
            logger.info(f"Resultados por fold: {fold_results}")
            
            # Adicionar resultados ao TensorBoard
            writer.add_scalar("Cross_Validation/mean_accuracy", cross_val_mean, 0)
            writer.add_scalar("Cross_Validation/std_accuracy", cross_val_std, 0)
            
            # Visualizar resultados por fold
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, len(fold_results) + 1), fold_results)
            plt.axhline(y=cross_val_mean, color='r', linestyle='-', label=f'Média: {cross_val_mean:.4f}')
            plt.fill_between(
                range(0, len(fold_results) + 2), 
                cross_val_mean - cross_val_std, 
                cross_val_mean + cross_val_std, 
                alpha=0.2, color='r', label=f'Desvio: ±{cross_val_std:.4f}'
            )
            plt.xlabel('Fold')
            plt.ylabel('Acurácia')
            plt.title('Resultados da Validação Cruzada')
            plt.xticks(range(1, len(fold_results) + 1))
            plt.ylim(0.5, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig('results/cross_validation_results.png')
            writer.add_figure("Cross_Validation/results", plt.gcf(), 0)
            plt.close()
            
            # Liberar memória após validação cruzada
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada após validação cruzada")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        
        #######################################################
        # ETAPA 4: OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA
        #######################################################
        
        # Dicionários para armazenar os melhores parâmetros
        best_params_dict = {}
        
        # Inicializar com parâmetros padrão
        resnet_best_params = {
            'batch_size': 32, 'lr': 0.001, 'optimizer': 'Adam', 'dropout_rate': 0.3,
            'scheduler': 'StepLR', 'step_size': 7, 'gamma': 0.1,
            'freeze_layers': 6, 'n_layers': 2, 'fc_size_0': 512
        }
        
        efficientnet_best_params = {
            'batch_size': 32, 'lr': 0.0005, 'optimizer': 'AdamW', 'dropout_rate': 0.2,
            'scheduler': 'CosineAnnealingLR', 'T_max': 10, 'freeze_percent': 0.7
        }
        
        mobilenet_best_params = {
            'batch_size': 64, 'lr': 0.0003, 'optimizer': 'Adam', 'dropout_rate': 0.1,
            'scheduler': 'OneCycleLR', 'max_lr': 0.003, 'freeze_percent': 0.5
        }
        
        vit_best_params = {
            'batch_size': 32, 'lr': 0.0001, 'dropout_rate': 0.1,
            'model_type': 'vit_base_patch16_224', 'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR', 'T_max': 10
        }
        
        # Preparar subconjuntos para otimização (menor para ser mais rápido)
        optim_size = min(5000, len(df_train))
        val_size = min(1000, len(df_test))
        
        # Índices aleatórios para subconjuntos
        import random
        random.seed(SEED)
        train_indices = random.sample(range(len(df_train)), optim_size)
        test_indices = random.sample(range(len(df_test)), val_size)
        
        # Criar subconjuntos dos DataFrames
        optim_train_df = df_train.iloc[train_indices].reset_index(drop=True)
        optim_test_df = df_test.iloc[test_indices].reset_index(drop=True)
        
        # Criar datasets para otimização
        optim_train_dataset = dataset.IntelImageDataset(
            optim_train_df, 
            transform=cnn_transforms, 
            balance_classes=True
        )
        
        optim_test_dataset = dataset.IntelImageDataset(
            optim_test_df, 
            transform=val_transforms,
            balance_classes=False
        )
        
        # Otimização para cada modelo
        if RUN_RESNET_OPTIMIZATION and USE_RESNET:
            logger.info("=== INICIANDO OTIMIZAÇÃO DO RESNET50 ===")
            try:
                # Liberar memória antes da otimização do ResNet
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada antes da otimização do ResNet50")
                    logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                
                resnet_best_params, resnet_best_value = optimization.run_isolated_optimization(
                    'resnet50', optim_train_dataset, optim_test_dataset
                )
                logger.info(f"Melhores parâmetros ResNet50: {resnet_best_params}")
                logger.info(f"Melhor acurácia: {resnet_best_value:.4f}")
                
                # Salvar resultados
                best_params_dict['resnet50'] = {
                    'params': resnet_best_params,
                    'accuracy': float(resnet_best_value)
                }
                
                # Salvar resultados em arquivo
                utils.save_model_hyperparameters('resnet50', resnet_best_params, resnet_best_value)
                
                # Liberar memória após otimização
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após otimização do ResNet50")
            except Exception as e:
                logger.error(f"Erro durante otimização do ResNet: {str(e)}")
                # Manter parâmetros padrão
                best_params_dict['resnet50'] = {
                    'params': resnet_best_params,
                    'accuracy': 0.0,
                    'note': 'Default parameters due to optimization error'
                }
                
                # Liberar memória em caso de erro
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após erro na otimização do ResNet50")
        
        if RUN_EFFICIENTNET_OPTIMIZATION and USE_EFFICIENTNET:
            logger.info("=== INICIANDO OTIMIZAÇÃO DO EFFICIENTNET ===")
            try:
                # Liberar memória antes da otimização do EfficientNet
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada antes da otimização do EfficientNet")
                    logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                
                efficientnet_best_params, efficientnet_best_value = optimization.run_isolated_optimization(
                    'efficientnet', optim_train_dataset, optim_test_dataset
                )
                logger.info(f"Melhores parâmetros EfficientNet: {efficientnet_best_params}")
                logger.info(f"Melhor acurácia: {efficientnet_best_value:.4f}")
                
                # Salvar resultados
                best_params_dict['efficientnet'] = {
                    'params': efficientnet_best_params,
                    'accuracy': float(efficientnet_best_value)
                }
                
                # Salvar resultados em arquivo
                utils.save_model_hyperparameters('efficientnet', efficientnet_best_params, efficientnet_best_value)
                
                # Liberar memória após otimização
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após otimização do EfficientNet")
            except Exception as e:
                logger.error(f"Erro durante otimização do EfficientNet: {str(e)}")
                # Manter parâmetros padrão
                best_params_dict['efficientnet'] = {
                    'params': efficientnet_best_params,
                    'accuracy': 0.0,
                    'note': 'Default parameters due to optimization error'
                }
                
                # Liberar memória em caso de erro
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após erro na otimização do EfficientNet")

        if RUN_MOBILENET_OPTIMIZATION and USE_MOBILENET:
            logger.info("=== INICIANDO OTIMIZAÇÃO DO MOBILENET ===")
            try:
                # Liberar memória antes da otimização do MobileNet
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada antes da otimização do MobileNet")
                    logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                
                mobilenet_best_params, mobilenet_best_value = optimization.run_isolated_optimization(
                    'mobilenet', optim_train_dataset, optim_test_dataset
                )
                logger.info(f"Melhores parâmetros MobileNet: {mobilenet_best_params}")
                logger.info(f"Melhor acurácia: {mobilenet_best_value:.4f}")
                
                # Salvar resultados
                best_params_dict['mobilenet'] = {
                    'params': mobilenet_best_params,
                    'accuracy': float(mobilenet_best_value)
                }
                
                # Salvar resultados em arquivo
                utils.save_model_hyperparameters('mobilenet', mobilenet_best_params, mobilenet_best_value)
                
                # Liberar memória após otimização
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após otimização do MobileNet")
            except Exception as e:
                logger.error(f"Erro durante otimização do MobileNet: {str(e)}")
                # Manter parâmetros padrão
                best_params_dict['mobilenet'] = {
                    'params': mobilenet_best_params,
                    'accuracy': 0.0,
                    'note': 'Default parameters due to optimization error'
                }
                
                # Liberar memória em caso de erro
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após erro na otimização do MobileNet")

        if RUN_VIT_OPTIMIZATION and USE_VIT:
            logger.info("=== INICIANDO OTIMIZAÇÃO DO VISION TRANSFORMER ===")
            try:
                # Liberar memória antes da otimização do ViT
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada antes da otimização do Vision Transformer")
                    logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                
                vit_best_params, vit_best_value = optimization.run_isolated_optimization(
                    'vit', optim_train_dataset, optim_test_dataset
                )
                logger.info(f"Melhores parâmetros ViT: {vit_best_params}")
                logger.info(f"Melhor acurácia: {vit_best_value:.4f}")
                
                # Salvar resultados
                best_params_dict['vit'] = {
                    'params': vit_best_params,
                    'accuracy': float(vit_best_value)
                }
                
                # Salvar resultados em arquivo
                utils.save_model_hyperparameters('vit', vit_best_params, vit_best_value)
                
                # Liberar memória após otimização
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após otimização do Vision Transformer")
            except Exception as e:
                logger.error(f"Erro durante otimização do ViT: {str(e)}")
                # Manter parâmetros padrão
                best_params_dict['vit'] = {
                    'params': vit_best_params,
                    'accuracy': 0.0,
                    'note': 'Default parameters due to optimization error'
                }
                
                # Liberar memória em caso de erro
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após erro na otimização do Vision Transformer")
        
        # Salvar todos os resultados em um único arquivo
        with open('results/best_hyperparameters.json', 'w') as f:
            json.dump(best_params_dict, f, indent=2)
        logger.info("Resultados da otimização salvos em results/best_hyperparameters.json")
        
        # Carregar parâmetros de arquivo se não foram otimizados agora
        if not any([RUN_RESNET_OPTIMIZATION, RUN_EFFICIENTNET_OPTIMIZATION, RUN_MOBILENET_OPTIMIZATION, RUN_VIT_OPTIMIZATION]):
            try:
                if os.path.exists('results/best_hyperparameters.json'):
                    with open('results/best_hyperparameters.json', 'r') as f:
                        saved_params = json.load(f)
                    
                    # Atualizar apenas os parâmetros que não foram otimizados agora
                    if not RUN_RESNET_OPTIMIZATION and 'resnet50' in saved_params:
                        resnet_best_params = saved_params['resnet50'].get('params', resnet_best_params)
                    
                    if not RUN_EFFICIENTNET_OPTIMIZATION and 'efficientnet' in saved_params:
                        efficientnet_best_params = saved_params['efficientnet'].get('params', efficientnet_best_params)
                    
                    if not RUN_MOBILENET_OPTIMIZATION and 'mobilenet' in saved_params:
                        mobilenet_best_params = saved_params['mobilenet'].get('params', mobilenet_best_params)
                    
                    if not RUN_VIT_OPTIMIZATION and 'vit' in saved_params:
                        vit_best_params = saved_params['vit'].get('params', vit_best_params)
                    
                    logger.info("Parâmetros carregados do arquivo de resultados anteriores.")
            except Exception as e:
                logger.warning(f"Erro ao carregar parâmetros: {str(e)}. Usando parâmetros padrão.")

        #######################################################
        # ETAPA 5: TREINAMENTO DOS MODELOS
        #######################################################
        
        # Inicializar dicionários para armazenar modelos e métricas
        trained_models = {}
        model_accuracies = {}
        model_reports = {}
        training_history = {}
        
        # Treinar ResNet50
        if RUN_RESNET_TRAINING and USE_RESNET:
            logger.info("=== TREINAMENTO DO RESNET50 ===")
            try:
                # Liberar memória antes do treinamento do ResNet50
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info(f"Memória GPU liberada antes do treinamento do ResNet50")
                    logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                
                # Criar modelo com hiperparâmetros otimizados
                resnet_model = models.create_model_with_best_params(resnet_best_params, model_name='resnet50')

                # Ativar gradient checkpointing para o modelo
                if utils.enable_gradient_checkpointing(resnet_model):
                    logger.info(f"Gradient checkpointing ativado para ResNet50")
                
                # Definir critério de perda
                label_smoothing = resnet_best_params.get('label_smoothing', 0.0)
                criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                
                # Configurar otimizador e scheduler
                resnet_optimizer, resnet_scheduler, resnet_scheduler_name = models.create_optimizer_and_scheduler(
                    resnet_model, resnet_best_params, train_loader_resnet
                )
                
                # Criar writer específico para este modelo
                resnet_writer = SummaryWriter(log_dir=os.path.join(log_dir, "resnet50"))

                # Realizar profiling antes do treinamento completo
                if RUN_PROFILING:
                    logger.info(f"Executando profiling para o modelo ResNet50")
                    prof = utils.profile_model_training(resnet_model, train_loader_resnet, num_steps=10)
                    logger.info("Profiling completo. Resultados disponíveis no TensorBoard.")
                
                # Parâmetros para treinamento
                use_mixup = resnet_best_params.get('use_mixup', True)
                use_cutmix = resnet_best_params.get('use_cutmix', True)
                early_stopping = resnet_best_params.get('early_stopping', True)
                patience = resnet_best_params.get('patience_stopping', 3)
                
                # Treinar modelo
                logger.info(f"Iniciando treinamento do ResNet50 com {NUM_EPOCHS} épocas...")
                resnet_model, resnet_train_losses, resnet_train_accs, resnet_val_losses, resnet_val_accs = training.train_model_optimized(
                    resnet_model, train_loader_resnet, test_loader, criterion, resnet_optimizer, resnet_scheduler,
                    NUM_EPOCHS, resnet_writer, "resnet50", use_mixup=use_mixup, use_cutmix=use_cutmix,
                    early_stopping=early_stopping, patience=patience, model_type='cnn',
                    use_amp=True,                           # Usar precisão mista automática
                    use_gradient_checkpointing=True,        # Usar gradient checkpointing
                    dynamic_batch=True,                     # Permitir redução de batch size se OOM
                    min_batch_size=4                        # Tamanho mínimo de batch permitido

                )
                
                # Salvar modelo
                os.makedirs("models", exist_ok=True)
                torch.save(resnet_model.state_dict(), "models/resnet50_final.pth")
                
                # Armazenar para uso posterior
                trained_models['resnet50'] = resnet_model
                training_history['resnet50'] = {
                    'train_losses': resnet_train_losses,
                    'val_losses': resnet_val_losses,
                    'train_accs': resnet_train_accs,
                    'val_accs': resnet_val_accs
                }
                
                # Visualizar curvas de aprendizado
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 2, 1)
                plt.plot(resnet_train_losses, label='Treino')
                plt.plot(resnet_val_losses, label='Validação')
                plt.title('Curva de Perda - ResNet50')
                plt.xlabel('Época')
                plt.ylabel('Perda')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(resnet_train_accs, label='Treino')
                plt.plot(resnet_val_accs, label='Validação')
                plt.title('Curva de Acurácia - ResNet50')
                plt.xlabel('Época')
                plt.ylabel('Acurácia')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig('results/resnet50_learning_curves.png')
                plt.close()
                
                logger.info("Treinamento do ResNet50 concluído com sucesso!")
                
                # Liberar memória após treinamento
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após treinamento do ResNet50")
                    logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            except Exception as e:
                logger.error(f"Erro durante treinamento do ResNet50: {str(e)}", exc_info=True)
                trained_models['resnet50'] = None
                
                # Liberar memória após erro
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após erro no treinamento do ResNet50")
        else:
            logger.info("Pulando treinamento do ResNet50.")
            # Tentar carregar modelo pré-treinado se disponível
            try:
                if USE_RESNET:
                    resnet_model = models.create_model_with_best_params(resnet_best_params, model_name='resnet50')
                    resnet_model.load_state_dict(torch.load('models/resnet50_final.pth'))
                    trained_models['resnet50'] = resnet_model
                    logger.info("Modelo ResNet50 pré-treinado carregado com sucesso.")
            except Exception as e:
                logger.warning(f"Não foi possível carregar o modelo ResNet50 pré-treinado: {str(e)}")
                trained_models['resnet50'] = None

        # Treinar EfficientNet
        if RUN_EFFICIENTNET_TRAINING and USE_EFFICIENTNET:
            logger.info("=== TREINAMENTO DO EFFICIENTNET ===")
            try:
                # Liberar memória antes do treinamento do EfficientNet
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info(f"Memória GPU liberada antes do treinamento do EfficientNet")
                    logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                
                # Criar modelo com hiperparâmetros otimizados
                efficient_model = models.create_model_with_best_params(efficientnet_best_params, model_name='efficientnet')

                # Ativar gradient checkpointing para o modelo
                if utils.enable_gradient_checkpointing(efficient_model):
                    logger.info(f"Gradient checkpointing ativado para EfficientNet")
                
                # Definir critério de perda
                label_smoothing = efficientnet_best_params.get('label_smoothing', 0.0)
                criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                
                # Configurar otimizador e scheduler
                efficient_optimizer, efficient_scheduler, efficient_scheduler_name = models.create_optimizer_and_scheduler(
                    efficient_model, efficientnet_best_params, train_loader_efficientnet
                )
                
                # Criar writer específico para este modelo
                efficient_writer = SummaryWriter(log_dir=os.path.join(log_dir, "efficientnet"))

                # Realizar profiling antes do treinamento completo
                if RUN_PROFILING:
                    logger.info(f"Executando profiling para o modelo EfficientNet")
                    prof = utils.profile_model_training(efficient_model, train_loader_efficientnet, num_steps=10)
                    logger.info("Profiling completo. Resultados disponíveis no TensorBoard.")
                
                # Parâmetros para treinamento
                use_mixup = efficientnet_best_params.get('use_mixup', True)
                use_cutmix = efficientnet_best_params.get('use_cutmix', True)
                early_stopping = efficientnet_best_params.get('early_stopping', True)
                patience = efficientnet_best_params.get('patience_stopping', 3)
                
                # Monitorar memória antes do treinamento
                utils.monitor_memory_usage(logger, "antes do treinamento do EfficientNet")


                # Treinar modelo
                logger.info(f"Iniciando treinamento do EfficientNet com {NUM_EPOCHS} épocas...")
                efficient_model, efficient_train_losses, efficient_train_accs, efficient_val_losses, efficient_val_accs = training.train_model_optimized(
                    efficient_model, train_loader_efficientnet, test_loader, criterion, efficient_optimizer, efficient_scheduler,
                    NUM_EPOCHS, efficient_writer, "efficientnet", use_mixup=use_mixup, use_cutmix=use_cutmix,
                    early_stopping=early_stopping, patience=patience, model_type='cnn',
                    use_amp=True,                           # Usar precisão mista automática
                    use_gradient_checkpointing=True,        # Usar gradient checkpointing
                    dynamic_batch=True,                     # Permitir redução de batch size se OOM
                    min_batch_size=4                        # Tamanho mínimo de batch permitido

                )
                
                # Monitorar memória após treinamento
                utils.monitor_memory_usage(logger, "após treinamento do EfficientNet")

                # Salvar modelo
                torch.save(efficient_model.state_dict(), "models/efficientnet_final.pth")
                
                # Armazenar para uso posterior
                trained_models['efficientnet'] = efficient_model
                training_history['efficientnet'] = {
                    'train_losses': efficient_train_losses,
                    'val_losses': efficient_val_losses,
                    'train_accs': efficient_train_accs,
                    'val_accs': efficient_val_accs
                }
                
                # Visualizar curvas de aprendizado
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 2, 1)
                plt.plot(efficient_train_losses, label='Treino')
                plt.plot(efficient_val_losses, label='Validação')
                plt.title('Curva de Perda - EfficientNet')
                plt.xlabel('Época')
                plt.ylabel('Perda')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(efficient_train_accs, label='Treino')
                plt.plot(efficient_val_accs, label='Validação')
                plt.title('Curva de Acurácia - EfficientNet')
                plt.xlabel('Época')
                plt.ylabel('Acurácia')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig('results/efficientnet_learning_curves.png')
                plt.close()
                
                logger.info("Treinamento do EfficientNet concluído com sucesso!")
                
                # Liberar memória após treinamento
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após treinamento do EfficientNet")
                    logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            except Exception as e:
                logger.error(f"Erro durante treinamento do EfficientNet: {str(e)}", exc_info=True)
                trained_models['efficientnet'] = None
                
                # Liberar memória após erro
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após erro no treinamento do EfficientNet")
        else:
            logger.info("Pulando treinamento do EfficientNet.")
            # Tentar carregar modelo pré-treinado se disponível
            try:
                if USE_EFFICIENTNET:
                    efficient_model = models.create_model_with_best_params(efficientnet_best_params, model_name='efficientnet')
                    efficient_model.load_state_dict(torch.load('models/efficientnet_final.pth'))
                    trained_models['efficientnet'] = efficient_model
                    logger.info("Modelo EfficientNet pré-treinado carregado com sucesso.")
            except Exception as e:
                logger.warning(f"Não foi possível carregar o modelo EfficientNet pré-treinado: {str(e)}")
                trained_models['efficientnet'] = None

        # Treinar Vision Transformer
        if RUN_VIT_TRAINING and USE_VIT:
            logger.info("=== TREINAMENTO DO VISION TRANSFORMER ===")
            try:
                # Liberar memória antes do treinamento do Vision Transformer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info(f"Memória GPU liberada antes do treinamento do Vision Transformer")
                    logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                
                # Criar modelo ViT - IMPORTANTE: usar modelo consistente para treinamento e ensemble
                # Fixando modelo específico em vez de usar vit_best_params para garantir compatibilidade
                vit_model = models.create_vit_model({
                    'model_name': 'vit_base_patch16_224',  # Usando modelo específico para consistência
                    'pretrained': True,  # Pretrained é OK durante a inicialização para treinamento
                    'dropout_rate': vit_best_params.get('dropout_rate', 0.1)
                })

                # Ativar gradient checkpointing para o modelo
                if utils.enable_gradient_checkpointing(vit_model):
                    logger.info(f"Gradient checkpointing ativado para Vision Transformer")
                
                # Definir critério de perda
                label_smoothing = vit_best_params.get('label_smoothing', 0.1)
                criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                
                # Configurar otimizador e scheduler
                vit_optimizer, vit_scheduler, vit_scheduler_name = models.create_optimizer_and_scheduler(
                    vit_model, vit_best_params, train_loader_vit
                )
                
                # Criar writer específico para este modelo
                vit_writer = SummaryWriter(log_dir=os.path.join(log_dir, "vit"))

                # Realizar profiling antes do treinamento completo
                if RUN_PROFILING:
                    logger.info(f"Executando profiling para o modelo Vision Transformer")
                    prof = utils.profile_model_training(vit_model, train_loader_vit, num_steps=10)
                    logger.info("Profiling completo. Resultados disponíveis no TensorBoard.")
                
                # Parâmetros para treinamento
                use_mixup = vit_best_params.get('use_mixup', True)
                use_cutmix = vit_best_params.get('use_cutmix', True)
                early_stopping = vit_best_params.get('early_stopping', True)
                patience = vit_best_params.get('patience_stopping', 3)
                
                # Treinar modelo
                logger.info(f"Iniciando treinamento do Vision Transformer com {NUM_EPOCHS} épocas...")
                vit_model, vit_train_losses, vit_train_accs, vit_val_losses, vit_val_accs = training.train_model_optimized(
                    vit_model, train_loader_vit, test_loader, criterion, vit_optimizer, vit_scheduler,
                    NUM_EPOCHS, vit_writer, "vit", use_mixup=use_mixup, use_cutmix=use_cutmix,
                    early_stopping=early_stopping, patience=patience, model_type='vit',
                    use_amp=True,                           # Usar precisão mista automática
                    use_gradient_checkpointing=True,        # Usar gradient checkpointing
                    dynamic_batch=True,                     # Permitir redução de batch size se OOM
                    min_batch_size=2                        # Tamanho mínimo de batch permitido para ViT

                )
                
                # Salvar modelo
                torch.save(vit_model.state_dict(), "models/vit_final.pth")
                
                # Armazenar para uso posterior
                trained_models['vit'] = vit_model
                training_history['vit'] = {
                    'train_losses': vit_train_losses,
                    'val_losses': vit_val_losses,
                    'train_accs': vit_train_accs,
                    'val_accs': vit_val_accs
                }
                
                # Visualizar curvas de aprendizado
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 2, 1)
                plt.plot(vit_train_losses, label='Treino')
                plt.plot(vit_val_losses, label='Validação')
                plt.title('Curva de Perda - Vision Transformer')
                plt.xlabel('Época')
                plt.ylabel('Perda')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(vit_train_accs, label='Treino')
                plt.plot(vit_val_accs, label='Validação')
                plt.title('Curva de Acurácia - Vision Transformer')
                plt.xlabel('Época')
                plt.ylabel('Acurácia')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig('results/vit_learning_curves.png')
                plt.close()
                
                logger.info("Treinamento do Vision Transformer concluído com sucesso!")
                
                # Liberar memória após treinamento
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após treinamento do Vision Transformer")
                    logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            except Exception as e:
                logger.error(f"Erro durante treinamento do Vision Transformer: {str(e)}", exc_info=True)
                trained_models['vit'] = None
                
                # Liberar memória após erro
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após erro no treinamento do Vision Transformer")
        else:
            logger.info("Pulando treinamento do Vision Transformer.")
            # Tentar carregar modelo pré-treinado se disponível
            try:
                if USE_VIT:
                    vit_params = {
                        'model_name': vit_best_params.get('model_type', 'vit_base_patch16_224'),
                        'pretrained': True,
                        'dropout_rate': vit_best_params.get('dropout_rate', 0.1)
                    }
                    vit_model = models.create_vit_model(vit_params)
                    vit_model.load_state_dict(torch.load('models/vit_final.pth'))
                    trained_models['vit'] = vit_model
                    logger.info("Modelo Vision Transformer pré-treinado carregado com sucesso.")
            except Exception as e:
                logger.warning(f"Não foi possível carregar o modelo Vision Transformer pré-treinado: {str(e)}")
                trained_models['vit'] = None
        
        # Treinar MobileNet com tratamento especial
        if RUN_MOBILENET_TRAINING and USE_MOBILENET:
            logger.info("=== TREINAMENTO SEGURO DO MOBILENET ===")
            try:
                # Liberar memória antes do treinamento do MobileNet
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info(f"Memória GPU liberada antes do treinamento do MobileNet")
                    logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                
                # Usar função especial de treinamento seguro para MobileNet
                logger.info("Iniciando treinamento seguro do MobileNet...")

                # Ativar gradient checkpointing para o modelo (antes de treinar)
                mobile_model = tv_models.mobilenet_v3_small(weights=None, num_classes=len(classes))
                if utils.enable_gradient_checkpointing(mobile_model):
                    logger.info(f"Gradient checkpointing ativado para MobileNet")

                # Realizar profiling se necessário
                if RUN_PROFILING:
                    logger.info(f"Executando profiling para o modelo MobileNet")
                    prof = utils.profile_model_training(mobile_model, 
                        DataLoader(cached_train_dataset_mobilenet, batch_size=BATCH_SIZE), 
                        num_steps=10)
                    logger.info("Profiling completo. Resultados disponíveis no TensorBoard.")

                mobile_model, mobile_train_losses, mobile_train_accs, mobile_val_losses, mobile_val_accs = models.safe_mobilenet_training(
                    cached_train_dataset_mobilenet, cached_test_dataset, mobilenet_best_params, model_type='cnn',
                    use_amp=True,
                    use_gradient_checkpointing=True
                )
                
                # Salvar modelo
                torch.save(mobile_model.state_dict(), "models/mobilenet_final.pth")
                
                # Armazenar para uso posterior
                trained_models['mobilenet'] = mobile_model
                training_history['mobilenet'] = {
                    'train_losses': mobile_train_losses,
                    'val_losses': mobile_val_losses,
                    'train_accs': mobile_train_accs,
                    'val_accs': mobile_val_accs
                }
                
                # Visualizar curvas de aprendizado
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 2, 1)
                plt.plot(mobile_train_losses, label='Treino')
                plt.plot(mobile_val_losses, label='Validação')
                plt.title('Curva de Perda - MobileNet')
                plt.xlabel('Época')
                plt.ylabel('Perda')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(mobile_train_accs, label='Treino')
                plt.plot(mobile_val_accs, label='Validação')
                plt.title('Curva de Acurácia - MobileNet')
                plt.xlabel('Época')
                plt.ylabel('Acurácia')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig('results/mobilenet_learning_curves.png')
                plt.close()
                
                logger.info("Treinamento do MobileNet concluído com sucesso!")
                
                # Liberar memória após treinamento
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após treinamento do MobileNet")
                    logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            except Exception as e:
                logger.error(f"Erro durante treinamento do MobileNet: {str(e)}", exc_info=True)
                trained_models['mobilenet'] = None
                
                # Liberar memória após erro
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Memória GPU liberada após erro no treinamento do MobileNet")
        else:
            logger.info("Pulando treinamento do MobileNet.")
            # Tentar carregar modelo pré-treinado se disponível
            try:
                if USE_MOBILENET:
                    mobile_model = tv_models.mobilenet_v3_small(weights=None, num_classes=len(classes))
                    mobile_model.load_state_dict(torch.load('models/mobilenet_final.pth'))
                    trained_models['mobilenet'] = mobile_model
                    logger.info("Modelo MobileNet pré-treinado carregado com sucesso.")
            except Exception as e:
                logger.warning(f"Não foi possível carregar o modelo MobileNet pré-treinado: {str(e)}")
                trained_models['mobilenet'] = None

        # Snapshot de memória antes de salvar o histórico
        memory_snapshot = utils.get_memory_snapshot()
        logger.info(f"Snapshot de memória após todos os treinamentos:")
        logger.info(f"GPU alocada: {memory_snapshot.get('cuda', {}).get('gpu_0', {}).get('allocated_gb', 0):.3f} GB")
        logger.info(f"GPU reservada: {memory_snapshot.get('cuda', {}).get('gpu_0', {}).get('reserved_gb', 0):.3f} GB")
        logger.info(f"RAM processo: {memory_snapshot['process_memory_gb']:.3f} GB")

        # Salvar histórico de treinamento
        serializable_history = {}
        for model, history in training_history.items():
            serializable_history[model] = {
                'train_losses': [float(loss) for loss in history['train_losses']],
                'val_losses': [float(loss) for loss in history['val_losses']],
                'train_accs': [float(acc) for acc in history['train_accs']],
                'val_accs': [float(acc) for acc in history['val_accs']]
            }
        
        with open('results/training_history.json', 'w') as f:
            json.dump(serializable_history, f, indent=2)
            
        # Liberar memória após todos os treinamentos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Memória GPU liberada após todos os treinamentos")
            logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

        #######################################################
        # ETAPA 6: AVALIAÇÃO DETALHADA DOS MODELOS
        #######################################################
        
        if RUN_MODEL_EVALUATION:
            logger.info("=== ETAPA 6: AVALIAÇÃO DETALHADA DOS MODELOS ===")
            
            # Liberar memória antes da avaliação
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada antes da avaliação dos modelos")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            
            # Criar writer específico para avaliação
            eval_writer = SummaryWriter(log_dir=os.path.join(log_dir, "evaluation"))
            
            print("Avaliando modelos...")
            evaluation_results = {}
            
            # Avaliar cada modelo
            for model_name, model in trained_models.items():
                if model is not None:
                    try:
                        logger.info(f"Avaliando {model_name}...")
                        
                        # Liberar memória antes da avaliação de cada modelo
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            logger.info(f"Memória GPU liberada antes da avaliação de {model_name}")
                            logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                        
                        if model_name == 'resnet50':
                            eval_loader = test_loader
                            model_writer = resnet_writer if 'resnet_writer' in locals() else eval_writer
                        elif model_name == 'efficientnet':
                            eval_loader = test_loader
                            model_writer = efficient_writer if 'efficient_writer' in locals() else eval_writer
                        elif model_name == 'mobilenet':
                            eval_loader = test_loader
                            model_writer = mobile_writer if 'mobile_writer' in locals() else eval_writer
                        elif model_name == 'vit':
                            eval_loader = test_loader
                            model_writer = vit_writer if 'vit_writer' in locals() else eval_writer
                        else:
                            eval_loader = test_loader
                            model_writer = eval_writer
                        
                        accuracy, report, conf_matrix, predictions, true_labels, probabilities = training.evaluate_model(
                            model, eval_loader, classes, model_writer, model_name
                        )
                        
                        # Salvar resultados
                        evaluation_results[model_name] = {
                            'accuracy': float(accuracy),
                            'report': report,
                            'conf_matrix': conf_matrix.tolist() if hasattr(conf_matrix, 'tolist') else conf_matrix
                        }
                        
                        # Armazenar para uso posterior
                        model_accuracies[model_name] = accuracy
                        model_reports[model_name] = report
                        
                        # Analisar casos de erro
                        logger.info(f"Analisando casos de erro para {model_name}...")
                        error_cases = training.analyze_error_cases(
                            model, eval_loader, classes, num_samples=10, writer=model_writer, model_name=model_name
                        )
                        
                        print(f"{model_name}: Acurácia = {accuracy:.4f}")
                        
                        # Mostrar métricas por classe
                        print(f"\nMétricas detalhadas por classe para {model_name}:")
                        for cls in classes:
                            precision = report[cls]['precision']
                            recall = report[cls]['recall']
                            f1 = report[cls]['f1-score']
                            support = report[cls]['support']
                            print(f"  {cls}: Precisão={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Amostras={support}")
                            
                        # Salvar matriz de confusão como imagem
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
                        plt.title(f'Matriz de Confusão - {model_name}')
                        plt.ylabel('Real')
                        plt.xlabel('Predito')
                        plt.tight_layout()
                        plt.savefig(f'results/{model_name}_confusion_matrix.png')
                        plt.close()
                        
                        # Liberar memória após avaliação de cada modelo
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            logger.info(f"Memória GPU liberada após avaliação de {model_name}")
                            logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                    except Exception as e:
                        logger.error(f"Erro durante avaliação do modelo {model_name}: {str(e)}", exc_info=True)
                        
                        # Liberar memória após erro
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            logger.info(f"Memória GPU liberada após erro na avaliação de {model_name}")
            
            # Comparar modelos
            if evaluation_results:
                # Salvar resultados em formato JSON
                with open('results/evaluation_results.json', 'w') as f:
                    json.dump(evaluation_results, f, indent=2)
                
                # Criar tabela comparativa
                print("\nComparação de modelos:")
                models_df = pd.DataFrame({
                    'Modelo': list(evaluation_results.keys()),
                    'Acurácia': [results['accuracy'] for results in evaluation_results.values()]
                })
                
                # Adicionar métricas por classe
                for cls in classes:
                    models_df[f"F1 ({cls})"] = [
                        results['report'][cls]['f1-score'] 
                        for results in evaluation_results.values()
                    ]
                
                # Ordenar por acurácia
                models_df = models_df.sort_values('Acurácia', ascending=False)
                print(models_df)
                
                # Visualizar comparação
                plt.figure(figsize=(12, 6))
                sns.barplot(x='Modelo', y='Acurácia', data=models_df)
                plt.title('Comparação de Acurácia por Modelo')
                plt.ylim(0, 1.0)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig("results/model_comparison.png")
                
                # Adicionar ao TensorBoard
                eval_writer.add_figure("Evaluation/model_comparison", plt.gcf(), 0)
                plt.close()
                
                # Visualizar F1-score por classe
                f1_data = models_df.melt(
                    id_vars=['Modelo'], 
                    value_vars=[f"F1 ({cls})" for cls in classes],
                    var_name='Classe', 
                    value_name='F1-Score'
                )
                f1_data['Classe'] = f1_data['Classe'].apply(lambda x: x[4:-1])  # Remover "F1 (" e ")"
                
                plt.figure(figsize=(14, 8))
                sns.barplot(x='Classe', y='F1-Score', hue='Modelo', data=f1_data)
                plt.title('F1-Score por Classe e Modelo')
                plt.ylim(0, 1.0)
                plt.xticks(rotation=45)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig("results/f1_comparison.png")
                
                # Adicionar ao TensorBoard
                eval_writer.add_figure("Evaluation/f1_comparison", plt.gcf(), 0)
                plt.close()
            
            # Fechar writer
            eval_writer.close()
            
            # Liberar memória após toda a avaliação
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada após toda a avaliação dos modelos")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        else:
            logger.info("Pulando etapa de avaliação.")
            try:
                # Tentar carregar resultados de avaliação anterior
                if os.path.exists('results/evaluation_results.json'):
                    with open('results/evaluation_results.json', 'r') as f:
                        evaluation_results = json.load(f)
                    
                    # Extrair acurácias e relatórios
                    for model_name, results in evaluation_results.items():
                        model_accuracies[model_name] = results['accuracy']
                        model_reports[model_name] = results['report']
                    
                    logger.info("Resultados de avaliação carregados de arquivo anterior.")
            except Exception as e:
                logger.warning(f"Erro ao carregar resultados de avaliação: {str(e)}")
        
        #######################################################
        # ETAPA 7: VISUALIZAÇÕES DE INTERPRETABILIDADE
        #######################################################
        
        if RUN_VISUALIZATION:
            logger.info("=== ETAPA 7: VISUALIZAÇÕES DE INTERPRETABILIDADE ===")
            
            # Liberar memória antes das visualizações
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada antes das visualizações de interpretabilidade")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            
            # Criar writer para visualizações
            vis_writer = SummaryWriter(log_dir=os.path.join(log_dir, "visualization"))
            
            # Aplicar Grad-CAM aos modelos CNN
            for model_name, model in trained_models.items():
                if model is None:
                    continue
                    
                try:
                    logger.info(f"Gerando visualizações para {model_name}...")
                    
                    # Liberar memória antes de visualizar cada modelo
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info(f"Memória GPU liberada antes das visualizações para {model_name}")
                        logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                    
                    if model_name == 'resnet50':
                        # Obter a última camada convolucional
                        target_layer = model.layer4[-1]
                        visualization.visualize_gradcam_batch(
                            model, test_loader, target_layer, classes, 
                            writer=vis_writer, model_name=model_name
                        )

                    elif model_name == 'efficientnet':
                        # Última camada convolucional do EfficientNet
                        if hasattr(model, 'features'):
                            target_layer = model.features[-1]
                            visualization.visualize_gradcam_batch(
                                model, test_loader, target_layer, classes, 
                                writer=vis_writer, model_name=model_name
                            )
                        
                    elif model_name == 'mobilenet':
                        # Última camada convolucional do MobileNet
                        if hasattr(model, 'features'):
                            target_layer = model.features[-1]
                            visualization.visualize_gradcam_batch(
                                model, test_loader, target_layer, classes, 
                                writer=vis_writer, model_name=model_name
                            )
                        
                    elif model_name == 'vit':
                        # Para ViT, usar mapas de atenção em vez de Grad-CAM
                        visualization.visualize_attention_maps(
                            model, test_loader, classes, 
                            writer=vis_writer, model_name=model_name
                        )
                    
                    # Liberar memória após visualizar cada modelo
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info(f"Memória GPU liberada após visualizações para {model_name}")
                        logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                        
                except Exception as e:
                    logger.error(f"Erro ao gerar visualizações para {model_name}: {str(e)}", exc_info=True)
                    
                    # Liberar memória após erro
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info(f"Memória GPU liberada após erro nas visualizações para {model_name}")
            
            # Fechar writer
            vis_writer.close()
            
            # Liberar memória após todas as visualizações
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada após todas as visualizações de interpretabilidade")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        else:
            logger.info("Pulando etapa de visualização.")
        
        #######################################################
        # ETAPA 8: COMPRESSÃO E OTIMIZAÇÃO DE MODELOS
        #######################################################
        
        if RUN_COMPRESSION:
            logger.info("=== ETAPA 8: COMPRESSÃO E OTIMIZAÇÃO DE MODELOS ===")
            
            # Liberar memória antes da compressão
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada antes da compressão de modelos")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            
            # Criar writer para compressão
            compression_writer = SummaryWriter(log_dir=os.path.join(log_dir, "compression"))
            
            # Dicionários para armazenar modelos comprimidos
            quantized_models = {}
            pruned_models = {}
            
            # Para cada modelo treinado, aplicar técnicas de compressão
            for model_name, model in trained_models.items():
                if model is None:
                    continue
                    
                logger.info(f"Aplicando técnicas de compressão ao modelo {model_name}...")
                
                try:
                    # Liberar memória antes de comprimir cada modelo
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info(f"Memória GPU liberada antes da compressão de {model_name}")
                        logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                    
                    # 1. Quantização
                    logger.info(f"Iniciando quantização do modelo {model_name}...")
                    
                    if model_name == 'vit':
                        # Quantização específica para ViT
                        quantized_model = model_compression.quantize_vit(model, test_loader, model_name)
                    else:
                        # Quantização padrão para CNNs
                        quantized_model = model_compression.quantize_model(model, test_loader, model_name)
                    
                    if quantized_model is not None:
                        quantized_models[model_name] = quantized_model
                        logger.info(f"Quantização do modelo {model_name} concluída com sucesso")
                    
                    # Liberar memória após quantização
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info(f"Memória GPU liberada após quantização de {model_name}")
                    
                    # 2. Pruning
                    logger.info(f"Iniciando pruning do modelo {model_name}...")
                    
                    if model_name == 'vit':
                        # Pruning específico de cabeças de atenção para ViT
                        pruning_amount = 0.2  # 20% das cabeças
                        pruned_model = model_compression.prune_attention_heads(model, prune_amount=pruning_amount, model_name=model_name)
                    else:
                        # Pruning padrão para CNNs
                        pruning_amount = 0.2  # 20% dos pesos
                        pruned_model = model_compression.run_model_pruning(model, test_loader, prune_amount=pruning_amount, model_name=model_name)
                    
                    if pruned_model is not None:
                        pruned_models[model_name] = pruned_model
                        logger.info(f"Pruning do modelo {model_name} concluído com sucesso")
                    
                    # Liberar memória após pruning
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info(f"Memória GPU liberada após pruning de {model_name}")
                    
                    # Adicionar visualizações no TensorBoard (comparação de tamanhos)
                    if quantized_model is not None and pruned_model is not None:
                        try:
                            model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024*1024)  # MB
                            
                            # Estimar tamanho do modelo quantizado
                            quant_size = 0
                            for name, param in quantized_model.named_parameters():
                                if param.dtype == torch.qint8:
                                    quant_size += param.numel() * 1  # INT8 = 1 byte
                                else:
                                    quant_size += param.numel() * 4  # FLOAT32 = 4 bytes
                            
                            quant_size = quant_size / (1024*1024)  # MB
                            
                            # Estimar tamanho do modelo podado
                            zeros = 0
                            total = 0
                            for name, param in pruned_model.named_parameters():
                                zeros += (param == 0).sum().item()
                                total += param.numel()
                            
                            pruned_size = model_size * (1 - (zeros / total))
                            
                            # Criar gráfico comparativo
                            labels = ['Original', 'Quantizado', 'Podado']
                            sizes = [model_size, quant_size, pruned_size]
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.bar(labels, sizes, color=['blue', 'green', 'orange'])
                            ax.set_title(f'Comparação de Tamanho do Modelo - {model_name}')
                            ax.set_ylabel('Tamanho (MB)')
                            ax.set_ylim(0, max(sizes) * 1.2)
                            
                            # Adicionar valores nas barras
                            for bar, size in zip(bars, sizes):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{size:.2f} MB',
                                        ha='center', va='bottom')
                            
                            plt.tight_layout()
                            compression_writer.add_figure(f"{model_name}/model_size_comparison", fig, 0)
                            plt.close(fig)
                            
                            # Adicionar métricas de compressão
                            quant_reduction = (model_size - quant_size) / model_size * 100
                            prune_reduction = (model_size - pruned_size) / model_size * 100
                            
                            compression_writer.add_scalar(f"{model_name}/quantization_reduction", quant_reduction, 0)
                            compression_writer.add_scalar(f"{model_name}/pruning_reduction", prune_reduction, 0)
                            
                            # Log sobre taxa de compressão
                            logger.info(f"Modelo {model_name} - Taxa de redução por quantização: {quant_reduction:.2f}%")
                            logger.info(f"Modelo {model_name} - Taxa de redução por pruning: {prune_reduction:.2f}%")
                            
                        except Exception as e:
                            logger.error(f"Erro durante análise de compressão do modelo {model_name}: {str(e)}", exc_info=True)
                
                except Exception as e:
                    logger.error(f"Erro durante compressão do modelo {model_name}: {str(e)}", exc_info=True)
                    
                    # Liberar memória após erro
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info(f"Memória GPU liberada após erro na compressão de {model_name}")
            
            # Exportar para ONNX se selecionado
            if RUN_EXPORT:
                logger.info("=== EXPORTANDO MODELOS PARA ONNX ===")
                
                for model_name, model in trained_models.items():
                    if model is None:
                        continue
                        
                    try:
                        # Liberar memória antes da exportação de cada modelo
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            logger.info(f"Memória GPU liberada antes da exportação de {model_name}")
                            logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                        
                        logger.info(f"Exportando modelo {model_name} para ONNX...")
                        onnx_path = model_compression.export_to_onnx(model, model_name)
                        
                        if onnx_path:
                            logger.info(f"Modelo {model_name} exportado com sucesso para {onnx_path}")
                            
                            # Se o modelo quantizado existir, também exportá-lo
                            if model_name in quantized_models:
                                quantized_onnx_path = model_compression.export_to_onnx(
                                    quantized_models[model_name], 
                                    f"{model_name}_quantized"
                                )
                                if quantized_onnx_path:
                                    logger.info(f"Modelo quantizado {model_name} exportado para {quantized_onnx_path}")
                        
                        # Liberar memória após exportação
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            logger.info(f"Memória GPU liberada após exportação de {model_name}")
                    
                    except Exception as e:
                        logger.error(f"Erro ao exportar modelo {model_name} para ONNX: {str(e)}", exc_info=True)
                        
                        # Liberar memória após erro
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            logger.info(f"Memória GPU liberada após erro na exportação de {model_name}")
            
            # Fechar writer
            compression_writer.close()
            
            # Liberar memória após todas as compressões
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada após todas as compressões e exportações")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        else:
            logger.info("Pulando etapas de compressão e exportação de modelos.")

        #######################################################
        # ETAPA 9: ENSEMBLE DE MODELOS
        ####################################################### 
        
        if RUN_ENSEMBLE:
            logger.info("=== ETAPA 9: ENSEMBLE DE MODELOS ===")
            
            # Liberar memória antes do ensemble
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada antes do ensemble de modelos")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    
            # Verificar modelos disponíveis no diretório 'models'
            available_model_paths = {
                'resnet50': 'models/resnet50_final.pth',
                'efficientnet': 'models/efficientnet_final.pth',
                'mobilenet': 'models/mobilenet_final.pth',
                'vit': 'models/vit_final.pth'
            }
    
            # Lista de modelos disponíveis para o ensemble
            ensemble_models = []
            ensemble_names = []
            test_loaders = []
    
            # Primeiro, adicionar modelos já carregados na memória
            for model_name, model in trained_models.items():
                if model is not None:
                    ensemble_models.append(model)
                    ensemble_names.append(model_name)
                    test_loaders.append(test_loader)
    
            # Depois, tentar carregar modelos adicionais do disco se não estiverem já na memória
            for model_name, model_path in available_model_paths.items():
                # Pular apenas se o modelo já está na lista, ignorando o status da flag
                if model_name in ensemble_names:
                    continue
    
                if os.path.exists(model_path):
                    try:
                        logger.info(f"Tentando carregar modelo {model_name} do disco para ensemble...")
                        
                        # Liberar memória antes de carregar cada modelo adicional
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            logger.info(f"Memória GPU liberada antes de carregar {model_name} para ensemble")
                            logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        
                        # Criar o modelo apropriado
                        if model_name == 'resnet50':
                            model = models.create_model_with_best_params(resnet_best_params, model_name='resnet50')
                        elif model_name == 'efficientnet':
                            model = models.create_model_with_best_params(efficientnet_best_params, model_name='efficientnet')
                        elif model_name == 'mobilenet':
                            model = models.mobilenet_v3_small(weights=None, num_classes=len(classes))
                        elif model_name == 'vit':
                            # Verificar o tipo de modelo usado durante o treinamento
                            logger.info("Criando modelo Vision Transformer com a mesma arquitetura que foi treinada...")
                            vit_params = {
                                # Use exatamente o mesmo tipo de modelo que foi usado no treinamento
                                'model_name': 'vit_base_patch16_224',  # Usar o mesmo nome que foi usado no treinamento
                                'pretrained': False,  # Importante: NÃO carregue pesos pré-treinados
                                'dropout_rate': vit_best_params.get('dropout_rate', 0.1)
                            }
                            model = models.create_vit_model(vit_params)

                            # Agora carregue os pesos salvos com tratamento especial
                            try:
                                # Primeiro, mova o modelo para a CPU para evitar problemas de memória
                                model = model.to('cpu')
                                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                                # Após carregar, mova para o dispositivo apropriado
                                model = model.to(device)
                                logger.info("Modelo ViT carregado com sucesso para o ensemble")
                            except Exception as e:
                                logger.warning(f"Erro ao carregar pesos do modelo ViT: {str(e)}")
                                # Se falhar ao carregar os pesos, pule este modelo
                                continue
        
                        # Adicionar à lista de ensemble
                        ensemble_models.append(model)
                        ensemble_names.append(model_name)
                        test_loaders.append(test_loader)
        
                        logger.info(f"Modelo {model_name} carregado com sucesso para ensemble.")
        
                        # Adicionar ao dicionário de modelos treinados se ainda não estiver lá
                        if model_name not in trained_models or trained_models[model_name] is None:
                            trained_models[model_name] = model
        
                        # Obter acurácia se não tiver sido avaliado
                        if model_name not in model_accuracies or model_accuracies[model_name] == 0:
                            logger.info(f"Avaliando modelo {model_name} para ensemble...")
                            
                            # Liberar memória antes da avaliação
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                gc.collect()
                                logger.info(f"Memória GPU liberada antes da avaliação de {model_name} para ensemble")
                                
                            accuracy, report, conf_matrix, _, _, _ = training.evaluate_model(
                                model, test_loader, classes, writer, model_name
                            )
                            model_accuracies[model_name] = accuracy
                            model_reports[model_name] = report
                            logger.info(f"Modelo {model_name}: acurácia = {accuracy:.4f}")
                            
                            # Liberar memória após avaliação
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                gc.collect()
                                logger.info(f"Memória GPU liberada após avaliação de {model_name} para ensemble")
        
                    except Exception as e:
                        logger.warning(f"Não foi possível carregar o modelo {model_name} para ensemble: {str(e)}")
                        
                        # Liberar memória após erro
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            logger.info(f"Memória GPU liberada após erro no carregamento de {model_name} para ensemble")
    
            if len(ensemble_models) > 1:
                logger.info(f"Criando ensemble com {len(ensemble_models)} modelos: {ensemble_names}")

                # Criar writer para ensemble
                ensemble_writer = SummaryWriter(log_dir=os.path.join(log_dir, "ensemble"))

                try:
                    # Liberar memória antes da avaliação do ensemble
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info("Memória GPU liberada antes da avaliação do ensemble")
                        logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                    
                    # Avaliar o ensemble
                    ensemble_accuracy, ensemble_report, ensemble_conf_matrix, ensemble_preds, ensemble_labels = ensemble.evaluate_ensemble_with_memory_management(
                        ensemble_models, test_loaders, classes, 
                        model_names=ensemble_names, writer=ensemble_writer,
                        batch_size=BATCH_SIZE // 2,  # Reduzir batch size para avaliação
                        use_amp=True  # Usar AMP para avaliação
                    )

                    logger.info(f"Acurácia do ensemble: {ensemble_accuracy:.4f}")

                    # Salvar metadados do ensemble
                    ensemble.save_ensemble_metadata(ensemble_models, ensemble_names, save_path='models/ensemble_metadata.pt')

                    # Mostrar comparação com modelos individuais
                    print("\nComparação entre modelos individuais e ensemble:")
                    print(f"Ensemble: {ensemble_accuracy:.4f}")
                    for name, acc in model_accuracies.items():
                        if acc > 0:  # Apenas modelos válidos
                            print(f"{name}: {acc:.4f}")

                    # Salvar resultados do ensemble
                    utils.save_ensemble_results(ensemble_accuracy, ensemble_names)

                    # Criar visualização com o ensemble
                    plt.figure(figsize=(10, 6))
                    all_accuracies = [model_accuracies.get(name, 0) for name in ensemble_names]
                    all_accuracies.append(ensemble_accuracy)
                    all_names = ensemble_names + ['Ensemble']

                    bars = plt.bar(all_names, all_accuracies, color='skyblue')
                    bars[-1].set_color('red')  # Destacar o ensemble

                    plt.title('Comparação de Acurácia: Modelos vs Ensemble')
                    plt.ylabel('Acurácia')
                    plt.ylim(0, 1.0)
                    plt.xticks(rotation=45)

                    # Adicionar valores nas barras
                    for bar, acc in zip(bars, all_accuracies):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{acc:.4f}', ha='center', va='bottom')

                    plt.tight_layout()
                    plt.savefig('results/ensemble_comparison.png')
                    ensemble_writer.add_figure("Ensemble/comparison", plt.gcf(), 0)
                    plt.close()

                    # Fechar writer
                    ensemble_writer.close()
                    
                    # Liberar memória após avaliação do ensemble
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info("Memória GPU liberada após avaliação do ensemble")
                        logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

                except Exception as e:
                    logger.error(f"Erro durante a avaliação do ensemble: {str(e)}", exc_info=True)
                    
                    # Liberar memória após erro
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info("Memória GPU liberada após erro na avaliação do ensemble")
            else:
                logger.info("Apenas um modelo disponível. Criando ensemble com modelos pré-treinados do torchvision...")
    
                # Adicionar modelos pré-treinados do torchvision para complementar o ensemble
                try:
                    # Liberar memória antes de carregar modelos pré-treinados
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info("Memória GPU liberada antes de carregar modelos pré-treinados")
                        logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        
                    # Adicionar ResNet50 pré-treinado
                    if 'resnet50' not in ensemble_names:
                        resnet_model = tv_models.resnet50(weights='IMAGENET1K_V2')
                        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, len(classes))
                        resnet_model = resnet_model.to(device)
                        ensemble_models.append(resnet_model)
                        ensemble_names.append('resnet50_pretrained')
                        test_loaders.append(test_loader)
                        logger.info("ResNet50 pré-treinado adicionado ao ensemble")
                        
                        # Liberar memória após carregar modelo
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            logger.info("Memória GPU liberada após carregar ResNet50 pré-treinado")
        
                    # Adicionar EfficientNet pré-treinado
                    if 'efficientnet' not in ensemble_names:
                        efficient_model = tv_models.efficientnet_b0(weights='IMAGENET1K_V1')
                        efficient_model.classifier[1] = nn.Linear(efficient_model.classifier[1].in_features, len(classes))
                        efficient_model = efficient_model.to(device)
                        ensemble_models.append(efficient_model)
                        ensemble_names.append('efficientnet_pretrained')
                        test_loaders.append(test_loader)
                        logger.info("EfficientNet pré-treinado adicionado ao ensemble")
                        
                        # Liberar memória após carregar modelo
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            logger.info("Memória GPU liberada após carregar EfficientNet pré-treinado")

                    # Adicionar MobileNet pré-treinado
                    if 'mobilenet' not in ensemble_names:
                        mobile_model = tv_models.mobilenet_v3_small(weights='IMAGENET1K_V1')
                        mobile_model.classifier[3] = nn.Linear(mobile_model.classifier[3].in_features, len(classes))
                        mobile_model = mobile_model.to(device)
                        ensemble_models.append(mobile_model)
                        ensemble_names.append('mobilenet_pretrained')
                        test_loaders.append(test_loader)
                        logger.info("MobileNet pré-treinado adicionado ao ensemble")
                        
                        # Liberar memória após carregar modelo
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            logger.info("Memória GPU liberada após carregar MobileNet pré-treinado")
        
                    # Agora que adicionamos modelos pré-treinados, podemos criar o ensemble
                    if len(ensemble_models) > 1:
                        logger.info(f"Criando ensemble com {len(ensemble_models)} modelos: {ensemble_names}")
                        
                        # Liberar memória antes de criar o ensemble
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            logger.info("Memória GPU liberada antes de criar o ensemble com modelos pré-treinados")
                            logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            
                        # Criar writer para ensemble
                        ensemble_writer = SummaryWriter(log_dir=os.path.join(log_dir, "ensemble"))
            
                        # Avaliar o ensemble
                        ensemble_accuracy, ensemble_report, ensemble_conf_matrix, ensemble_preds, ensemble_labels = ensemble.evaluate_ensemble_with_memory_management(
                            ensemble_models, test_loaders, classes, 
                            model_names=ensemble_names
                        )
            
                        logger.info(f"Acurácia do ensemble: {ensemble_accuracy:.4f}")
                        
                        # Resto do código para visualização e relatórios do ensemble...
                        # Igual à parte anterior onde o ensemble é criado com modelos normais
                        
                        # Mostrar comparação com modelos individuais
                        print("\nComparação entre modelos individuais e ensemble:")
                        print(f"Ensemble: {ensemble_accuracy:.4f}")
                        for name, acc in model_accuracies.items():
                            if acc > 0:  # Apenas modelos válidos
                                print(f"{name}: {acc:.4f}")

                        # Salvar resultados do ensemble
                        utils.save_ensemble_results(ensemble_accuracy, ensemble_names)

                        # Criar visualização com o ensemble
                        plt.figure(figsize=(10, 6))
                        all_accuracies = [model_accuracies.get(name, 0) for name in ensemble_names]
                        all_accuracies.append(ensemble_accuracy)
                        all_names = ensemble_names + ['Ensemble']

                        bars = plt.bar(all_names, all_accuracies, color='skyblue')
                        bars[-1].set_color('red')  # Destacar o ensemble

                        plt.title('Comparação de Acurácia: Modelos vs Ensemble')
                        plt.ylabel('Acurácia')
                        plt.ylim(0, 1.0)
                        plt.xticks(rotation=45)

                        # Adicionar valores nas barras
                        for bar, acc in zip(bars, all_accuracies):
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                    f'{acc:.4f}', ha='center', va='bottom')

                        plt.tight_layout()
                        plt.savefig('results/ensemble_comparison.png')
                        ensemble_writer.add_figure("Ensemble/comparison", plt.gcf(), 0)
                        plt.close()
            
                        # Fechar writer
                        ensemble_writer.close()
                        
                        # Liberar memória após ensemble com modelos pré-treinados
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            logger.info("Memória GPU liberada após ensemble com modelos pré-treinados")
                            logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                    else:
                        logger.warning("Não foi possível adicionar modelos pré-treinados para o ensemble.")
                except Exception as e:
                    logger.error(f"Erro ao adicionar modelos pré-treinados para ensemble: {str(e)}")
                    logger.warning("Não há modelos suficientes para criar um ensemble (mínimo de 2 modelos necessários)")
                    
                    # Liberar memória após erro
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info("Memória GPU liberada após erro na criação do ensemble com modelos pré-treinados")
            
            # Liberar memória após toda a etapa de ensemble
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada após toda a etapa de ensemble")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        
        #######################################################
        # ETAPA 10: RELATÓRIO FINAL E CONCLUSÕES
        #######################################################
        
        if RUN_FINAL_REPORT:
            logger.info("=== ETAPA 10: RELATÓRIO FINAL E CONCLUSÕES ===")
            
            # Liberar memória antes do relatório final
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada antes do relatório final")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            
            # Gerar resumo do experimento
            print("="*50)
            print(f"RESUMO DO EXPERIMENTO {EXPERIMENT_ID}")
            print("="*50)
            
            # Listar modelos treinados
            print("\nModelos treinados:")
            for model_name, model in trained_models.items():
                if model is not None:
                    accuracy = model_accuracies.get(model_name, 0.0)
                    status = "[OK]" if accuracy > 0 else "[FALHA]"
                    print(f"  {status} {model_name}: Acurácia = {accuracy:.4f}")
                else:
                    print(f"  ✗ {model_name}: Não treinado")
            
            # Resumo das etapas executadas
            print("\nEtapas executadas:")
            steps = [
                ("Análise exploratória de dados", RUN_DATA_EXPLORATION),
                ("Preparação de datasets", RUN_DATASET_PREP),
                ("Validação cruzada", RUN_CROSS_VALIDATION),
                ("Otimização de hiperparâmetros (ResNet)", RUN_RESNET_OPTIMIZATION),
                ("Otimização de hiperparâmetros (EfficientNet)", RUN_EFFICIENTNET_OPTIMIZATION),
                ("Otimização de hiperparâmetros (MobileNet)", RUN_MOBILENET_OPTIMIZATION),
                ("Otimização de hiperparâmetros (ViT)", RUN_VIT_OPTIMIZATION),
                ("Treinamento de modelos", any([RUN_RESNET_TRAINING, RUN_EFFICIENTNET_TRAINING, RUN_MOBILENET_TRAINING, RUN_VIT_TRAINING])),
                ("Avaliação detalhada", RUN_MODEL_EVALUATION),
                ("Visualizações de interpretabilidade", RUN_VISUALIZATION),
                ("Compressão de modelos", RUN_COMPRESSION),
                ("Ensemble de modelos", RUN_ENSEMBLE),
                ("Exportação para ONNX", RUN_EXPORT)
            ]
            
            for step_name, executed in steps:
                status = "[OK]" if executed else "[FALHA]"
                print(f"  {status} {step_name}")
            
            # Informações sobre o ambiente
            print("\nInformações do ambiente:")
            print(f"  Dispositivo: {device}")
            print(f"  Versão PyTorch: {torch.__version__}")
            
            # Resumo do conjunto de dados
            print("\nInformações do conjunto de dados:")
            print(f"  Total de imagens de treinamento: {len(df_train)}")
            print(f"  Total de imagens de teste: {len(df_test)}")
            print(f"  Classes: {classes}")
            
            # Melhor modelo
            if model_accuracies:
                best_model_name = max(model_accuracies, key=model_accuracies.get)
                best_model_acc = model_accuracies[best_model_name]
                print(f"\nMelhor modelo individual: {best_model_name} (Acurácia: {best_model_acc:.4f})")
                
                # Informações sobre o ensemble se disponível
                if RUN_ENSEMBLE and 'ensemble_accuracy' in locals():
                    ensemble_gain = ensemble_accuracy - best_model_acc
                    print(f"Ensemble: Acurácia = {ensemble_accuracy:.4f} (Ganho: {ensemble_gain*100:.2f}%)")
            
            # Criar visualização final agrupando os principais resultados
            try:
                # Verificar se os resultados existem
                if model_accuracies:
                    # Preparar dados para visualização
                    fig = plt.figure(figsize=(15, 10))
                    fig.suptitle(f"Resumo do Experimento {EXPERIMENT_ID}", fontsize=16)
                    
                    # 1. Acurácia por modelo
                    plt.subplot(2, 2, 1)
                    model_names = list(model_accuracies.keys())
                    accuracies = [model_accuracies[name] for name in model_names]
                    
                    # Adicionar ensemble se disponível
                    if RUN_ENSEMBLE and 'ensemble_accuracy' in locals():
                        model_names.append('Ensemble')
                        accuracies.append(ensemble_accuracy)
                    
                    colors = ['skyblue'] * len(model_names)
                    # Destacar melhor modelo e ensemble
                    best_idx = accuracies.index(max(accuracies))
                    colors[best_idx] = 'red'
                    
                    plt.bar(model_names, accuracies, color=colors)
                    plt.title('Acurácia por Modelo')
                    plt.ylabel('Acurácia')
                    plt.ylim(0, 1.0)
                    plt.xticks(rotation=45)
                    
                    # 2. Matriz de confusão do melhor modelo
                    plt.subplot(2, 2, 2)
                    best_model = model_names[best_idx] if 'Ensemble' not in model_names else model_names[best_idx]
                    
                    try:
                        if os.path.exists(f'results/{best_model}_confusion_matrix.png'):
                            conf_matrix_img = plt.imread(f'results/{best_model}_confusion_matrix.png')
                            plt.imshow(conf_matrix_img)
                            plt.title(f'Matriz de Confusão - {best_model}')
                            plt.axis('off')
                        else:
                            plt.text(0.5, 0.5, f"Matriz de confusão não disponível para {best_model}", 
                                     ha='center', va='center', transform=plt.gca().transAxes)
                    except Exception as e:
                        print(f"Erro ao carregar matriz de confusão: {str(e)}")
                        plt.text(0.5, 0.5, "Erro ao carregar matriz de confusão", 
                                 ha='center', va='center', transform=plt.gca().transAxes)
                    
                    # 3. Curvas de aprendizado do melhor modelo
                    plt.subplot(2, 2, 3)
                    if best_model in training_history and len(training_history[best_model]['train_losses']) > 0:
                        history = training_history[best_model]
                        epochs = len(history['train_losses'])
                        plt.plot(range(epochs), history['train_losses'], label='Treino')
                        plt.plot(range(epochs), history['val_losses'], label='Validação')
                        plt.title(f'Curva de Perda - {best_model}')
                        plt.xlabel('Época')
                        plt.ylabel('Perda')
                        plt.legend()
                    else:
                        plt.text(0.5, 0.5, "Histórico de treinamento não disponível", 
                                 ha='center', va='center', transform=plt.gca().transAxes)
                    
                    # 4. Exemplos de visualização de interpretabilidade
                    plt.subplot(2, 2, 4)
                    try:
                        # Tentar carregar uma visualização Grad-CAM ou mapa de atenção
                        vis_path = None
                        
                        if os.path.exists(f'gradcam_{best_model}/gradcam_summary_{best_model}.png'):
                            vis_path = f'gradcam_{best_model}/gradcam_summary_{best_model}.png'
                        elif os.path.exists(f'attention_maps_{best_model}/attention_summary_{best_model}.png'):
                            vis_path = f'attention_maps_{best_model}/attention_summary_{best_model}.png'
                        
                        if vis_path:
                            vis_img = plt.imread(vis_path)
                            plt.imshow(vis_img)
                            plt.title(f'Visualização de Interpretabilidade - {best_model}')
                            plt.axis('off')
                        else:
                            plt.text(0.5, 0.5, "Visualização de interpretabilidade não disponível", 
                                     ha='center', va='center', transform=plt.gca().transAxes)
                    except Exception as e:
                        logger.error(f"Erro ao carregar visualização: {str(e)}")
                        plt.text(0.5, 0.5, "Erro ao carregar visualização", 
                                 ha='center', va='center', transform=plt.gca().transAxes)
                    
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(f'results/experiment_summary_{EXPERIMENT_ID}.png', dpi=150)
                    plt.close()
                    
                    logger.info(f"Visualização de resumo salva em results/experiment_summary_{EXPERIMENT_ID}.png")

                    # Também salvar resumo em arquivo de texto
                    summary_path = os.path.join('results', f'experiment_summary_{EXPERIMENT_ID}.txt')
                    with open(summary_path, 'w', encoding='utf-8') as f:
                        f.write(f"RESUMO DO EXPERIMENTO {EXPERIMENT_ID}\n")
                        f.write("="*50 + "\n\n")
                        
                        f.write("Modelos treinados:\n")
                        for model_name, model in trained_models.items():
                            if model is not None:
                                accuracy = model_accuracies.get(model_name, 0.0)
                                status = "[OK]" if accuracy > 0 else "[FALHA]"
                                f.write(f"  {status} {model_name}: Acurácia = {accuracy:.4f}\n")
                            else:
                                f.write(f"  ✗ {model_name}: Não treinado\n")
                        
                        f.write("\nEtapas executadas:\n")
                        steps = [
                            ("Análise exploratória de dados", RUN_DATA_EXPLORATION),
                            ("Validação cruzada", RUN_CROSS_VALIDATION),
                            ("Otimização de hiperparâmetros", any([RUN_RESNET_OPTIMIZATION, RUN_EFFICIENTNET_OPTIMIZATION, 
                                                                 RUN_MOBILENET_OPTIMIZATION, RUN_VIT_OPTIMIZATION])),
                            ("Treinamento de modelos", any([RUN_RESNET_TRAINING, RUN_EFFICIENTNET_TRAINING, 
                                                          RUN_MOBILENET_TRAINING, RUN_VIT_TRAINING])),
                            ("Avaliação detalhada", RUN_MODEL_EVALUATION),
                            ("Visualizações de interpretabilidade", RUN_VISUALIZATION),
                            ("Compressão de modelos", RUN_COMPRESSION),
                            ("Ensemble de modelos", RUN_ENSEMBLE),
                            ("Exportação para ONNX", RUN_EXPORT)
                        ]
                        
                        for step_name, executed in steps:
                            status = "[OK]" if executed else "[FALHA]"
                            f.write(f"  {status} {step_name}\n")
                        
                        f.write("\nInformações do ambiente:\n")
                        f.write(f"  Dispositivo: {device}\n")
                        f.write(f"  Versão PyTorch: {torch.__version__}\n")
                        
                        f.write("\nInformações do conjunto de dados:\n")
                        f.write(f"  Total de imagens de treinamento: {len(df_train)}\n")
                        f.write(f"  Total de imagens de teste: {len(df_test)}\n")
                        f.write(f"  Classes: {classes}\n")
                        
                        if model_accuracies:
                            best_model_name = max(model_accuracies, key=model_accuracies.get)
                            best_model_acc = model_accuracies[best_model_name]
                            f.write(f"\nMelhor modelo individual: {best_model_name} (Acurácia: {best_model_acc:.4f})\n")
                            
                            if RUN_ENSEMBLE and 'ensemble_accuracy' in locals():
                                ensemble_gain = ensemble_accuracy - best_model_acc
                                f.write(f"Ensemble: Acurácia = {ensemble_accuracy:.4f} (Ganho: {ensemble_gain*100:.2f}%)\n")
                    
                    logger.info(f"Resumo do experimento salvo em: {summary_path}")
                    
            except Exception as e:
                logger.error(f"Erro ao criar visualização de resumo: {str(e)}", exc_info=True)
                
            # Liberar memória após relatório final
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada após relatório final")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                
        # Fechar o logger principal
        logger.info("Pipeline concluído com sucesso! Experimento finalizado.")
    
    except Exception as e:
        print(f"Erro durante execução do pipeline: {str(e)}")
        if 'logger' in locals():
            logger.error(f"Erro fatal durante execução do pipeline: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
        
        # Liberar memória em caso de erro
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Memória GPU liberada após erro fatal")
            except:
                pass
    
    finally:
        # Fechar writers para evitar vazamento de recursos
        if 'writer' in locals() and writer is not None:
            writer.close()
            
        # Fechar writers específicos de modelos
        for writer_name in ['resnet_writer', 'efficient_writer', 'vit_writer', 'mobile_writer']:
            if writer_name in locals() and locals()[writer_name] is not None:
                try:
                    locals()[writer_name].close()
                except:
                    pass
        
        # Fechar writers específicos de etapa
        for writer_name in ['eval_writer', 'vis_writer', 'compression_writer', 'ensemble_writer']:
            if writer_name in locals() and locals()[writer_name] is not None:
                try:
                    locals()[writer_name].close()
                except:
                    pass
        

        # Limpar memória CUDA
        if torch.cuda.is_available():
            try:
                # Relatório final de uso de memória
                memory_snapshot = utils.get_memory_snapshot()
                logger.info("Relatório final de uso de memória:")
                logger.info(f"GPU={memory_snapshot.get('cuda', {}).get('gpu_0', {}).get('allocated_gb', 0):.2f} GB (alocado)")
                logger.info(f"GPU={memory_snapshot.get('cuda', {}).get('gpu_0', {}).get('reserved_gb', 0):.2f} GB (reservado)")
                logger.info(f"RAM={memory_snapshot['ram_used_gb']:.2f} GB ({memory_snapshot['ram_percent']}%)")
                logger.info(f"Processo={memory_snapshot['process_memory_gb']:.2f} GB")
            except Exception as e:
                logger.warning(f"Erro ao gerar relatório final de memória: {str(e)}")
        
        # Liberar memória explicitamente
        try:
            del trained_models
            del model_accuracies
            del model_reports
            del training_history
        except:
            pass

        try:
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Memória CUDA final liberada")
            logger.info(f"Memória GPU final alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        except:
            pass
        
        logger.info("Recursos liberados e pipeline finalizado.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erro durante execução do pipeline: {str(e)}")
        import logging
        from utils import setup_logger
        
        # Configura o logger apenas durante o tratamento da exceção
        logger = setup_logger(log_dir="logs", log_level=logging.ERROR)
        logger.error(f"Erro fatal durante execução do pipeline: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()