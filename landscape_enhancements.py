"""
Módulo para integração das melhorias propostas ao sistema de classificação de paisagens.

Este módulo integra as seguintes melhorias ao pipeline existente:
1. Detecção de imagens duplicadas usando imagehash
2. Modelo leve NatureLightNet com destilação de conhecimento 
3. Otimização hierárquica de hiperparâmetros
4. Amostragem inteligente para redução de dados de treinamento
"""

import os
import torch
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import yaml
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import imagehash
from PIL import Image
from tqdm import tqdm

# Configurar logger
logger = logging.getLogger("landscape_classifier")

def detect_duplicate_images(dataframe, hash_diff_threshold=3):
    """
    Detecta imagens duplicadas ou quase idênticas no dataset usando imagehash.
    
    Args:
        dataframe: DataFrame pandas com coluna 'image_path' e 'image_hash'
        hash_diff_threshold: Limiar para considerar imagens similares (menor = mais restritivo)
        
    Returns:
        DataFrame com grupos de imagens duplicadas/similares
    """
    logger.info(f"Iniciando detecção de imagens duplicadas com limiar {hash_diff_threshold}...")
    
    # Verificar se a coluna 'image_hash' existe
    if 'image_hash' not in dataframe.columns:
        logger.error("Coluna 'image_hash' não encontrada no DataFrame. Execute create_df primeiro.")
        return None
    
    # Criar dicionário para agrupar imagens por hash
    similar_images = {}
    duplicates_count = 0
    
    # Para cada imagem no dataframe
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Comparando hashes"):
        img_path = row['image_path']
        img_hash = row['image_hash']
        
        if pd.isna(img_hash) or img_hash is None:
            continue
            
        img_hash_obj = imagehash.hex_to_hash(img_hash) if isinstance(img_hash, str) else img_hash
        
        # Verificar similaridade com hashes já processados
        found_similar = False
        for existing_hash in list(similar_images.keys()):
            existing_hash_obj = imagehash.hex_to_hash(existing_hash) if isinstance(existing_hash, str) else existing_hash
            
            # Calcular diferença de hash
            hash_diff = abs(img_hash_obj - existing_hash_obj)
            
            if hash_diff <= hash_diff_threshold:
                similar_images[existing_hash].append((img_path, hash_diff))
                found_similar = True
                duplicates_count += 1
                break
        
        if not found_similar:
            similar_images[img_hash] = [(img_path, 0)]  # Primeira imagem deste hash
    
    # Filtrar apenas grupos com mais de uma imagem (duplicatas)
    duplicate_groups = {h: imgs for h, imgs in similar_images.items() if len(imgs) > 1}
    
    # Criar dataframe com resultados
    result_data = []
    for hash_val, image_list in duplicate_groups.items():
        for img_path, diff in image_list:
            result_data.append({
                'hash': str(hash_val),
                'image_path': img_path,
                'similarity_diff': diff,
                'group_size': len(image_list)
            })
    
    # Criar dataframe com o resultado
    if result_data:
        result_df = pd.DataFrame(result_data)
        logger.info(f"Encontradas {duplicates_count} imagens similares em {len(duplicate_groups)} grupos")
        return result_df
    else:
        logger.info("Nenhuma imagem duplicada encontrada")
        return pd.DataFrame(columns=['hash', 'image_path', 'similarity_diff', 'group_size'])

def visualize_duplicates(duplicates_df, num_groups=3, output_dir="results/duplicates"):
    """
    Visualiza grupos de imagens duplicadas encontradas.
    
    Args:
        duplicates_df: DataFrame com informações sobre duplicatas
        num_groups: Número de grupos a visualizar
        output_dir: Diretório para salvar as visualizações
    
    Returns:
        bool: True se visualizações foram geradas com sucesso
    """
    if duplicates_df is None or len(duplicates_df) == 0:
        logger.info("Nenhum grupo de duplicatas para visualizar")
        return False
    
    try:
        # Criar diretório de saída
        os.makedirs(output_dir, exist_ok=True)
        
        # Obter grupos de hashes únicos
        groups = duplicates_df['hash'].unique()
        
        # Limitar ao número de grupos solicitado
        groups = groups[:min(num_groups, len(groups))]
        
        # Para cada grupo
        for group_idx, group_hash in enumerate(groups):
            # Filtrar imagens deste grupo
            group_images = duplicates_df[duplicates_df['hash'] == group_hash]
            
            # Ordenar por diferença (as mais similares primeiro)
            group_images = group_images.sort_values('similarity_diff')
            
            # Número de imagens neste grupo (limitado a 4 para visualização)
            num_images = min(4, len(group_images))
            
            plt.figure(figsize=(num_images * 4, 4))
            
            # Mostrar cada imagem
            for i, (_, row) in enumerate(group_images.head(num_images).iterrows()):
                plt.subplot(1, num_images, i + 1)
                
                try:
                    img = Image.open(row['image_path']).convert('RGB')
                    plt.imshow(img)
                    plt.title(f"Diff: {row['similarity_diff']}")
                    plt.axis('off')
                except Exception as e:
                    plt.text(0.5, 0.5, f"Erro: {str(e)}", ha='center', va='center')
            
            # Título global
            plt.suptitle(f"Grupo {group_idx+1}: Hash {group_hash[:8]}... ({group_images['group_size'].iloc[0]} imagens)")
            plt.tight_layout()
            
            # Salvar figura
            output_path = os.path.join(output_dir, f"duplicate_group_{group_idx+1}.png")
            plt.savefig(output_path)
            logger.info(f"Visualização do grupo {group_idx+1} salva em {output_path}")
            plt.close()
        
        # Criar resumo
        plt.figure(figsize=(10, 6))
        
        # Contar tamanhos dos grupos
        group_sizes = duplicates_df.groupby('hash').size().value_counts().sort_index()
        
        # Plotar histograma dos tamanhos dos grupos
        plt.bar(group_sizes.index, group_sizes.values)
        plt.xlabel("Tamanho do grupo")
        plt.ylabel("Número de grupos")
        plt.title("Distribuição de tamanhos de grupos de imagens similares")
        plt.grid(axis='y', alpha=0.3)
        
        # Adicionar rótulos nas barras
        for x, y in zip(group_sizes.index, group_sizes.values):
            plt.text(x, y + 0.1, str(y), ha='center')
        
        # Salvar resumo
        summary_path = os.path.join(output_dir, "duplicate_summary.png")
        plt.savefig(summary_path)
        logger.info(f"Resumo salvo em {summary_path}")
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Erro ao visualizar duplicatas: {str(e)}")
        return False

def filter_duplicates(df_train, duplicates_df, strategy='first'):
    """
    Filtra o DataFrame de treinamento para remover imagens duplicadas.
    
    Args:
        df_train: DataFrame original de treinamento
        duplicates_df: DataFrame com informações de duplicatas
        strategy: Estratégia de filtragem ('first', 'best_quality', 'random')
        
    Returns:
        DataFrame filtrado sem duplicatas
    """
    if duplicates_df is None or len(duplicates_df) == 0:
        logger.info("Nenhuma duplicata para remover")
        return df_train
    
    # Para cada grupo de duplicatas, manter apenas uma imagem
    unique_images = []
    removed_images = []
    
    # Agrupar por hash
    for hash_val in duplicates_df['hash'].unique():
        group_df = duplicates_df[duplicates_df['hash'] == hash_val]
        
        if strategy == 'first':
            # Manter a primeira imagem de cada grupo
            keep_image = group_df.iloc[0]['image_path']
        elif strategy == 'best_quality':
            # Manter a imagem com menor diferença (mais próxima do hash de referência)
            keep_image = group_df.loc[group_df['similarity_diff'].idxmin()]['image_path']
        elif strategy == 'random':
            # Manter uma imagem aleatória do grupo
            keep_image = group_df.sample(1).iloc[0]['image_path']
        else:
            # Estratégia padrão: manter a primeira
            keep_image = group_df.iloc[0]['image_path']
        
        unique_images.append(keep_image)
        
        # Registrar imagens removidas
        removed_list = group_df[group_df['image_path'] != keep_image]['image_path'].tolist()
        removed_images.extend(removed_list)
    
    # Filtrar DataFrame original
    df_filtered = df_train[~df_train['image_path'].isin(removed_images)]
    
    # Atualizar estatísticas
    reduction = len(df_train) - len(df_filtered)
    reduction_percent = reduction / len(df_train) * 100
    
    logger.info(f"Dataset reduzido de {len(df_train)} para {len(df_filtered)} imagens")
    logger.info(f"Redução de {reduction_percent:.2f}% ({reduction} imagens removidas)")
    
    return df_filtered

def initialize_naturelight_model(num_classes=6, device=None):
    """
    Inicializa e retorna um modelo NatureLightNet.
    
    Args:
        num_classes: Número de classes para classificação
        device: Dispositivo para processamento (CPU/GPU)
        
    Returns:
        tuple: (modelo, métricas de redução)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Importar o modelo NatureLightNet
        from NatureLightNet import create_naturelight_model
        
        # Criar instância do modelo
        naturelight_model = create_naturelight_model(num_classes=num_classes)
        naturelight_model = naturelight_model.to(device)
        
        # Obter contagens de parâmetros
        trainable_params, total_params = naturelight_model.get_params_count()
        
        # Comparar com MobileNet para calcular redução
        import torchvision.models as tvm
        mobilenet = tvm.mobilenet_v3_small(weights=None, num_classes=num_classes)
        mobilenet_params = sum(p.numel() for p in mobilenet.parameters())
        
        # Calcular métricas de redução
        reduction_percent = (1 - trainable_params / mobilenet_params) * 100
        
        logger.info(f"NatureLightNet criado com {trainable_params:,} parâmetros")
        logger.info(f"MobileNetV3-Small tem {mobilenet_params:,} parâmetros")
        logger.info(f"Redução de parâmetros: {reduction_percent:.1f}%")
        
        metrics = {
            'params': int(trainable_params),
            'mobilenet_params': int(mobilenet_params),
            'reduction_percent': float(reduction_percent),
            'reduction_ratio': float(mobilenet_params / trainable_params)
        }
        
        return naturelight_model, metrics
    
    except ImportError:
        logger.error("Módulo NatureLightNet não encontrado. Verifique se está no PYTHONPATH.")
        return None, None
    except Exception as e:
        logger.error(f"Erro ao inicializar NatureLightNet: {str(e)}")
        return None, None

def apply_knowledge_distillation(df_train, df_test, num_classes=6, device=None, num_epochs=5):
    """
    Aplica destilação de conhecimento do MobileNet para o NatureLightNet.
    
    Args:
        df_train: DataFrame com dados de treinamento
        df_test: DataFrame com dados de teste
        num_classes: Número de classes para classificação
        device: Dispositivo para processamento (CPU/GPU)
        num_epochs: Número de épocas para destilação
        
    Returns:
        tuple: (modelo_destilado, histórico_treinamento)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Importar função de destilação
        from knowledge_distillation import train_naturelight_with_distillation
        
        # Criar datasets
        from dataset import IntelImageDataset
        from data_processing import create_augmentation_transforms
        
        # Transformações
        train_transforms = create_augmentation_transforms(use_randaugment=True, model_type='cnn')
        val_transforms = create_augmentation_transforms(use_randaugment=False, model_type='cnn')
        
        # Datasets
        train_dataset = IntelImageDataset(df_train, transform=train_transforms)
        test_dataset = IntelImageDataset(df_test, transform=val_transforms)
        
        # Executar destilação
        logger.info(f"Iniciando destilação de conhecimento com {num_epochs} épocas")
        distilled_model, history = train_naturelight_with_distillation(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            device=device,
            num_epochs=num_epochs,
            save_path="models/naturelight_distilled.pth"
        )
        
        logger.info(f"Destilação concluída com acurácia final: {history['best_accuracy']:.2f}%")
        return distilled_model, history
    
    except ImportError:
        logger.error("Módulos necessários não encontrados. Verifique a instalação dos componentes.")
        return None, None
    except Exception as e:
        logger.error(f"Erro durante destilação de conhecimento: {str(e)}")
        return None, None

def optimize_model_hierarchically(df_train, df_test, model_name='resnet50'):
    """
    Aplica otimização hierárquica de hiperparâmetros para o modelo especificado.
    
    Args:
        df_train: DataFrame com dados de treinamento
        df_test: DataFrame com dados de teste
        model_name: Nome do modelo a otimizar
        
    Returns:
        dict: Melhores hiperparâmetros encontrados
    """
    try:
        # Importar funções de otimização
        from hierarchical_optimization import hierarchical_optimization, run_naturelight_optimization
        
        # Criar datasets para otimização
        from dataset import IntelImageDataset
        from data_processing import create_augmentation_transforms
        
        # Transformações
        train_transforms = create_augmentation_transforms(use_randaugment=True)
        val_transforms = create_augmentation_transforms(use_randaugment=False)
        
        # Datasets
        train_dataset = IntelImageDataset(df_train, transform=train_transforms)
        test_dataset = IntelImageDataset(df_test, transform=val_transforms)
        
        # Criar subset para otimização mais rápida
        optim_size = min(2000, len(train_dataset))
        val_size = min(500, len(test_dataset))
        
        # Índices aleatórios
        train_indices = torch.randperm(len(train_dataset))[:optim_size].tolist()
        test_indices = torch.randperm(len(test_dataset))[:val_size].tolist()
        
        # Criar subsets
        optim_train_subset = Subset(train_dataset, train_indices)
        optim_test_subset = Subset(test_dataset, test_indices)
        
        # Executar otimização
        logger.info(f"Iniciando otimização hierárquica para {model_name}")
        if model_name.lower() == 'naturelight':
            best_params = run_naturelight_optimization(optim_train_subset, optim_test_subset)
        else:
            best_params = hierarchical_optimization(
                optim_train_subset, 
                optim_test_subset, 
                model_name=model_name
            )
        
        logger.info(f"Otimização hierárquica concluída para {model_name}")
        return best_params
    
    except ImportError:
        logger.error("Módulos de otimização não encontrados. Verifique a instalação dos componentes.")
        return None
    except Exception as e:
        logger.error(f"Erro durante otimização hierárquica: {str(e)}")
        return None

def apply_smart_sampling(df_train, df_test, sampling_ratio=0.5, device=None):
    """
    Aplica técnicas de amostragem inteligente para reduzir o dataset de treinamento.
    
    Args:
        df_train: DataFrame com dados de treinamento
        df_test: DataFrame com dados de teste
        sampling_ratio: Fração do dataset original a ser mantida
        device: Dispositivo para processamento (CPU/GPU)
        
    Returns:
        tuple: (índices selecionados, resultados da comparação)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Importar funções de amostragem
        from smart_sampling import select_diverse_coreset, select_coreset_kmeans
        
        # Criar datasets
        from dataset import IntelImageDataset
        from data_processing import create_augmentation_transforms
        
        # Transformações
        train_transforms = create_augmentation_transforms(use_randaugment=True)
        val_transforms = create_augmentation_transforms(use_randaugment=False)
        
        # Datasets
        train_dataset = IntelImageDataset(df_train, transform=train_transforms)
        test_dataset = IntelImageDataset(df_test, transform=val_transforms)
        
        # Calcular número de amostras
        num_samples = int(len(train_dataset) * sampling_ratio)
        logger.info(f"Selecionando {num_samples} amostras ({sampling_ratio*100:.1f}%) com amostragem inteligente")
        
        # Extrator de características (MobileNet)
        import torchvision.models as models
        feature_extractor = models.mobilenet_v3_small(weights='DEFAULT')
        feature_extractor = feature_extractor.to(device)
        
        # Selecionar amostras com coreset
        coreset_indices = select_diverse_coreset(
            train_dataset, 
            num_samples=num_samples,
            feature_extractor=feature_extractor,
            device=device
        )
        
        # Selecionar amostras com k-means
        kmeans_indices = select_coreset_kmeans(
            train_dataset,
            num_samples=num_samples,
            feature_extractor=feature_extractor,
            device=device
        )
        
        # Selecionar amostras aleatórias para comparação
        random_indices = torch.randperm(len(train_dataset))[:num_samples].tolist()
        
        # Comparar os métodos
        comparison_results = {
            'coreset': {
                'indices': coreset_indices,
                'method': 'Coreset (diversidade)',
                'ratio': sampling_ratio
            },
            'kmeans': {
                'indices': kmeans_indices,
                'method': 'K-Means Clustering',
                'ratio': sampling_ratio
            },
            'random': {
                'indices': random_indices,
                'method': 'Amostragem Aleatória',
                'ratio': sampling_ratio
            }
        }
        
        # Retornar os índices do coreset (melhor método) e resultados completos
        return coreset_indices, comparison_results
    
    except ImportError:
        logger.error("Módulos de amostragem não encontrados. Verifique a instalação dos componentes.")
        return None, None
    except Exception as e:
        logger.error(f"Erro durante amostragem inteligente: {str(e)}")
        return None, None

def enhance_landscape_classification(df_train, df_test, pipeline_config=None):
    """
    Aplica melhorias ao pipeline de classificação de paisagens.
    
    Args:
        df_train: DataFrame com metadados do dataset de treinamento
        df_test: DataFrame com metadados do dataset de teste
        pipeline_config: Configurações personalizadas
        
    Returns:
        dict: Resultados e métricas das melhorias
    """
    # Configurações padrão
    if pipeline_config is None:
        pipeline_config = {
            'detect_duplicates': True,
            'threshold': 3,  # Limiar para detecção de duplicatas
            'use_naturelight': True,
            'apply_distillation': True,
            'distillation_epochs': 5,
            'hierarchical_optim': False,
            'smart_sampling': True,
            'sampling_ratio': 0.5  # Usar 50% dos dados
        }
    
    logger.info("Iniciando pipeline de melhorias para classificação de paisagens")
    logger.info(f"Configurações: {pipeline_config}")
    
    # Verificar disponibilidade de GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")
    
    # Criar diretório para resultados
    os.makedirs("results/enhancements", exist_ok=True)
    
    # Resultados a retornar
    results = {
        'duplicates': None,
        'naturelight': None,
        'optimization': None,
        'sampling': None
    }
    
    # === 1. DETECÇÃO DE IMAGENS DUPLICADAS ===
    if pipeline_config['detect_duplicates']:
        logger.info("=== DETECÇÃO DE IMAGENS DUPLICADAS ===")
        
        # Executar detecção de duplicatas
        threshold = pipeline_config['threshold']
        duplicates_df = detect_duplicate_images(df_train, hash_diff_threshold=threshold)
        results['duplicates'] = duplicates_df
        
        if duplicates_df is not None and len(duplicates_df) > 0:
            # Visualizar duplicatas
            visualize_duplicates(
                duplicates_df, 
                num_groups=3, 
                output_dir="results/enhancements/duplicates"
            )
            
            # Filtrar dataset para remover duplicatas
            df_train_filtered = filter_duplicates(df_train, duplicates_df, strategy='best_quality')
            
            # Atualizar DataFrame de treinamento
            df_train = df_train_filtered
            
            # Salvar relatório de duplicatas
            duplicates_df.to_csv("results/enhancements/duplicate_images.csv", index=False)
            logger.info("Relatório de duplicatas salvo em results/enhancements/duplicate_images.csv")
    
    # === 2. MODELO NATURELIGHT COM DESTILAÇÃO ===
    if pipeline_config['use_naturelight']:
        logger.info("=== CRIAÇÃO DO MODELO NATURELIGHT ===")
        
        # Obter número de classes
        num_classes = len(df_train['label'].unique())
        
        # Inicializar modelo
        naturelight_model, model_metrics = initialize_naturelight_model(num_classes=num_classes, device=device)
        results['naturelight'] = model_metrics
        
        # Aplicar destilação de conhecimento
        if pipeline_config['apply_distillation'] and naturelight_model is not None:
            logger.info("=== DESTILAÇÃO DE CONHECIMENTO ===")
            
            distilled_model, distillation_history = apply_knowledge_distillation(
                df_train, 
                df_test, 
                num_classes=num_classes,
                device=device,
                num_epochs=pipeline_config.get('distillation_epochs', 5)
            )
            
            if distilled_model is not None:
                # Atualizar resultados
                results['naturelight']['distilled_model'] = True
                results['naturelight']['distillation_history'] = {
                    'best_accuracy': float(distillation_history['best_accuracy']),
                    'epochs': len(distillation_history['train_losses']),
                    'final_val_acc': float(distillation_history['val_accuracies'][-1])
                }
                
                # Atualizar modelo
                naturelight_model = distilled_model
    
    # === 3. OTIMIZAÇÃO HIERÁRQUICA DE HIPERPARÂMETROS ===
    if pipeline_config['hierarchical_optim']:
        logger.info("=== OTIMIZAÇÃO HIERÁRQUICA DE HIPERPARÂMETROS ===")
        
        # Determinar modelo a otimizar
        model_to_optimize = 'naturelight' if pipeline_config['use_naturelight'] else 'resnet50'
        
        # Executar otimização
        best_params = optimize_model_hierarchically(
            df_train, 
            df_test, 
            model_name=model_to_optimize
        )
        
        if best_params is not None:
            results['optimization'] = {
                'model': model_to_optimize,
                'best_params': best_params
            }
    
    # === 4. AMOSTRAGEM INTELIGENTE ===
    if pipeline_config['smart_sampling']:
        logger.info("=== AMOSTRAGEM INTELIGENTE ===")
        
        # Executar amostragem
        sampling_ratio = pipeline_config.get('sampling_ratio', 0.5)
        selected_indices, sampling_results = apply_smart_sampling(
            df_train, 
            df_test, 
            sampling_ratio=sampling_ratio,
            device=device
        )
        
        if selected_indices is not None:
            results['sampling'] = {
                'selected_indices': selected_indices,
                'results': sampling_results,
                'num_samples': len(selected_indices),
                'ratio': sampling_ratio,
                'reduction_percent': (1 - sampling_ratio) * 100
            }
    
    # === 5. RELATÓRIO FINAL ===
    logger.info("=== GERANDO RELATÓRIO FINAL ===")
    
    try:
        # Gerar resumo das melhorias
        summary = {
            'duplicate_images': len(results['duplicates']) if results['duplicates'] is not None else 0,
            'naturelight_reduction': results['naturelight']['reduction_percent'] if results['naturelight'] is not None else 0,
            'sampling_reduction': results['sampling']['reduction_percent'] if 'sampling' in results and results['sampling'] is not None else 0
        }
        
        # Salvar relatório
        with open("results/enhancements/enhancement_summary.txt", "w") as f:
            f.write("===== RELATÓRIO DE MELHORIAS =====\n\n")
            f.write(f"1. DETECÇÃO DE IMAGENS DUPLICADAS\n")
            f.write(f"   - Imagens similares detectadas: {summary['duplicate_images']}\n\n")
            
            f.write(f"2. MODELO LEVE NATURELIGHT\n")
            if 'naturelight' in results and results['naturelight'] is not None:
                f.write(f"   - Parâmetros: {results['naturelight'].get('params', 0):,}\n")
                f.write(f"   - Redução em relação ao MobileNet: {results['naturelight'].get('reduction_percent', 0):.1f}%\n")
                if 'distillation_history' in results['naturelight']:
                    f.write(f"   - Acurácia após destilação: {results['naturelight']['distillation_history'].get('best_accuracy', 0):.2f}%\n\n")
            
            f.write(f"3. OTIMIZAÇÃO HIERÁRQUICA\n")
            if 'optimization' in results and results['optimization'] is not None:
                f.write(f"   - Modelo otimizado: {results['optimization'].get('model', '')}\n")
                f.write(f"   - Parâmetros otimizados: {len(results['optimization'].get('best_params', {}))}\n\n")
            
            f.write(f"4. AMOSTRAGEM INTELIGENTE\n")
            if 'sampling' in results and results['sampling'] is not None:
                f.write(f"   - Amostras selecionadas: {results['sampling'].get('num_samples', 0):,} ({results['sampling'].get('ratio', 0)*100:.1f}%)\n")
                f.write(f"   - Redução do conjunto de treinamento: {results['sampling'].get('reduction_percent', 0):.1f}%\n\n")
            
            f.write(f"===== RESUMO FINAL =====\n")
            f.write(f"Redução de parâmetros: {summary.get('naturelight_reduction', 0):.1f}%\n")
            f.write(f"Redução de dados: {summary.get('sampling_reduction', 0):.1f}%\n")
            
            # Calcular ganho de eficiência aproximado
            param_reduction = summary.get('naturelight_reduction', 0) / 100
            data_reduction = summary.get('sampling_reduction', 0) / 100
            
            efficiency_gain = (1 - (1 - param_reduction) * (1 - data_reduction)) * 100
            f.write(f"Ganho de eficiência estimado: {efficiency_gain:.1f}%\n")
        
        logger.info(f"Relatório final gerado em results/enhancements/enhancement_summary.txt")
        
        # Gerar visualização das melhorias
        plt.figure(figsize=(12, 8))
        
        # Gráfico de barras para redução de parâmetros
        plt.subplot(2, 2, 1)
        if 'naturelight' in results and results['naturelight'] is not None:
            models = ["MobileNet", "NatureLightNet"]
            model_params = [1.0, 1.0 - results['naturelight'].get('reduction_percent', 0)/100]
            plt.bar(models, model_params, color=['skyblue', 'green'])
            plt.title('Redução de Parâmetros')
            plt.ylabel('Proporção de Parâmetros')
            plt.ylim(0, 1.1)
        
        # Gráfico de barras para redução de dados
        plt.subplot(2, 2, 2)
        if 'sampling' in results and results['sampling'] is not None:
            sampling_methods = ["Dataset Original", "Amostragem Inteligente"]
            data_sizes = [1.0, results['sampling'].get('ratio', 1.0)]
            plt.bar(sampling_methods, data_sizes, color=['skyblue', 'orange'])
            plt.title('Redução do Dataset')
            plt.ylabel('Proporção de Dados')
            plt.ylim(0, 1.1)
        
        # Gráfico de pizza para duplicatas
        plt.subplot(2, 2, 3)
        if 'duplicates' in results and results['duplicates'] is not None and len(results['duplicates']) > 0:
            duplicates_count = len(results['duplicates'].drop_duplicates(subset='hash'))
            total_images = len(df_train) + duplicates_count  # Aproximação do total original
            unique_images = total_images - duplicates_count
            plt.pie([unique_images, duplicates_count], 
                   labels=['Imagens Únicas', 'Grupos Duplicados'],
                   autopct='%1.1f%%',
                   colors=['lightgreen', 'tomato'])
            plt.title(f'Detecção de Duplicatas\n({duplicates_count} grupos)')
        
        # Gráfico de linha para ganho de eficiência
        plt.subplot(2, 2, 4)
        # Simular curva de eficiência para diferentes níveis de redução
        reduction_levels = np.linspace(0, 0.9, 10)  # 0% a 90% de redução
        efficiency_levels = [(1 - (1-x)*(1-data_reduction))*100 for x in reduction_levels]
        
        plt.plot(reduction_levels*100, efficiency_levels, 'b-', marker='o')
        plt.axvline(x=param_reduction*100, color='r', linestyle='--')
        plt.text(param_reduction*100 + 2, 50, f'{param_reduction*100:.1f}%', 
                color='r', fontweight='bold')
        plt.title('Ganho de Eficiência vs Redução de Parâmetros')
        plt.xlabel('Redução de Parâmetros (%)')
        plt.ylabel('Ganho de Eficiência (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("results/enhancements/enhancement_summary.png")
        logger.info("Visualização de melhorias salva em results/enhancements/enhancement_summary.png")
        plt.close()
        
        return results
    
    except Exception as e:
        logger.error(f"Erro ao gerar relatório final: {str(e)}")
        return results