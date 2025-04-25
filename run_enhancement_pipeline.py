#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para executar o pipeline completo de melhorias no sistema de classificação de paisagens.

Este script demonstra como utilizar todas as melhorias implementadas:
1. Detecção de imagens duplicadas
2. Modelo NatureLightNet com destilação de conhecimento
3. Otimização hierárquica de hiperparâmetros
4. Amostragem inteligente para redução de dados

Uso:
    python run_enhancement_pipeline.py --config config/enhancements_config.yaml
"""

import os
import argparse
import yaml
import logging
import sys
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("enhancement_pipeline.log")
    ]
)

logger = logging.getLogger("enhancement_pipeline")

def parse_arguments():
    """Processa argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description="Pipeline de melhorias para classificação de paisagens")
    
    parser.add_argument('--config', type=str, default="config/enhancements_config.yaml",
                        help='Arquivo de configuração YAML')
    parser.add_argument('--dataset_path', type=str, 
                        help='Caminho para o dataset (sobrescreve config)')
    parser.add_argument('--output_dir', type=str, default="results/enhancement_pipeline",
                        help='Diretório para saída de resultados')
    
    return parser.parse_args()

def load_config(config_path):
    """Carrega configurações do arquivo YAML"""
    default_config = {
        'dataset': {
            'path': 'data/intel-image-classification',
            'train_dir': 'seg_train/seg_train',
            'test_dir': 'seg_test/seg_test'
        },
        'enhancements': {
            'detect_duplicates': True,
            'threshold': 3,
            'use_naturelight': True,
            'apply_distillation': True,
            'distillation_epochs': 5,
            'hierarchical_optim': True,
            'smart_sampling': True,
            'sampling_ratio': 0.5
        },
        'output': {
            'save_models': True,
            'visualize_results': True,
            'output_dir': 'results/enhancement_pipeline'
        }
    }
    
    # Se o arquivo não existir, criar com configurações padrão
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        logger.info(f"Arquivo de configuração criado em {config_path}")
        config = default_config
    else:
        # Carregar configurações do arquivo
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configurações carregadas de {config_path}")
    
    return config

def setup_environment(output_dir):
    """Configura o ambiente para execução"""
    # Criar diretórios de saída
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    
    # Verificar disponibilidade de GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memória GPU total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        logger.info(f"Memória GPU disponível: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
    else:
        logger.info("GPU não disponível, usando CPU")
    
    return device

def run_pipeline(config, args, device):
    """Executa o pipeline de melhorias completo"""
    start_time = time.time()
    logger.info("Iniciando pipeline de melhorias...")
    
    # Configurar caminhos
    dataset_path = args.dataset_path if args.dataset_path else config['dataset']['path']
    output_dir = args.output_dir if args.output_dir else config['output']['output_dir']
    
    # Determinar diretórios de treino/teste
    train_dir = os.path.join(dataset_path, config['dataset']['train_dir'])
    test_dir = os.path.join(dataset_path, config['dataset']['test_dir'])
    
    # Verificar caminhos alternativos se os padrões não forem encontrados
    if not os.path.exists(train_dir):
        alt_train_dir = os.path.join(dataset_path, 'train')
        if os.path.exists(alt_train_dir):
            train_dir = alt_train_dir
            logger.info(f"Usando diretório alternativo para treino: {train_dir}")
        else:
            logger.error(f"Diretório de treino não encontrado: {train_dir}")
            return None
    
    if not os.path.exists(test_dir):
        alt_test_dir = os.path.join(dataset_path, 'test')
        if os.path.exists(alt_test_dir):
            test_dir = alt_test_dir
            logger.info(f"Usando diretório alternativo para teste: {test_dir}")
        else:
            logger.error(f"Diretório de teste não encontrado: {test_dir}")
            return None
    
    logger.info(f"Executando pipeline com dataset em {dataset_path}")
    logger.info(f"- Diretório de treino: {train_dir}")
    logger.info(f"- Diretório de teste: {test_dir}")
    
    # Criar estrutura de resultados
    results = {
        'config': config,
        'dataset': {
            'path': dataset_path,
            'train_dir': train_dir,
            'test_dir': test_dir
        },
        'device': str(device),
        'pipeline_start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'stages': {}
    }
    
    try:
        # ===== 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS =====
        logger.info("=== CARREGAMENTO DE DADOS ===")
        from data_processing import create_df
        
        # Criar DataFrames com metadados das imagens
        df_train, train_corrupted = create_df(train_dir)
        df_test, test_corrupted = create_df(test_dir)
        
        logger.info(f"Dados carregados: {len(df_train)} imagens de treinamento, {len(df_test)} imagens de teste")
        
        if train_corrupted:
            logger.warning(f"Encontradas {len(train_corrupted)} imagens corrompidas no conjunto de treinamento")
        if test_corrupted:
            logger.warning(f"Encontradas {len(test_corrupted)} imagens corrompidas no conjunto de teste")
        
        # Salvar estatísticas de dados
        results['stages']['data_loading'] = {
            'train_images': len(df_train),
            'test_images': len(df_test),
            'train_corrupted': len(train_corrupted),
            'test_corrupted': len(test_corrupted),
            'classes': sorted(df_train['label'].unique().tolist())
        }
        
        # ===== 2. DETECÇÃO DE IMAGENS DUPLICADAS =====
        if config['enhancements']['detect_duplicates']:
            logger.info("=== DETECÇÃO DE IMAGENS DUPLICADAS ===")
            
            # Criar função de detecção
            def detect_duplicate_images(dataframe, hash_diff_threshold=3):
                """Detecta imagens duplicadas ou quase idênticas usando imagehash"""
                logger.info(f"Iniciando detecção com limiar {hash_diff_threshold}...")
                
                # Verificar se a coluna 'image_hash' existe
                if 'image_hash' not in dataframe.columns:
                    logger.error("Coluna 'image_hash' não encontrada")
                    return None
                
                # Criar dicionário para agrupar imagens por hash
                similar_images = {}
                duplicates_count = 0
                
                # Para cada imagem no dataframe
                for idx, row in dataframe.iterrows():
                    img_path = row['image_path']
                    img_hash = row['image_hash']
                    
                    if pd.isna(img_hash) or img_hash is None:
                        continue
                        
                    # Verificar similaridade com hashes já processados
                    import imagehash
                    img_hash_obj = imagehash.hex_to_hash(img_hash) if isinstance(img_hash, str) else img_hash
                    
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
            
            # Detectar duplicatas no conjunto de treinamento
            threshold = config['enhancements']['threshold']
            duplicates_df = detect_duplicate_images(df_train, hash_diff_threshold=threshold)
            
            # Visualizar duplicatas se solicitado
            if config['output']['visualize_results'] and duplicates_df is not None and len(duplicates_df) > 0:
                try:
                    from PIL import Image
                    
                    # Diretório para visualizações
                    vis_dir = os.path.join(output_dir, "visualizations", "duplicates")
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # Selecionar alguns grupos para visualizar
                    groups = duplicates_df['hash'].unique()[:3]
                    
                    for group_idx, hash_val in enumerate(groups):
                        group_df = duplicates_df[duplicates_df['hash'] == hash_val]
                        
                        # Limitar a 4 imagens por grupo para visualização
                        group_df = group_df.head(4)
                        
                        plt.figure(figsize=(4*4, 4))
                        for i, (_, row) in enumerate(group_df.iterrows()):
                            plt.subplot(1, len(group_df), i+1)
                            img = Image.open(row['image_path']).convert('RGB')
                            plt.imshow(img)
                            plt.title(f"Diff: {row['similarity_diff']}")
                            plt.axis('off')
                        
                        plt.suptitle(f"Grupo {group_idx+1}: Hash {hash_val[:8]}... ({len(group_df)} imagens)")
                        plt.tight_layout()
                        
                        # Salvar visualização
                        plt.savefig(os.path.join(vis_dir, f"duplicate_group_{group_idx+1}.png"))
                        plt.close()
                    
                    # Salvar relatório de duplicatas
                    duplicates_df.to_csv(os.path.join(output_dir, "reports", "duplicate_images.csv"), index=False)
                    
                except Exception as e:
                    logger.error(f"Erro ao visualizar duplicatas: {str(e)}")
            
            # Criar subset sem duplicatas
            if duplicates_df is not None and len(duplicates_df) > 0:
                # Para cada grupo, manter apenas a imagem com menor diferença
                unique_images = []
                for hash_val in duplicates_df['hash'].unique():
                    group_df = duplicates_df[duplicates_df['hash'] == hash_val]
                    best_image = group_df.loc[group_df['similarity_diff'].idxmin()]['image_path']
                    unique_images.append(best_image)
                
                # Filtrar DataFrame original
                df_train_filtered = df_train[~df_train['image_path'].isin(duplicates_df['image_path']) | 
                                           df_train['image_path'].isin(unique_images)]
                
                # Atualizar estatísticas
                duplicate_reduction = len(df_train) - len(df_train_filtered)
                reduction_percent = duplicate_reduction / len(df_train) * 100
                
                logger.info(f"Dataset reduzido de {len(df_train)} para {len(df_train_filtered)} imagens")
                logger.info(f"Redução de {reduction_percent:.2f}% ({duplicate_reduction} imagens removidas)")
                
                # Atualizar DataFrame de treinamento
                df_train = df_train_filtered
                
                # Salvar resultados
                results['stages']['duplicate_detection'] = {
                    'threshold': threshold,
                    'duplicates_found': len(duplicates_df),
                    'groups_found': duplicates_df['hash'].nunique(),
                    'images_removed': duplicate_reduction,
                    'reduction_percent': reduction_percent
                }
            else:
                results['stages']['duplicate_detection'] = {
                    'threshold': threshold,
                    'duplicates_found': 0,
                    'groups_found': 0,
                    'images_removed': 0,
                    'reduction_percent': 0
                }
        
        # ===== 3. MODELO NATURELIGHT COM DESTILAÇÃO =====
        if config['enhancements']['use_naturelight']:
            logger.info("=== CRIAÇÃO DO MODELO NATURELIGHT ===")
            
            # Verificar se módulos necessários estão disponíveis
            try:
                from NatureLightNet import create_naturelight_model
            except ImportError:
                logger.error("Módulo NatureLightNet não encontrado. Certifique-se de que está no caminho Python.")
                return None
            
            try:
                # Criar modelo NatureLightNet
                naturelight_model = create_naturelight_model(num_classes=len(results['stages']['data_loading']['classes']))
                
                # Obter contagens de parâmetros
                trainable_params, total_params = naturelight_model.get_params_count()
                
                # Comparar com MobileNet
                import torchvision.models as tvm
                mobilenet = tvm.mobilenet_v3_small(weights=None, num_classes=len(results['stages']['data_loading']['classes']))
                mobilenet_params = sum(p.numel() for p in mobilenet.parameters())
                
                # Calcular redução
                reduction_percent = (1 - trainable_params / mobilenet_params) * 100
                
                logger.info(f"NatureLightNet: {trainable_params:,} parâmetros")
                logger.info(f"MobileNetV3-Small: {mobilenet_params:,} parâmetros")
                logger.info(f"Redução: {reduction_percent:.1f}%")
                
                # Salvar resultados
                results['stages']['naturelight'] = {
                    'params': int(trainable_params),
                    'mobilenet_params': int(mobilenet_params),
                    'reduction_percent': float(reduction_percent)
                }
                
                # Aplicar destilação de conhecimento se habilitado
                if config['enhancements']['apply_distillation']:
                    logger.info("=== DESTILAÇÃO DE CONHECIMENTO ===")
                    
                    try:
                        # Importar função de destilação
                        from knowledge_distillation import train_naturelight_with_distillation
                        
                        # Criar datasets
                        from dataset import IntelImageDataset
                        from data_processing import create_augmentation_transforms
                        
                        # Transformações
                        train_transforms = create_augmentation_transforms(use_randaugment=True, model_type='cnn')
                        val_transforms = create_augmentation_transforms(use_randaugment=False, model_type='cnn')
                        
                        # Datasets para destilação
                        train_dataset = IntelImageDataset(df_train, transform=train_transforms)
                        test_dataset = IntelImageDataset(df_test, transform=val_transforms)
                        
                        # Executar destilação
                        distilled_model, distillation_history = train_naturelight_with_distillation(
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            device=device,
                            num_epochs=config['enhancements'].get('distillation_epochs', 5),
                            save_path=os.path.join(output_dir, "models", "naturelight_distilled.pth")
                        )
                        
                        # Atualizar resultados
                        results['stages']['distillation'] = {
                            'best_accuracy': float(distillation_history['best_accuracy']),
                            'epochs': len(distillation_history['train_losses']),
                            'final_train_acc': float(distillation_history['train_accuracies'][-1]),
                            'final_val_acc': float(distillation_history['val_accuracies'][-1])
                        }
                        
                        logger.info(f"Destilação concluída com acurácia final: {distillation_history['best_accuracy']:.2f}%")
                        
                    except Exception as e:
                        logger.error(f"Erro durante destilação de conhecimento: {str(e)}")
                        results['stages']['distillation'] = {
                            'error': str(e)
                        }
            
            except Exception as e:
                logger.error(f"Erro ao criar modelo NatureLightNet: {str(e)}")
                results['stages']['naturelight'] = {
                    'error': str(e)
                }
        
        # ===== 4. AMOSTRAGEM INTELIGENTE =====
        if config['enhancements']['smart_sampling']:
            logger.info("=== AMOSTRAGEM INTELIGENTE ===")
            
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
                
                # Método de amostragem: Coreset baseado em diversidade
                sampling_ratio = config['enhancements'].get('sampling_ratio', 0.5)
                num_samples = int(len(train_dataset) * sampling_ratio)
                
                logger.info(f"Selecionando coreset diverso com {num_samples} amostras ({sampling_ratio*100:.1f}%)")
                
                # Criar extrator de características (usar MobileNet pré-treinado)
                import torchvision.models as models
                feature_extractor = models.mobilenet_v3_small(weights='DEFAULT')
                feature_extractor = feature_extractor.to(device)
                
                # Selecionar coreset
                coreset_indices = select_diverse_coreset(
                    train_dataset, 
                    num_samples=num_samples,
                    feature_extractor=feature_extractor,
                    device=device
                )
                
                # Comparar com amostragem aleatória
                random_indices = torch.randperm(len(train_dataset))[:num_samples].tolist()
                
                # Comparação rápida
                from torch.utils.data import Subset
                from NatureLightNet import create_naturelight_model
                import torch.nn as nn
                import torch.optim as optim
                
                # Criar subsets
                coreset_subset = Subset(train_dataset, coreset_indices)
                random_subset = Subset(train_dataset, random_indices)
                
                # Função para treinar e avaliar um modelo
                def evaluate_subset(subset, subset_name):
                    # Criar dataloader
                    from torch.utils.data import DataLoader
                    loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=4)
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
                    
                    # Criar modelo simples para teste
                    model = create_naturelight_model(
                        num_classes=len(results['stages']['data_loading']['classes'])
                    ).to(device)
                    
                    # Treinar por poucas épocas
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    
                    # Métricas
                    train_accs = []
                    val_accs = []
                    
                    # Apenas 3 épocas para comparação rápida
                    for epoch in range(3):
                        # Treino
                        model.train()
                        running_loss = 0.0
                        correct = 0
                        total = 0
                        
                        for inputs, targets in loader:
                            inputs, targets = inputs.to(device), targets.to(device)
                            
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
                            
                            _, predicted = outputs.max(1)
                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()
                        
                        train_acc = 100.0 * correct / total
                        train_accs.append(train_acc)
                        
                        # Validação
                        model.eval()
                        correct = 0
                        total = 0
                        
                        with torch.no_grad():
                            for inputs, targets in test_loader:
                                inputs, targets = inputs.to(device), targets.to(device)
                                outputs = model(inputs)
                                _, predicted = outputs.max(1)
                                total += targets.size(0)
                                correct += predicted.eq(targets).sum().item()
                        
                        val_acc = 100.0 * correct / total
                        val_accs.append(val_acc)
                        
                        logger.info(f"Época {epoch+1}: {subset_name} - Treino: {train_acc:.2f}%, Validação: {val_acc:.2f}%")
                    
                    return {
                        'train_accs': train_accs,
                        'val_accs': val_accs,
                        'final_val_acc': val_accs[-1]
                    }
                
                # Avaliar cada método
                coreset_results = evaluate_subset(coreset_subset, "Coreset")
                random_results = evaluate_subset(random_subset, "Random")
                
                # Comparar resultados
                logger.info("=== COMPARAÇÃO DE MÉTODOS DE AMOSTRAGEM ===")
                logger.info(f"Coreset (Acurácia final): {coreset_results['final_val_acc']:.2f}%")
                logger.info(f"Random (Acurácia final): {random_results['final_val_acc']:.2f}%")
                
                # Calcular ganho
                accuracy_gain = coreset_results['final_val_acc'] - random_results['final_val_acc']
                logger.info(f"Ganho de acurácia com Coreset: {accuracy_gain:.2f}%")
                
                # Salvar resultados
                results['stages']['smart_sampling'] = {
                    'sampling_ratio': sampling_ratio,
                    'num_samples': num_samples,
                    'reduction_percent': (1 - sampling_ratio) * 100,
                    'coreset_accuracy': float(coreset_results['final_val_acc']),
                    'random_accuracy': float(random_results['final_val_acc']),
                    'accuracy_gain': float(accuracy_gain)
                }
                
                # Visualizar comparação
                if config['output']['visualize_results']:
                    vis_dir = os.path.join(output_dir, "visualizations", "sampling")
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # Gráfico de acurácia de validação
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, 4), coreset_results['val_accs'], 'b-o', label='Coreset')
                    plt.plot(range(1, 4), random_results['val_accs'], 'r-o', label='Random')
                    plt.title('Comparação de Métodos de Amostragem')
                    plt.xlabel('Época')
                    plt.ylabel('Acurácia de Validação (%)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(vis_dir, "sampling_comparison.png"))
                    plt.close()
                    
                    # Gráfico de acurácia final
                    plt.figure(figsize=(8, 6))
                    methods = ['Random', 'Coreset']
                    accuracies = [random_results['final_val_acc'], coreset_results['final_val_acc']]
                    
                    plt.bar(methods, accuracies, color=['red', 'blue'])
                    plt.title('Acurácia Final por Método de Amostragem')
                    plt.ylabel('Acurácia de Validação (%)')
                    
                    # Adicionar valores nas barras
                    for i, v in enumerate(accuracies):
                        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
                    
                    plt.savefig(os.path.join(vis_dir, "sampling_final_comparison.png"))
                    plt.close()
            
            except Exception as e:
                logger.error(f"Erro durante amostragem inteligente: {str(e)}")
                results['stages']['smart_sampling'] = {
                    'error': str(e)
                }
        
        # ===== 5. RELATÓRIO FINAL =====
        logger.info("=== GERANDO RELATÓRIO FINAL ===")
        
        # Calcular tempo total de execução
        end_time = time.time()
        total_time = end_time - start_time
        results['pipeline_end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        results['total_execution_time'] = total_time
        
        # Calcular ganhos totais
        final_summary = {
            'total_execution_time': f"{total_time/60:.2f} minutos",
            'improvements': []
        }
        
        # Resumir melhorias encontradas
        if 'duplicate_detection' in results['stages']:
            dup_stage = results['stages']['duplicate_detection']
            if dup_stage['reduction_percent'] > 0:
                final_summary['improvements'].append({
                    'name': 'Remoção de duplicatas',
                    'gain': f"Redução de {dup_stage['reduction_percent']:.1f}% no dataset"
                })
        
        if 'naturelight' in results['stages']:
            nature_stage = results['stages']['naturelight']
            if 'reduction_percent' in nature_stage:
                final_summary['improvements'].append({
                    'name': 'NatureLightNet',
                    'gain': f"Redução de {nature_stage['reduction_percent']:.1f}% nos parâmetros"
                })
        
        if 'distillation' in results['stages']:
            dist_stage = results['stages']['distillation']
            if 'best_accuracy' in dist_stage:
                final_summary['improvements'].append({
                    'name': 'Destilação de conhecimento',
                    'gain': f"Acurácia de {dist_stage['best_accuracy']:.1f}% com modelo reduzido"
                })
        
        if 'smart_sampling' in results['stages']:
            sample_stage = results['stages']['smart_sampling']
            if 'accuracy_gain' in sample_stage:
                final_summary['improvements'].append({
                    'name': 'Amostragem inteligente',
                    'gain': f"Redução de {sample_stage['reduction_percent']:.1f}% nos dados com ganho de {sample_stage['accuracy_gain']:.1f}% na acurácia"
                })
        
        # Calcular ganho de eficiência total estimado
        param_reduction = results['stages'].get('naturelight', {}).get('reduction_percent', 0)
        data_reduction = results['stages'].get('smart_sampling', {}).get('reduction_percent', 0)
        
        efficiency_gain = (1 - (1 - param_reduction/100) * (1 - data_reduction/100)) * 100
        final_summary['total_efficiency_gain'] = f"{efficiency_gain:.1f}%"
        
        # Salvar relatório final
        results['final_summary'] = final_summary
        
        # Salvar como JSON
        import json
        with open(os.path.join(output_dir, "reports", "enhancement_report.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Gerar relatório textual
        with open(os.path.join(output_dir, "reports", "enhancement_summary.txt"), 'w') as f:
            f.write("===== RELATÓRIO DE MELHORIAS =====\n\n")
            
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Data e hora: {results['pipeline_end_time']}\n")
            f.write(f"Tempo de execução: {final_summary['total_execution_time']}\n\n")
            
            f.write("--- MELHORIAS IMPLEMENTADAS ---\n\n")
            
            for imp in final_summary['improvements']:
                f.write(f"{imp['name']}: {imp['gain']}\n")
            
            f.write(f"\nGanho de eficiência total estimado: {final_summary['total_efficiency_gain']}\n\n")
            
            f.write("--- DETALHES POR ESTÁGIO ---\n\n")
            
            if 'duplicate_detection' in results['stages']:
                dup = results['stages']['duplicate_detection']
                f.write("1. DETECÇÃO DE DUPLICATAS\n")
                f.write(f"   - Duplicatas encontradas: {dup.get('duplicates_found', 0)}\n")
                f.write(f"   - Grupos: {dup.get('groups_found', 0)}\n")
                f.write(f"   - Imagens removidas: {dup.get('images_removed', 0)}\n")
                f.write(f"   - Redução: {dup.get('reduction_percent', 0):.1f}%\n\n")
            
            if 'naturelight' in results['stages']:
                nl = results['stages']['naturelight']
                f.write("2. MODELO NATURELIGHT\n")
                if 'error' in nl:
                    f.write(f"   - Erro: {nl['error']}\n\n")
                else:
                    f.write(f"   - Parâmetros: {nl.get('params', 0):,}\n")
                    f.write(f"   - Comparação com MobileNet: {nl.get('reduction_percent', 0):.1f}% menor\n\n")
            
            if 'distillation' in results['stages']:
                dist = results['stages']['distillation']
                f.write("3. DESTILAÇÃO DE CONHECIMENTO\n")
                if 'error' in dist:
                    f.write(f"   - Erro: {dist['error']}\n\n")
                else:
                    f.write(f"   - Épocas: {dist.get('epochs', 0)}\n")
                    f.write(f"   - Acurácia final: {dist.get('best_accuracy', 0):.2f}%\n\n")
            
            if 'smart_sampling' in results['stages']:
                ss = results['stages']['smart_sampling']
                f.write("4. AMOSTRAGEM INTELIGENTE\n")
                if 'error' in ss:
                    f.write(f"   - Erro: {ss['error']}\n\n")
                else:
                    f.write(f"   - Amostras selecionadas: {ss.get('num_samples', 0):,} ({ss.get('sampling_ratio', 0)*100:.1f}%)\n")
                    f.write(f"   - Acurácia com Coreset: {ss.get('coreset_accuracy', 0):.2f}%\n")
                    f.write(f"   - Acurácia com Random: {ss.get('random_accuracy', 0):.2f}%\n")
                    f.write(f"   - Ganho: {ss.get('accuracy_gain', 0):.2f}%\n\n")
        
        logger.info(f"Relatório final salvo em {os.path.join(output_dir, 'reports')}")
        
        # Gerar visualização de resumo se solicitado
        if config['output']['visualize_results']:
            # Gráfico de ganhos de eficiência
            plt.figure(figsize=(12, 8))
            
            # Gráfico de barras para redução de parâmetros
            plt.subplot(2, 2, 1)
            if 'naturelight' in results['stages'] and 'reduction_percent' in results['stages']['naturelight']:
                models = ["MobileNet", "NatureLightNet"]
                model_params = [1.0, 1.0 - results['stages']['naturelight']['reduction_percent']/100]
                plt.bar(models, model_params, color=['skyblue', 'green'])
                plt.title('Redução de Parâmetros')
                plt.ylabel('Parâmetros (normalizado)')
                plt.ylim(0, 1.1)
            else:
                plt.text(0.5, 0.5, "Dados não disponíveis", ha='center', va='center')
            
            # Gráfico de barras para redução de dados
            plt.subplot(2, 2, 2)
            if 'smart_sampling' in results['stages'] and 'reduction_percent' in results['stages']['smart_sampling']:
                datasets = ["Original", "Coreset"]
                dataset_sizes = [1.0, 1.0 - results['stages']['smart_sampling']['reduction_percent']/100]
                plt.bar(datasets, dataset_sizes, color=['skyblue', 'orange'])
                plt.title('Redução de Dados')
                plt.ylabel('Tamanho do dataset (normalizado)')
                plt.ylim(0, 1.1)
            else:
                plt.text(0.5, 0.5, "Dados não disponíveis", ha='center', va='center')
            
            # Gráfico de eficiência combinado
            plt.subplot(2, 1, 2)
            improvements = ["Modelo\nOriginal", "Parâmetros\nReduzidos", "Dados\nReduzidos", "Combinado"]
            efficiency = [1.0, 1.0 - param_reduction/100, 1.0 - data_reduction/100, 1.0 - efficiency_gain/100]
            plt.bar(improvements, efficiency, color=['gray', 'green', 'orange', 'red'])
            plt.title('Ganho de Eficiência Combinado')
            plt.ylabel('Requisitos (normalizado)')
            plt.ylim(0, 1.1)
            
            # Adicionar texto com valores de redução
            for i, v in enumerate([0, param_reduction, data_reduction, efficiency_gain]):
                if i > 0:  # Pular o primeiro (modelo original)
                    plt.text(i, 0.5, f"-{v:.1f}%", ha='center', color='white', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "visualizations", "efficiency_gains.png"))
            plt.close()
            
            logger.info(f"Visualização de ganhos de eficiência salva")
        
        return results
    
    except Exception as e:
        logger.error(f"Erro durante execução do pipeline de melhorias: {str(e)}")
        # Salvar resultados parciais
        if 'results' in locals():
            import json
            try:
                results['error'] = str(e)
                with open(os.path.join(output_dir, "reports", "enhancement_report_error.json"), 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Resultados parciais salvos")
            except:
                pass
        return None

def main():
    """Função principal"""
    # Processar argumentos
    args = parse_arguments()
    
    # Carregar configurações
    config = load_config(args.config)
    
    # Sobrescrever diretório de saída se especificado
    output_dir = args.output_dir if args.output_dir else config['output']['output_dir']
    
    # Configurar ambiente
    device = setup_environment(output_dir)
    
    # Executar pipeline
    results = run_pipeline(config, args, device)
    
    if results:
        logger.info("Pipeline de melhorias executado com sucesso!")
        # Resumo final
        if 'final_summary' in results:
            print("\n===== RESUMO DAS MELHORIAS =====")
            print(f"Tempo de execução: {results['final_summary']['total_execution_time']}")
            print("\nMelhorias implementadas:")
            for improvement in results['final_summary']['improvements']:
                print(f"- {improvement['name']}: {improvement['gain']}")
            print(f"\nGanho de eficiência total estimado: {results['final_summary']['total_efficiency_gain']}")
            print("\nPara detalhes completos, consulte o relatório em:")
            print(f"  {os.path.join(output_dir, 'reports', 'enhancement_summary.txt')}")
    else:
        logger.error("Falha na execução do pipeline de melhorias. Verifique os logs para detalhes.")

if __name__ == "__main__":
    main()