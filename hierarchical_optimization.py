"""
Módulo para otimização hierárquica de hiperparâmetros.

Este módulo implementa a otimização de hiperparâmetros em múltiplas fases,
permitindo uma busca mais eficiente no espaço de hiperparâmetros.
"""

import os
import torch
import optuna
import json
import logging
import yaml
import numpy as np
from optuna.trial import TrialState
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import gc
import time
from functools import partial

# Obter o logger
logger = logging.getLogger("landscape_classifier")

def optimize_stage(param_ranges, train_dataset, test_dataset, 
                  model_name='resnet50', n_trials=10, stage_name="base"):
    """
    Executa otimização para uma fase específica com Optuna.
    
    Args:
        param_ranges: Dicionário com ranges de parâmetros a serem otimizados
        train_dataset: Dataset de treinamento
        test_dataset: Dataset de teste
        model_name: Nome do modelo base a otimizar
        n_trials: Número de trials para esta fase
        stage_name: Nome identificador da fase de otimização
        
    Returns:
        dict: Melhores hiperparâmetros encontrados
    """
    
    logger.info(f"Iniciando fase de otimização '{stage_name}' para {model_name}")
    
    # Definir função objetivo para esta fase
    def objective(trial):
        # Limpar memória GPU
        torch.cuda.empty_cache()
        gc.collect()
        
        # Gerar hiperparâmetros de acordo com os ranges definidos
        params = {}
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, list) and len(param_range) == 1:
                # Parâmetro fixo de fase anterior
                params[param_name] = param_range[0]
            elif isinstance(param_range, list):
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, min(param_range), max(param_range))
                elif isinstance(param_range[0], float):
                    if param_name.startswith('lr'):
                        # Learning rates tipicamente variam em escala logarítmica
                        params[param_name] = trial.suggest_float(param_name, min(param_range), max(param_range), log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, min(param_range), max(param_range))
                elif isinstance(param_range[0], bool):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                # Range contínuo (min, max)
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    if param_name.startswith('lr'):
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1], log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
        
        logger.info(f"Trial {trial.number} com parâmetros: {params}")
        
        # Criar modelo com base nos parâmetros
        from models import create_model_with_best_params
        model = create_model_with_best_params(params, model_name=model_name)
        
        # Criar otimizador e scheduler
        from models import create_optimizer_and_scheduler
        optimizer, scheduler, _ = create_optimizer_and_scheduler(model, params)
        
        # Criar dataloaders (reduzidos para avaliação mais rápida)
        max_train = min(2000, len(train_dataset))
        max_test = min(500, len(test_dataset))
        
        train_indices = torch.randperm(len(train_dataset))[:max_train].tolist()
        test_indices = torch.randperm(len(test_dataset))[:max_test].tolist()
        
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)
        
        batch_size = params.get('batch_size', 32)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Configurar treinamento
        criterion = nn.CrossEntropyLoss()
        epochs = params.get('epochs', 3)  # Poucas épocas para otimização
        
        # Treinar modelo
        from training import train_model
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=f"tensorboard_logs/optim_{stage_name}_trial{trial.number}")
            
            model, _, _, val_losses, val_accs = train_model(
                model, train_loader, test_loader, criterion, optimizer, scheduler,
                epochs, writer, f"{model_name}_trial{trial.number}", use_mixup=params.get('use_mixup', False),
                use_cutmix=params.get('use_cutmix', False), early_stopping=True, patience=2
            )
            
            # Usar a melhor acurácia de validação como métrica para Optuna
            accuracy = max(val_accs) if val_accs else 0.0
            writer.close()
            
            # Liberar memória
            del model, optimizer, scheduler, train_loader, test_loader
            torch.cuda.empty_cache()
            gc.collect()
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Erro durante avaliação do trial {trial.number}: {str(e)}")
            # Liberar memória em caso de erro
            try:
                del model, optimizer, scheduler, train_loader, test_loader
                torch.cuda.empty_cache()
                gc.collect()
            except:
                pass
            return 0.0
        
    # Função para registrar cada trial
    def log_trial(study, trial):
        if trial.state == TrialState.COMPLETE:
            logger.info(f"Trial {trial.number} completado com valor: {trial.value:.4f}")
            if study.best_trial.number == trial.number:
                logger.info(f"NOVO MELHOR TRIAL: {trial.number} com valor: {trial.value:.4f}")
                
                # Salvar parâmetros desta fase
                best_params_path = os.path.join("optuna_results", f"{model_name}_{stage_name}_best_params.json")
                os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
                with open(best_params_path, 'w') as f:
                    json.dump({
                        'trial': trial.number,
                        'value': trial.value,
                        'params': trial.params
                    }, f, indent=2)
    
    # Criar estudo Optuna
    study_name = f"hierarchical_{model_name}_{stage_name}"
    study = optuna.create_study(direction='maximize', study_name=study_name)
    
    # Executar otimização
    study.optimize(objective, n_trials=n_trials, callbacks=[log_trial])
    
    # Registrar e retornar melhores parâmetros
    logger.info(f"Fase '{stage_name}' concluída. Melhor valor: {study.best_value:.4f}")
    logger.info(f"Melhores parâmetros: {study.best_params}")
    
    return study.best_params

def hierarchical_optimization(train_dataset, test_dataset, model_name='resnet50'):
    """
    Executa otimização hierárquica em múltiplas fases.
    
    Args:
        train_dataset: Dataset de treinamento
        test_dataset: Dataset de teste
        model_name: Nome do modelo a ser otimizado
        
    Returns:
        dict: Melhores hiperparâmetros combinados de todas as fases
    """
    logger.info(f"Iniciando otimização hierárquica para {model_name}")
    
    # Fase 1: Otimizar hiperparâmetros fundamentais
    stage1_params = {
        'batch_size': [16, 32, 64],
        'learning_rate': (1e-4, 1e-2),  # Range contínuo
        'optimizer': ['Adam', 'SGD', 'AdamW']
    }
    
    best_stage1 = optimize_stage(stage1_params, train_dataset, test_dataset, 
                                model_name=model_name, n_trials=7, stage_name="stage1")
    
    # Fase 2: Otimizar arquitetura de rede com params fundamentais fixos
    stage2_params = {
        # Fixar parâmetros da fase 1
        'batch_size': [best_stage1['batch_size']],
        'learning_rate': [best_stage1['learning_rate']],
        'optimizer': [best_stage1['optimizer']],
        
        # Novos parâmetros para otimizar
        'dropout_rate': (0.0, 0.5),
    }
    
    # Adicionar parâmetros específicos por modelo
    if model_name == 'resnet50':
        stage2_params.update({
            'freeze_layers': [0, 3, 6, 9],
            'n_layers': [1, 2, 3],
            'fc_size_0': [128, 256, 512, 1024]
        })
    elif model_name == 'efficientnet' or model_name == 'mobilenet':
        stage2_params.update({
            'freeze_percent': (0.0, 0.9)
        })
    elif model_name == 'vit':
        stage2_params.update({
            'model_type': ['vit_small_patch16_224', 'vit_base_patch16_224', 
                          'deit_small_patch16_224']
        })
    
    best_stage2 = optimize_stage(stage2_params, train_dataset, test_dataset, 
                                model_name=model_name, n_trials=8, stage_name="stage2")
    
    # Fase 3: Otimizar schedulers e técnicas avançadas
    stage3_params = {
        # Fixar parâmetros das fases anteriores
        'batch_size': [best_stage1['batch_size']],
        'learning_rate': [best_stage1['learning_rate']],
        'optimizer': [best_stage1['optimizer']],
        'dropout_rate': [best_stage2['dropout_rate']],
    }
    
    # Adicionar parâmetros específicos fixos por modelo
    if model_name == 'resnet50':
        stage3_params.update({
            'freeze_layers': [best_stage2['freeze_layers']],
            'n_layers': [best_stage2['n_layers']],
            'fc_size_0': [best_stage2['fc_size_0']]
        })
    elif model_name == 'efficientnet' or model_name == 'mobilenet':
        stage3_params.update({
            'freeze_percent': [best_stage2['freeze_percent']]
        })
    elif model_name == 'vit':
        stage3_params.update({
            'model_type': [best_stage2['model_type']]
        })
    
    # Novos parâmetros para otimizar
    stage3_params.update({
        'scheduler': ['StepLR', 'CosineAnnealingLR', 'OneCycleLR'],
        'use_mixup': [True, False],
        'use_cutmix': [True, False],
        'label_smoothing': (0.0, 0.3)
    })
    
    # Adicionar parâmetros específicos de scheduler
    stage3_params.update({
        'step_size': [3, 5, 7],  # Para StepLR
        'gamma': (0.1, 0.5),     # Para StepLR
        'T_max': [5, 10],        # Para CosineAnnealingLR
        'max_lr': (0.001, 0.01)  # Para OneCycleLR
    })
    
    best_stage3 = optimize_stage(stage3_params, train_dataset, test_dataset, 
                                model_name=model_name, n_trials=10, stage_name="stage3")
    
    # Combinar todos os melhores parâmetros
    final_params = {**best_stage1, **best_stage2, **best_stage3}
    
    # Salvar combinação final
    final_params_path = os.path.join("optuna_results", f"{model_name}_hierarchical_final.json")
    with open(final_params_path, 'w') as f:
        json.dump({
            'params': final_params,
            'stages': {
                'stage1': best_stage1,
                'stage2': best_stage2,
                'stage3': best_stage3
            }
        }, f, indent=2)
    
    logger.info(f"Otimização hierárquica concluída para {model_name}")
    logger.info(f"Parâmetros finais salvos em {final_params_path}")
    
    return final_params

def run_naturelight_optimization(train_dataset, test_dataset):
    """
    Executa otimização específica para o NatureLightNet
    
    Args:
        train_dataset: Dataset de treinamento
        test_dataset: Dataset de teste
        
    Returns:
        dict: Melhores hiperparâmetros para o NatureLightNet
    """
    logger.info("Iniciando otimização para NatureLightNet")
    
    # Função específica para criar o NatureLightNet
    def create_model_with_params(params):
        from NatureLightNet import create_naturelight_model
        num_classes = 6  # Número padrão de classes para o dataset
        
        model = create_naturelight_model(
            num_classes=num_classes,
            input_size=params.get('input_size', 224),
            dropout_rate=params.get('dropout_rate', 0.2)
        )
        
        return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Definir ranges de parâmetros específicos para o NatureLightNet
    naturelight_params = {
        'batch_size': [16, 32, 64],
        'learning_rate': (1e-4, 5e-3),
        'optimizer': ['Adam', 'AdamW'],
        'dropout_rate': (0.0, 0.4),
        'scheduler': ['CosineAnnealingLR', 'OneCycleLR'],
        'T_max': [5, 10],
        'max_lr': (0.001, 0.01),
        'weight_decay': (1e-5, 1e-3),
        'use_mixup': [False, True],
        'label_smoothing': (0.0, 0.2)
    }
    
    # Função objetivo adaptada para o NatureLightNet
    def objective(trial):
        # Limpar memória GPU
        torch.cuda.empty_cache()
        gc.collect()
        
        # Gerar hiperparâmetros de acordo com os ranges definidos
        params = {}
        for param_name, param_range in naturelight_params.items():
            if isinstance(param_range, list):
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, min(param_range), max(param_range))
                elif isinstance(param_range[0], float):
                    if param_name.startswith('lr'):
                        params[param_name] = trial.suggest_float(param_name, min(param_range), max(param_range), log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, min(param_range), max(param_range))
                elif isinstance(param_range[0], bool):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    if param_name.startswith('lr'):
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1], log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
        
        logger.info(f"Trial {trial.number} com parâmetros: {params}")
        
        # Criar modelo NatureLightNet
        model = create_model_with_params(params)
        
        # Criar otimizador
        lr = params.get('learning_rate', 0.001)
        optimizer_name = params.get('optimizer', 'Adam')
        weight_decay = params.get('weight_decay', 0.0)
        
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:  # AdamW
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Criar scheduler
        scheduler_name = params.get('scheduler', 'CosineAnnealingLR')
        if scheduler_name == 'CosineAnnealingLR':
            T_max = params.get('T_max', 10)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        else:  # OneCycleLR
            max_lr = params.get('max_lr', 0.01)
            
            # Criar dataloaders para determinar steps_per_epoch
            batch_size = params.get('batch_size', 32)
            max_train = min(2000, len(train_dataset))
            train_indices = torch.randperm(len(train_dataset))[:max_train].tolist()
            train_subset = Subset(train_dataset, train_indices)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=5
            )
        
        # Criar dataloaders para avaliação
        max_train = min(2000, len(train_dataset))
        max_test = min(500, len(test_dataset))
        
        train_indices = torch.randperm(len(train_dataset))[:max_train].tolist()
        test_indices = torch.randperm(len(test_dataset))[:max_test].tolist()
        
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)
        
        batch_size = params.get('batch_size', 32)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Configurar treinamento
        label_smoothing = params.get('label_smoothing', 0.0)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        epochs = 5  # Poucas épocas para otimização
        
        # Treinar modelo
        from training import train_model
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=f"tensorboard_logs/naturelight_trial{trial.number}")
            
            model, _, _, val_losses, val_accs = train_model(
                model, train_loader, test_loader, criterion, optimizer, scheduler,
                epochs, writer, f"naturelight_trial{trial.number}", use_mixup=params.get('use_mixup', False),
                use_cutmix=False, early_stopping=True, patience=2
            )
            
            # Usar a melhor acurácia de validação como métrica para Optuna
            accuracy = max(val_accs) if val_accs else 0.0
            writer.close()
            
            # Liberar memória
            del model, optimizer, scheduler, train_loader, test_loader
            torch.cuda.empty_cache()
            gc.collect()
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Erro durante avaliação do trial {trial.number} para NatureLightNet: {str(e)}")
            # Liberar memória em caso de erro
            try:
                del model, optimizer, scheduler, train_loader, test_loader
                torch.cuda.empty_cache()
                gc.collect()
            except:
                pass
            return 0.0
    
    # Criar estudo Optuna
    study_name = "naturelight_optimization"
    study = optuna.create_study(direction='maximize', study_name=study_name)
    
    # Função para registrar cada trial
    def log_trial(study, trial):
        if trial.state == TrialState.COMPLETE:
            logger.info(f"Trial {trial.number} completado com valor: {trial.value:.4f}")
            if study.best_trial.number == trial.number:
                logger.info(f"NOVO MELHOR TRIAL: {trial.number} com valor: {trial.value:.4f}")
                
                # Salvar melhores parâmetros
                best_params_path = os.path.join("optuna_results", "naturelight_best_params.json")
                os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
                with open(best_params_path, 'w') as f:
                    json.dump({
                        'trial': trial.number,
                        'value': trial.value,
                        'params': trial.params
                    }, f, indent=2)
    
    # Executar otimização
    n_trials = 15
    logger.info(f"Iniciando otimização NatureLightNet com {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, callbacks=[log_trial])
    
    # Registrar e retornar melhores parâmetros
    logger.info(f"Otimização NatureLightNet concluída. Melhor valor: {study.best_value:.4f}")
    logger.info(f"Melhores parâmetros: {study.best_params}")
    
    return study.best_params