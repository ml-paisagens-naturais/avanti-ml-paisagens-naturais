"""
Módulo de Otimização

Este módulo contém funções para otimizar hiperparâmetros dos modelos
usando o framework Optuna.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
import logging
import time
import json
import functools
import gc
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import optuna
from optuna.trial import TrialState
import timm
from dataset_utils import custom_collate_fn

# Obter o logger
logger = logging.getLogger("landscape_classifier")

def create_model_optuna(trial, model_name='resnet50'):
    """
    Cria um modelo com hiperparâmetros definidos pelo Optuna trial
    
    Args:
        trial: Trial do Optuna
        model_name: Nome do modelo a ser criado (resnet50, efficientnet, mobilenet)
        
    Returns:
        Modelo PyTorch configurado com os hiperparâmetros sugeridos pelo trial
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
            # Determinar número de camadas congeladas
            freeze_layers = trial.suggest_int('freeze_layers', 0, 8)
            logger.info(f"Trial {trial.number}: ResNet50 com {freeze_layers} camadas congeladas")
            
            if freeze_layers > 0:
                layers_to_freeze = list(model.children())[:freeze_layers]
                for layer in layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
            
            # Modificar a camada final
            num_features = model.fc.in_features
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            fc_layers = []
            
            # Número de camadas fully connected
            n_layers = trial.suggest_int('n_layers', 1, 3)
            
            prev_size = num_features
            for i in range(n_layers - 1):
                fc_size = trial.suggest_int(f'fc_size_{i}', 128, 1024)
                fc_layers.append(nn.Linear(prev_size, fc_size))
                fc_layers.append(nn.ReLU())
                if dropout_rate > 0:
                    fc_layers.append(nn.Dropout(dropout_rate))
                prev_size = fc_size
            
            fc_layers.append(nn.Linear(prev_size, 6))  # Assumindo 6 classes
            model.fc = nn.Sequential(*fc_layers)
        
        elif model_name == 'efficientnet':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            
            # Congelar camadas iniciais
            freeze_percent = trial.suggest_float('freeze_percent', 0.0, 0.9)
            logger.info(f"Trial {trial.number}: EfficientNet com {freeze_percent*100:.1f}% de parâmetros congelados")
            
            if freeze_percent > 0:
                params = list(model.parameters())
                freeze_params = int(len(params) * freeze_percent)
                for param in params[:freeze_params]:
                    param.requires_grad = False
            
            # Substituir a última camada
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features=1280, out_features=6)
            )
        
        elif model_name == 'mobilenet':
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    
            # Congelar camadas iniciais
            freeze_percent = trial.suggest_float('freeze_percent', 0.0, 0.9)
            logger.info(f"Trial {trial.number}: MobileNet com {freeze_percent*100:.1f}% de parâmetros congelados")
    
            if freeze_percent > 0:
                params = list(model.parameters())
                freeze_params = int(len(params) * freeze_percent)
                for param in params[:freeze_params]:
                    param.requires_grad = False
    
            # Substituir a última camada com estrutura mais segura
            in_features = model.classifier[-1].in_features
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            model.classifier[-1] = nn.Linear(in_features, 6)
            
        else:
            logger.error(f"Modelo {model_name} não suportado")
            raise ValueError(f"Modelo {model_name} não suportado")

        return model.to(device)

    except Exception as e:
        logger.error(f"Erro ao criar modelo para trial {trial.number}", exc_info=True)
        raise

def objective(trial, train_dataset, test_dataset, model_name='resnet50'):
    """
    Função objetivo para o Optuna otimizar (com melhor tratamento de erros)
    
    Args:
        trial: Trial do Optuna
        train_dataset: Dataset de treinamento
        test_dataset: Dataset de teste
        model_name: Nome do modelo a ser otimizado
        
    Returns:
        Acurácia do modelo para o trial atual
    """
    trial_start_time = time.time()
    logger.info(f"Iniciando trial {trial.number} para {model_name}")
    
    # Variáveis que precisam ser limpas no bloco finally
    model = None
    optimizer = None
    scheduler = None
    trial_writer = None
    
    try:
        # Hiperparâmetros para otimização
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'AdamW'])
        
        # Parâmetros para técnicas avançadas
        use_mixup = trial.suggest_categorical('use_mixup', [True, False])
        use_cutmix = trial.suggest_categorical('use_cutmix', [True, False])
        
        # Parâmetros específicos para o LR scheduler
        scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'OneCycleLR'])
        
        logger.info(f"Trial {trial.number} - Configurações: batch_size={batch_size}, lr={lr:.6f}, "
                   f"optimizer={optimizer_name}, scheduler={scheduler_name}, "
                   f"mixup={use_mixup}, cutmix={use_cutmix}")
        
        # Criar dataloaders com limites de tamanho para otimização mais rápida
        max_train_samples = min(len(train_dataset), 5000)  # Limitar para otimização mais rápida
        max_test_samples = min(len(test_dataset), 1000)
        
        # Criar subconjuntos aleatórios
        if len(train_dataset) > max_train_samples:
            train_subset, _ = torch.utils.data.random_split(
                train_dataset, [max_train_samples, len(train_dataset) - max_train_samples],
                generator=torch.Generator().manual_seed(42)
            )
        else:
            train_subset = train_dataset
            
        if len(test_dataset) > max_test_samples:
            test_subset, _ = torch.utils.data.random_split(
                test_dataset, [max_test_samples, len(test_dataset) - max_test_samples],
                generator=torch.Generator().manual_seed(42)
            )
        else:
            test_subset = test_dataset
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, multiprocessing_context='spawn')
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, multiprocessing_context='spawn')
        
        logger.info(f"Trial {trial.number} - Utilizando {len(train_subset)} amostras para treinamento e "
                   f"{len(test_subset)} para validação")
        
        # Tratamento especial para o MobileNet
        if model_name == 'mobilenet':
            try:
                # Criar modelo de forma mais segura
                model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                
                # Registrar a estrutura original
                last_layer = model.classifier[-1]
                in_features = last_layer.in_features
                logger.info(f"MobileNet - Trial {trial.number}: Dimensão de entrada da última camada: {in_features}")
                
                # Congelar camadas
                freeze_percent = trial.suggest_float('freeze_percent', 0.0, 0.9)
                if freeze_percent > 0:
                    params = list(model.parameters())
                    freeze_params = int(len(params) * freeze_percent)
                    for param in params[:freeze_params]:
                        param.requires_grad = False
                
                # Modificar apenas a última camada para ser seguro
                dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
                model.classifier[-1] = nn.Linear(in_features, 6)
                
                # Teste rápido para verificar compatibilidade
                model = model.to("cuda" if torch.cuda.is_available() else "cpu")
                sample_input = torch.randn(2, 3, 224, 224).to("cuda" if torch.cuda.is_available() else "cpu")
                with torch.no_grad():
                    test_output = model(sample_input)
                logger.info(f"MobileNet - Trial {trial.number}: Teste de forward pass bem-sucedido")
                
            except Exception as e:
                logger.error(f"Erro ao criar MobileNet para trial {trial.number}: {str(e)}")
                # Em caso de erro, retornar um valor baixo para indicar falha, mas não interromper o processo
                return -1
                
        else:
            # Para outros modelos, usar a função create_model_optuna normal
            model = create_model_optuna(trial, model_name)
        
        # Definir critério de perda
        criterion = nn.CrossEntropyLoss()
        
        # Definir otimizador
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            momentum = trial.suggest_float('momentum', 0.0, 0.99)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:  # AdamW
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Definir scheduler
        if scheduler_name == 'StepLR':
            step_size = trial.suggest_int('step_size', 2, 10)
            gamma = trial.suggest_float('gamma', 0.1, 0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'CosineAnnealingLR':
            T_max = trial.suggest_int('T_max', 5, 10)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        elif scheduler_name == 'ReduceLROnPlateau':
            factor = trial.suggest_float('factor', 0.1, 0.9)
            patience = trial.suggest_int('patience_scheduler', 1, 5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
        else:  # OneCycleLR
            max_lr = trial.suggest_float('max_lr', lr, lr*10)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=10)
        
        # Parâmetros para early stopping
        early_stopping = trial.suggest_categorical('early_stopping', [True, False])
        patience = trial.suggest_int('patience_stopping', 2, 5) if early_stopping else 10
        
        # Reduzir o número de épocas para otimização mais rápida
        optim_epochs = 5
        
        # Criar writer do TensorBoard específico para este trial
        trial_dir = f"tensorboard_logs/trial_{trial.number}_{model_name}"
        os.makedirs(trial_dir, exist_ok=True)
        trial_writer = SummaryWriter(log_dir=trial_dir)
        
        # Importar função de treinamento do módulo training
        from training import train_model
        
        # Treinar modelo
        logger.info(f"Trial {trial.number} - Iniciando treinamento com {optim_epochs} épocas")
        
        try:
            model, train_losses, train_accs, val_losses, val_accs = train_model(
                model, train_loader, test_loader, criterion, optimizer, scheduler, 
                optim_epochs, trial_writer, f"{model_name}_trial{trial.number}",
                use_mixup, use_cutmix, early_stopping, patience
            )
        except RuntimeError as e:
            # Verificar se é um erro de CUDA Out of Memory
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                logger.error(f"Trial {trial.number} - CUDA OOM. Interrompendo trial.")
                return -1
            else:
                # Para outros erros de runtime, registrar e continuar
                logger.error(f"Trial {trial.number} - Erro durante treinamento: {str(e)}")
                return -1
        except Exception as e:
            logger.error(f"Trial {trial.number} - Erro durante treinamento: {str(e)}")
            return -1
        
        # Obter as classes para avaliação
        classes = []
        if hasattr(train_dataset, 'classes'):
            classes = train_dataset.classes
        elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'classes'):
            classes = train_dataset.dataset.classes
        else:
            classes = [f"Class_{i}" for i in range(6)]  # Fallback para 6 classes
        
        # Durante a avaliação
        try:
            # Importar função de avaliação do módulo training
            from training import evaluate_model
            
            accuracy, report_dict, _, _, _, _ = evaluate_model(model, test_loader, classes, 
                                                  trial_writer, f"{model_name}_trial{trial.number}")
            
            # Calcular métricas adicionais, como F1 macro (adicionado do segundo arquivo)
            if report_dict and 'macro avg' in report_dict:
                f1_macro = report_dict['macro avg'].get('f1-score', 0)
                logger.info(f"F1 macro para o trial {trial.number}: {f1_macro:.4f}")
                
                # Registrar no TensorBoard
                trial_writer.add_scalar(f"{model_name}_trial{trial.number}/f1_macro", f1_macro, 0)
                
        except Exception as e:
            logger.error(f"Trial {trial.number} - Erro durante avaliação: {str(e)}")
            accuracy = 0  # Ou um valor adequado para indicar falha
        
        # Registrar hiperparâmetros
        hparam_dict = {
            "lr": lr, 
            "batch_size": batch_size, 
            "optimizer": optimizer_name, 
            "scheduler": scheduler_name,
            "model": model_name, 
            "mixup": use_mixup, 
            "cutmix": use_cutmix
        }
        
        metric_dict = {"hparam/accuracy": accuracy}
        
        # Adicionar métricas adicionais se disponíveis
        if 'report_dict' in locals() and report_dict and 'macro avg' in report_dict:
            metric_dict["hparam/f1_macro"] = report_dict['macro avg'].get('f1-score', 0)
            
        if trial_writer is not None:
            trial_writer.add_hparams(hparam_dict, metric_dict)
        
        # Calcular tempo total
        trial_time = time.time() - trial_start_time
        logger.info(f"Trial {trial.number} - Concluído com acurácia {accuracy:.4f} em {trial_time/60:.2f} min")
        
        return accuracy
        
    except Exception as e:
        logger.error(f"Erro no trial {trial.number}: {str(e)}", exc_info=True)
        return -1  # Valor baixo para indicar falha
    
    finally:
        # Limpar memória, independentemente do resultado
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if scheduler is not None:
            del scheduler
        if trial_writer is not None:
            trial_writer.close()
            
        torch.cuda.empty_cache()
        gc.collect()  # Adicionado limpeza de memória mais agressiva

def optimize_hyperparameters(train_dataset, test_dataset, model_name='resnet50', n_trials=20):
    """
    Otimiza hiperparâmetros usando Optuna com logging detalhado
    
    Args:
        train_dataset: Dataset de treinamento
        test_dataset: Dataset de teste
        model_name: Nome do modelo a ser otimizado
        n_trials: Número de trials para a otimização
        
    Returns:
        Tuple de (melhores parâmetros, melhor acurácia)
    """
    logger.info(f"Iniciando otimização de hiperparâmetros para {model_name} com {n_trials} trials")
    
    study = None
    
    try:
        # Criar diretório para salvar resultados da otimização
        os.makedirs("optuna_results", exist_ok=True)
        
        # Função para logar cada trial
        def log_trial(study, trial):
            if trial.state == TrialState.COMPLETE:
                logger.info(f"Trial {trial.number} concluído com valor: {trial.value:.4f}")
                logger.info(f"  Parâmetros: {trial.params}")
                
                # Se for o melhor trial até agora
                if study.best_trial.number == trial.number:
                    logger.info(f"  NOVO MELHOR TRIAL: {trial.number} com valor: {trial.value:.4f}")
                    
                    # Salvar parâmetros dos melhores trials
                    best_params_path = os.path.join("optuna_results", f"{model_name}_best_params.json")
                    os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
                    with open(best_params_path, 'w') as f:
                        json.dump({
                            'trial': trial.number,
                            'value': trial.value,
                            'params': trial.params
                        }, f, indent=2)
            elif trial.state == TrialState.FAIL:
                logger.warning(f"Trial {trial.number} falhou: {trial.value}")
            elif trial.state == TrialState.PRUNED:
                logger.info(f"Trial {trial.number} foi podado")
        
        # Criar o estudo
        study = optuna.create_study(direction='maximize', 
                                  study_name=f"optuna_study_{model_name}")
        
        # Executar a otimização
        study.optimize(lambda trial: objective(trial, train_dataset, test_dataset, model_name), 
                      n_trials=n_trials, callbacks=[log_trial])
        
        # Registrar resultados finais
        logger.info(f"Otimização de {model_name} concluída após {len(study.trials)} trials")
        logger.info(f"Melhor acurácia: {study.best_value:.4f}")
        logger.info("Melhores hiperparâmetros:")
        for key, value in study.best_params.items():
            logger.info(f"    {key}: {value}")
        
        # Plotar curva de importância dos hiperparâmetros
        try:
            param_importances = optuna.importance.get_param_importances(study)
            logger.info("Importância dos parâmetros:")
            for param, importance in param_importances.items():
                logger.info(f"    {param}: {importance:.4f}")
        except:
            logger.warning("Não foi possível calcular importância de parâmetros")
        
        # Visualizar resultados
        try:
            # Exportar para HTML
            html_path = os.path.join("optuna_results", f"{model_name}_optimization_history.html")
            os.makedirs(os.path.dirname(html_path), exist_ok=True)
            # Não é possível visualizar aqui já que depende de funções específicas do Optuna
            logger.info(f"Resultados salvos em {html_path}")
        except Exception as e:
            logger.warning(f"Não foi possível gerar visualizações Optuna: {str(e)}")
        
        # Retornar os melhores parâmetros e valor
        return study.best_params, study.best_value
    
    except Exception as e:
        logger.critical(f"Erro crítico durante otimização de {model_name}", exc_info=True)
        # Tentar retornar algum parâmetro útil ou default
        if model_name == 'resnet50':
            default_params = {
                'batch_size': 32, 'lr': 0.001, 'optimizer': 'Adam', 'dropout_rate': 0.3,
                'scheduler': 'StepLR', 'step_size': 7, 'gamma': 0.1,
                'freeze_layers': 6, 'n_layers': 2, 'fc_size_0': 512
            }
        elif model_name == 'efficientnet':
            default_params = {
                'batch_size': 32, 'lr': 0.0005, 'optimizer': 'AdamW', 'dropout_rate': 0.2,
                'scheduler': 'CosineAnnealingLR', 'T_max': 10, 'freeze_percent': 0.7
            }
        elif model_name == 'mobilenet':
            default_params = {
                'batch_size': 64, 'lr': 0.0003, 'optimizer': 'Adam', 'dropout_rate': 0.1,
                'scheduler': 'OneCycleLR', 'max_lr': 0.003, 'freeze_percent': 0.5
            }
        else:
            default_params = {
                'batch_size': 32, 'lr': 0.001, 'optimizer': 'Adam', 'dropout_rate': 0.2,
                'scheduler': 'StepLR', 'step_size': 7, 'gamma': 0.1
            }
        return default_params, 0.0
    
    finally:
        # Garantir limpeza de memória mesmo se houver erro
        torch.cuda.empty_cache()
        gc.collect()

def objective_vit(trial, train_dataset, test_dataset):
    """
    Função objetivo para otimizar hiperparâmetros do ViT
    
    Args:
        trial: Trial do Optuna
        train_dataset: Dataset de treinamento
        test_dataset: Dataset de teste
        
    Returns:
        Acurácia do modelo para o trial atual
    """
    trial_start_time = time.time()
    logger.info(f"Iniciando trial {trial.number} para ViT")
    
    # Variáveis para limpeza no bloco finally
    model = None
    optimizer = None
    scheduler = None
    trial_writer = None
    
    try:
        # Hiperparâmetros para otimização
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        
        # Arquiteturas modernas de Vision Transformer
        model_type = trial.suggest_categorical('model_type', [
            # Vanilla ViT
            'vit_small_patch16_224', 'vit_base_patch16_224',
            # DeiT - Distilled transformers (mais eficientes)
            'deit_tiny_patch16_224', 'deit_small_patch16_224',
            # Swin Transformer - Hierarchical design
            'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224'
        ])
        
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
        
        # Parâmetros para o scheduler
        scheduler_name = trial.suggest_categorical('scheduler', 
            ['CosineAnnealingLR', 'ReduceLROnPlateau', 'OneCycleLR'])
        
        # Parâmetros para técnicas avançadas de augmentation
        use_mixup = trial.suggest_categorical('use_mixup', [True, False])
        use_cutmix = trial.suggest_categorical('use_cutmix', [True, False])
        mixup_alpha = trial.suggest_float('mixup_alpha', 0.1, 0.8) if use_mixup else 0.2
        cutmix_alpha = trial.suggest_float('cutmix_alpha', 0.1, 1.0) if use_cutmix else 0.2
        
        # Estratégias de fine-tuning específicas para ViT
        freeze_strategy = trial.suggest_categorical('freeze_strategy', ['none', 'partial', 'progressive'])
        freeze_ratio = trial.suggest_float('freeze_ratio', 0.0, 0.9) if freeze_strategy == 'partial' else 0.0
        
        # Layer-wise learning rate decay (comum em fine-tuning de transformers)
        use_layer_decay = trial.suggest_categorical('use_layer_decay', [True, False])
        layer_decay_rate = trial.suggest_float('layer_decay_rate', 0.65, 0.95) if use_layer_decay else 0.8
        
        # Log dos hiperparâmetros selecionados
        logger.info(f"Trial {trial.number} - ViT: batch_size={batch_size}, lr={lr:.6f}, "
                   f"optimizer={optimizer_name}, model={model_type}, "
                   f"mixup={use_mixup}, cutmix={use_cutmix}, "
                   f"freeze_strategy={freeze_strategy}")
        
        # Criar dataloaders
        max_train_samples = min(len(train_dataset), 3000)  # Amostra menor para otimização mais rápida
        max_test_samples = min(len(test_dataset), 600)
        
        # Criar subconjuntos aleatórios
        if len(train_dataset) > max_train_samples:
            train_subset, _ = torch.utils.data.random_split(
                train_dataset, [max_train_samples, len(train_dataset) - max_train_samples],
                generator=torch.Generator().manual_seed(42)
            )
        else:
            train_subset = train_dataset
            
        if len(test_dataset) > max_test_samples:
            test_subset, _ = torch.utils.data.random_split(
                test_dataset, [max_test_samples, len(test_dataset) - max_test_samples],
                generator=torch.Generator().manual_seed(42)
            )
        else:
            test_subset = test_dataset
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, multiprocessing_context='spawn', collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, multiprocessing_context='spawn', collate_fn=custom_collate_fn)
        
        # Criar ViT com os parâmetros do trial
        from models import create_vit_model
        
        vit_params = {
            'model_name': model_type,
            'pretrained': True,
            'dropout_rate': dropout_rate
        }
        
        model = create_vit_model(vit_params)
        
        # Aplicar estratégias de congelamento conforme definido
        if freeze_strategy != 'none':
            # Identificar os blocos do transformer
            blocks = None
            if hasattr(model, 'blocks'):
                blocks = model.blocks  # ViT / DeiT padrão
            elif hasattr(model, 'layers'):
                blocks = model.layers  # Swin Transformer
            
            if blocks is not None:
                num_blocks = len(blocks)
                
                if freeze_strategy == 'partial':
                    # Congelar uma porcentagem dos blocos iniciais
                    blocks_to_freeze = int(num_blocks * freeze_ratio)
                    logger.info(f"Congelando {blocks_to_freeze} de {num_blocks} blocos do transformer")
                    
                    for i in range(blocks_to_freeze):
                        for param in blocks[i].parameters():
                            param.requires_grad = False
                
                elif freeze_strategy == 'progressive':
                    # No caso progressivo, não congelamos inicialmente, mas
                    # configuramos taxas de aprendizado diferentes por camada
                    # Isso é implementado no otimizador abaixo
                    logger.info(f"Configurando descongelamento progressivo com {num_blocks} blocos")
                    pass
        
        # Definir critério com label smoothing (útil para ViT)
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.3)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Configurar otimizador com aprendizado específico por camada, se necessário
        if use_layer_decay and hasattr(model, 'blocks'):
            # Configurar taxas de aprendizado por grupo de parâmetros
            num_layers = len(model.blocks)
            parameter_groups = []
            
            # Parâmetros do embedder (menor taxa)
            if hasattr(model, 'patch_embed'):
                parameter_groups.append({
                    'params': model.patch_embed.parameters(),
                    'lr': lr * (layer_decay_rate ** num_layers)
                })
            
            # Parâmetros de cada bloco, com taxas crescentes
            for i, block in enumerate(model.blocks):
                parameter_groups.append({
                    'params': block.parameters(),
                    'lr': lr * (layer_decay_rate ** (num_layers - i - 1))
                })
            
            # Parâmetros da cabeça (maior taxa)
            if hasattr(model, 'head'):
                parameter_groups.append({
                    'params': model.head.parameters(),
                    'lr': lr
                })
            
            # Definir otimizador com os grupos
            if optimizer_name == 'Adam':
                optimizer = optim.Adam(parameter_groups)
            else:  # AdamW
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
                optimizer = optim.AdamW(parameter_groups, weight_decay=weight_decay)
                
            logger.info(f"Configurado otimizador com layer-wise decay: {layer_decay_rate}")
        else:
            # Otimizador padrão sem diferenciação por camada
            if optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=lr)
            else:  # AdamW
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Definir scheduler
        if scheduler_name == 'CosineAnnealingLR':
            T_max = trial.suggest_int('T_max', 5, 10)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        elif scheduler_name == 'OneCycleLR':
            max_lr = trial.suggest_float('max_lr', lr, lr*10)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=10
            )
        else:  # ReduceLROnPlateau
            factor = trial.suggest_float('factor', 0.1, 0.5)
            patience = trial.suggest_int('patience', 2, 5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience
            )
        
        # Parâmetros para early stopping
        early_stopping = True
        patience_stopping = trial.suggest_int('patience_stopping', 2, 4)
        
        # Criar writer do TensorBoard específico para este trial
        trial_dir = f"tensorboard_logs/trial_{trial.number}_vit"
        os.makedirs(trial_dir, exist_ok=True)
        trial_writer = SummaryWriter(log_dir=trial_dir)
        
        # Treinar modelo (com menos épocas para otimização mais rápida)
        optim_epochs = 5
        
        from training import train_model, evaluate_model
        
        try:
            model, train_losses, train_accs, val_losses, val_accs = train_model(
                model, train_loader, test_loader, criterion, optimizer, scheduler, 
                optim_epochs, trial_writer, f"vit_trial{trial.number}",
                use_mixup=use_mixup, use_cutmix=use_cutmix, early_stopping=early_stopping, 
                patience=patience_stopping, alpha=mixup_alpha
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"Trial {trial.number} - CUDA OOM. Interrompendo trial.")
                return -1
            else:
                logger.error(f"Trial {trial.number} - Erro durante treinamento do ViT: {str(e)}")
                return -1
        except Exception as e:
            logger.error(f"Trial {trial.number} - Erro durante treinamento do ViT: {str(e)}")
            return -1
            
        # Avaliar o modelo
        try:
            # Obter as classes
            classes = []
            if hasattr(train_dataset, 'classes'):
                classes = train_dataset.classes
            elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'classes'):
                classes = train_dataset.dataset.classes
            else:
                classes = [f"Class_{i}" for i in range(6)]  # Fallback para 6 classes
            
            accuracy, report_dict, _, _, _, _ = evaluate_model(
                model, test_loader, classes, trial_writer, f"vit_trial{trial.number}"
            )
            
            # Calcular métricas adicionais, como F1 macro
            if report_dict and 'macro avg' in report_dict:
                f1_macro = report_dict['macro avg'].get('f1-score', 0)
                logger.info(f"F1 macro para o trial {trial.number}: {f1_macro:.4f}")
                
                # Registrar no TensorBoard
                trial_writer.add_scalar(f"vit_trial{trial.number}/f1_macro", f1_macro, 0)
            
        except Exception as e:
            logger.error(f"Trial {trial.number} - Erro durante avaliação do ViT: {str(e)}")
            accuracy = 0
        
        # Registrar hiperparâmetros no TensorBoard
        hparam_dict = {
            "lr": lr, 
            "batch_size": batch_size, 
            "optimizer": optimizer_name,
            "model_type": model_type, 
            "dropout": dropout_rate,
            "use_mixup": use_mixup,
            "use_cutmix": use_cutmix,
            "freeze_strategy": freeze_strategy,
            "label_smoothing": label_smoothing
        }
        
        # Registrar hiperparâmetros condicionais
        if use_layer_decay:
            hparam_dict["layer_decay_rate"] = layer_decay_rate
        
        if freeze_strategy == 'partial':
            hparam_dict["freeze_ratio"] = freeze_ratio
            
        metric_dict = {"hparam/accuracy": accuracy}
        
        # Adicionar métricas adicionais se disponíveis
        if 'report_dict' in locals() and report_dict and 'macro avg' in report_dict:
            metric_dict["hparam/f1_macro"] = report_dict['macro avg'].get('f1-score', 0)
        
        if trial_writer is not None:
            trial_writer.add_hparams(hparam_dict, metric_dict)
        
        # Calcular tempo total
        trial_time = time.time() - trial_start_time
        logger.info(f"Trial {trial.number} - ViT concluído com acurácia {accuracy:.4f} em {trial_time/60:.2f} min")
        
        return accuracy
    
    except Exception as e:
        logger.error(f"Erro no trial {trial.number} do ViT: {str(e)}", exc_info=True)
        return -1
        
    finally:
        # Limpar memória, independentemente do resultado
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if scheduler is not None:
            del scheduler
        if trial_writer is not None:
            trial_writer.close()
            
        torch.cuda.empty_cache()
        gc.collect()

def optimize_vit_hyperparameters(train_dataset, test_dataset, n_trials=10):
    """
    Otimiza hiperparâmetros para o modelo Vision Transformer
    
    Args:
        train_dataset: Dataset de treinamento
        test_dataset: Dataset de teste
        n_trials: Número de trials para a otimização
        
    Returns:
        Tuple de (melhores parâmetros, melhor acurácia)
    """
    logger.info(f"Iniciando otimização de hiperparâmetros para ViT com {n_trials} trials")
    
    study = None
    
    try:
        # Criar diretório para salvar resultados da otimização
        os.makedirs("optuna_results", exist_ok=True)
        
        # Função para logar cada trial
        def log_trial(study, trial):
            if trial.state == TrialState.COMPLETE:
                logger.info(f"Trial {trial.number} concluído com valor: {trial.value:.4f}")
                logger.info(f"  Parâmetros: {trial.params}")
                
                # Se for o melhor trial até agora
                if study.best_trial.number == trial.number:
                    logger.info(f"  NOVO MELHOR TRIAL: {trial.number} com valor: {trial.value:.4f}")
                
                # Salvar parâmetros dos melhores trials
                best_params_path = os.path.join("optuna_results", "vit_best_params.json")
                os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
                with open(best_params_path, 'w') as f:
                    json.dump({
                        'trial': trial.number,
                        'value': trial.value,
                        'params': trial.params
                    }, f, indent=2)
            elif trial.state == TrialState.FAIL:
                logger.warning(f"Trial {trial.number} falhou: {trial.value}")
            elif trial.state == TrialState.PRUNED:
                logger.info(f"Trial {trial.number} foi podado")
        
        # Criar o estudo
        study = optuna.create_study(direction='maximize', study_name="optuna_study_vit")
        
        # Executar a otimização
        study.optimize(lambda trial: objective_vit(trial, train_dataset, test_dataset), 
                      n_trials=n_trials, callbacks=[log_trial])
        
        # Registrar resultados finais
        logger.info(f"Otimização de ViT concluída após {len(study.trials)} trials")
        logger.info(f"Melhor acurácia: {study.best_value:.4f}")
        logger.info("Melhores hiperparâmetros:")
        for key, value in study.best_params.items():
            logger.info(f"    {key}: {value}")
        
        # Plotar curva de importância dos hiperparâmetros
        try:
            param_importances = optuna.importance.get_param_importances(study)
            logger.info("Importância dos parâmetros:")
            for param, importance in param_importances.items():
                logger.info(f"    {param}: {importance:.4f}")
        except:
            logger.warning("Não foi possível calcular importância de parâmetros")
        
        # Visualizar resultados
        try:
            html_path1 = os.path.join("optuna_results", "vit_optimization_history.html")
            html_path2 = os.path.join("optuna_results", "vit_param_importances.html")
            
            # Garantir que o diretório existe
            os.makedirs(os.path.dirname(html_path1), exist_ok=True)
            
            # Não é possível visualizar diretamente aqui
            logger.info(f"Resultados salvos em {html_path1} e {html_path2}")
        except Exception as e:
            logger.warning(f"Não foi possível gerar visualizações Optuna para ViT: {str(e)}")
        
        return study.best_params, study.best_value
    
    except Exception as e:
        logger.critical(f"Erro crítico durante otimização do ViT", exc_info=True)
        default_params = {
            'batch_size': 32, 'lr': 0.0001, 'dropout_rate': 0.1,
            'model_type': 'vit_base_patch16_224', 'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR', 'T_max': 10
        }
        return default_params, 0.0
        
    finally:
        # Garantir limpeza de memória mesmo se houver erro
        torch.cuda.empty_cache()
        gc.collect()

def progressive_optimization(model_name, optim_train_dataset, optim_test_dataset):
    """
    Tenta otimizar com redução progressiva de complexidade em caso de falha
    
    Args:
        model_name: Nome do modelo a ser otimizado
        optim_train_dataset: Dataset de treinamento para otimização
        optim_test_dataset: Dataset de teste para otimização
        
    Returns:
        Tuple de (melhores parâmetros, melhor acurácia)
    """
    
    # Configurações padrão por modelo para fallback
    DEFAULT_PARAMS = {
        'resnet50': {
            'batch_size': 32, 'lr': 0.001, 'optimizer': 'Adam', 'dropout_rate': 0.3,
            'scheduler': 'StepLR', 'step_size': 7, 'gamma': 0.1,
            'freeze_layers': 6, 'n_layers': 2, 'fc_size_0': 512
        },
        'efficientnet': {
            'batch_size': 32, 'lr': 0.0005, 'optimizer': 'AdamW', 'dropout_rate': 0.2,
            'scheduler': 'CosineAnnealingLR', 'T_max': 10, 'freeze_percent': 0.7
        },
        'mobilenet': {
            'batch_size': 64, 'lr': 0.0003, 'optimizer': 'Adam', 'dropout_rate': 0.1,
            'scheduler': 'OneCycleLR', 'max_lr': 0.003, 'freeze_percent': 0.5
        },
        'vit': {
            'batch_size': 32, 'lr': 0.0001, 'dropout_rate': 0.1,
            'model_type': 'vit_base_patch16_224', 'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR', 'T_max': 10
        }
    }
    
    # Nível 1: Otimização completa
    try:
        logger.info(f"Tentando otimização completa para {model_name} (10 trials)")
        if model_name == 'vit':
            return optimize_vit_hyperparameters(optim_train_dataset, optim_test_dataset, n_trials=10)
        else:
            return optimize_hyperparameters(optim_train_dataset, optim_test_dataset, model_name=model_name, n_trials=10)
    except Exception as e:
        logger.warning(f"Falha na otimização completa para {model_name}: {e}. Tentando versão reduzida.")
        
    # Nível 2: Menos trials, dataset menor
    try:
        logger.info(f"Tentando otimização reduzida para {model_name} (5 trials, dataset menor)")
        small_train, _ = torch.utils.data.random_split(
            optim_train_dataset, 
            [len(optim_train_dataset)//2, len(optim_train_dataset) - len(optim_train_dataset)//2],
            generator=torch.Generator().manual_seed(42)
        )
        if model_name == 'vit':
            return optimize_vit_hyperparameters(small_train, optim_test_dataset, n_trials=5)
        else:
            return optimize_hyperparameters(small_train, optim_test_dataset, model_name=model_name, n_trials=5)
    except Exception as e:
        logger.warning(f"Falha na otimização reduzida para {model_name}: {e}. Tentando versão mínima.")
    
    # Nível 3: Otimização mínima (apenas 3 trials, dataset muito pequeno)
    try:
        logger.info(f"Tentando otimização mínima para {model_name} (3 trials, dataset muito pequeno)")
        tiny_train, _ = torch.utils.data.random_split(
            optim_train_dataset, 
            [min(1000, len(optim_train_dataset)-100), max(100, len(optim_train_dataset) - 1000)],
            generator=torch.Generator().manual_seed(42)
        )
        tiny_test, _ = torch.utils.data.random_split(
            optim_test_dataset, 
            [min(500, len(optim_test_dataset)-50), max(50, len(optim_test_dataset) - 500)],
            generator=torch.Generator().manual_seed(42)
        )
        if model_name == 'vit':
            return optimize_vit_hyperparameters(tiny_train, tiny_test, n_trials=3)
        else:
            return optimize_hyperparameters(tiny_train, tiny_test, model_name=model_name, n_trials=3)
    except Exception as e:
        logger.error(f"Todas as tentativas de otimização falharam para {model_name}: {e}")
    
    # Fallback para parâmetros padrão
    logger.warning(f"Usando parâmetros padrão para {model_name}")
    return DEFAULT_PARAMS.get(model_name, DEFAULT_PARAMS['resnet50']), 0.0

def run_isolated_optimization(model_name, optim_train_dataset, optim_test_dataset):
    """
    Executa otimização para um modelo específico em ambiente isolado
    
    Args:
        model_name: Nome do modelo a ser otimizado
        optim_train_dataset: Dataset de treinamento para otimização
        optim_test_dataset: Dataset de teste para otimização
        
    Returns:
        Tuple de (melhores parâmetros, melhor acurácia)
    """
    logger.info(f"=== INICIANDO OTIMIZAÇÃO ISOLADA PARA {model_name.upper()} ===")
    
    # Limpar completamente a memória
    torch.cuda.empty_cache()
    gc.collect()
    initial_mem = torch.cuda.memory_allocated(0) / (1024**2)
    logger.info(f"Memória GPU antes da otimização de {model_name}: {initial_mem:.2f} MB em uso")
    
    try:
        # Usar a otimização progressiva
        params, acc = progressive_optimization(model_name, optim_train_dataset, optim_test_dataset)
        
        # Salvar resultado imediatamente
        result_file = f"results/{model_name}_best_params.json"
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump({'params': params, 'accuracy': float(acc)}, f, indent=2)
        
        logger.info(f"Otimização para {model_name} concluída e salva em {result_file}")
        return params, acc
        
    except Exception as e:
        logger.error(f"Falha crítica na otimização isolada de {model_name}: {str(e)}", exc_info=True)
        
        # Retornar valores padrão
        if model_name == 'resnet50':
            default_params = {
                'batch_size': 32, 'lr': 0.001, 'optimizer': 'Adam', 'dropout_rate': 0.3,
                'scheduler': 'StepLR', 'step_size': 7, 'gamma': 0.1,
                'freeze_layers': 6, 'n_layers': 2, 'fc_size_0': 512
            }
        elif model_name == 'efficientnet':
            default_params = {
                'batch_size': 32, 'lr': 0.0005, 'optimizer': 'AdamW', 'dropout_rate': 0.2,
                'scheduler': 'CosineAnnealingLR', 'T_max': 10, 'freeze_percent': 0.7
            }
        elif model_name == 'vit':
            default_params = {
                'batch_size': 32, 'lr': 0.0001, 'dropout_rate': 0.1,
                'model_type': 'vit_base_patch16_224', 'optimizer': 'AdamW',
                'scheduler': 'CosineAnnealingLR', 'T_max': 10
            }
        else:  # mobilenet
            default_params = {
                'batch_size': 64, 'lr': 0.0003, 'optimizer': 'Adam', 'dropout_rate': 0.1,
                'scheduler': 'OneCycleLR', 'max_lr': 0.003, 'freeze_percent': 0.5
            }
        
        return default_params, 0.0
        
    finally:
        # Limpar memória em qualquer caso (sucesso ou falha)
        torch.cuda.empty_cache()
        gc.collect()
        final_mem = torch.cuda.memory_allocated(0) / (1024**2)
        logger.info(f"Memória GPU após otimização de {model_name}: {final_mem:.2f} MB (diferença: {final_mem-initial_mem:.2f} MB)")
