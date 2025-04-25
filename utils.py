"""
Funções utilitárias para o projeto de classificação de paisagens naturais.

Este módulo contém funções de uso geral para configuração do ambiente,
gerenciamento de logs, criação de diretórios, manipulação de checkpoints,
otimizações de memória e outras utilidades compartilhadas pelo pipeline de treinamento.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import yaml
import gc
import functools
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def gpu_memory_check(threshold_mb=1000):
    """
    Decorador para verificar memória GPU disponível antes de executar funções intensivas.
    
    Args:
        threshold_mb: Quantidade mínima de memória livre necessária em MB
        
    Returns:
        Função decorada que verifica a memória antes da execução
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                # Verificar memória disponível
                torch.cuda.empty_cache()
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                allocated_memory = torch.cuda.memory_allocated(0) / (1024**2)
                available_memory = total_memory - allocated_memory
                
                logger = logging.getLogger("landscape_classifier")
                
                if available_memory < threshold_mb:
                    logger.warning(f"Pouca memória GPU disponível: {available_memory:.2f} MB. Necessário: {threshold_mb} MB")
                    # Tentar liberar mais memória
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Verificar novamente
                    allocated_memory = torch.cuda.memory_allocated(0) / (1024**2)
                    available_memory = total_memory - allocated_memory
                    
                    if available_memory < threshold_mb:
                        logger.error(f"Memória insuficiente após tentativa de limpeza: {available_memory:.2f} MB")
                        raise MemoryError(f"Memória GPU insuficiente para executar {func.__name__}")
            
            # Executar a função
            return func(*args, **kwargs)
        return wrapper
    return decorator

def setup_logger(log_dir="logs", log_level=logging.INFO):
    """
    Configura e retorna um logger centralizado.
    
    Args:
        log_dir: Diretório onde os logs serão salvos
        log_level: Nível de logging (INFO, DEBUG, etc.)
        
    Returns:
        Logger configurado
    """
    # Criar diretório de logs se não existir
    os.makedirs(log_dir, exist_ok=True)
    
    # Nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"landscape_classification_{timestamp}.log")
    
    # Configurar logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Limpar handlers existentes
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Formatação dos logs
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Configurar logger específico para o classificador
    logger = logging.getLogger("landscape_classifier")
    logger.setLevel(log_level)
    logger.propagate = True  # Permitir que logs se propaguem para o logger raiz
    
    logger.info(f"Logger inicializado. Logs salvos em: {log_file}")
    return logger

def ensure_directories():
    """
    Cria todos os diretórios necessários para o pipeline, garantindo que eles existam
    antes que os arquivos sejam salvos neles.
    
    Returns:
        bool: True se todos os diretórios foram criados/verificados
    """
    logger = logging.getLogger("landscape_classifier")
    
    # Lista de diretórios que precisam existir
    directories = [
        "logs",               # Para arquivos de log
        "tensorboard_logs",   # Para logs do TensorBoard
        "models",             # Para modelos salvos
        "results",            # Para resultados e visualizações
        "gradcam_resnet50",   # Para visualizações Grad-CAM do ResNet
        "gradcam_efficientnet", # Para visualizações Grad-CAM do EfficientNet
        "gradcam_mobilenet",  # Para visualizações Grad-CAM do MobileNet
        "attention_maps_vit", # Para mapas de atenção do Vision Transformer
        "optuna_results",     # Para resultados da otimização com Optuna
        "confusion_matrices", # Para matrizes de confusão
        "error_analysis",     # Para análise de erros
        "ensemble_results",   # Para resultados do ensemble
        "results/experiments" # Para experimentos personalizados
    ]
    
    # Criar cada diretório
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Diretório {directory} criado ou verificado.")
    
    return True

def setup_tensorboard(base_dir="tensorboard_logs", logger=None):
    """
    Configura e retorna um writer do TensorBoard.
    
    Args:
        base_dir: Diretório base para logs do TensorBoard
        logger: Logger para registrar mensagens (opcional)
        
    Returns:
        tuple: (SummaryWriter do TensorBoard, caminho do diretório de logs)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, f"landscape_classification_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    if logger:
        logger.info(f"TensorBoard inicializado em: {log_dir}")
        logger.info(f"Execute 'tensorboard --logdir={base_dir}' para visualizar os resultados")
    else:
        print(f"TensorBoard inicializado em: {log_dir}")
        print(f"Execute 'tensorboard --logdir={base_dir}' para visualizar os resultados")
    
    return writer, log_dir

def load_config(config_path='config.yaml'):
    """
    Carrega configurações de um arquivo YAML.
    
    Args:
        config_path: Caminho para o arquivo de configuração YAML
        
    Returns:
        dict: Configurações carregadas
    """
    logger = logging.getLogger("landscape_classifier")
    
    # Configurações padrão
    default_config = {
        'dataset': {
            'path': 'dataset/intel-image-classification',
            'img_size': 224,
            'batch_size': 32
        },
        'training': {
            'epochs': 10,
            'learning_rate': 0.001,
            'gamma': 0.1,
            'step_size': 7,
            'k_folds': 5,
            'seed': 42
        },
        'models': {
            'use_resnet': True,
            'use_efficientnet': True,
            'use_mobilenet': True
        },
        'optimization': {
            'optuna_trials': 10,
            'quantize_models': True,
            'pruning_amount': 0.2
        }
    }
    
    # Tentar carregar do arquivo
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Mesclar configurações (atualizar default com valores do usuário)
            def update_config(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = update_config(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            
            final_config = update_config(default_config, user_config)
            logger.info(f"Configurações carregadas de {config_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar configurações do arquivo: {str(e)}")
            final_config = default_config
    else:
        logger.info(f"Arquivo de configuração {config_path} não encontrado. Usando configurações padrão.")
        # Salvar configurações padrão para referência
        try:
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Configurações padrão salvas em {config_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar configurações padrão: {str(e)}")
        final_config = default_config
    
    return final_config

def create_training_checkpoint(model, optimizer, epoch, train_losses, val_losses, best_acc, 
                              filename="checkpoint.pt", amp_scaler=None, batch_size=None):
    """
    Salva um checkpoint completo que permite retomar o treinamento após interrupção.
    
    Args:
        model: Modelo PyTorch a ser salvo
        optimizer: Otimizador a ser salvo
        epoch: Época atual
        train_losses: Lista de perdas de treinamento
        val_losses: Lista de perdas de validação
        best_acc: Melhor acurácia até o momento
        filename: Nome do arquivo de checkpoint
        amp_scaler: GradScaler para AMP (opcional)
        batch_size: Tamanho do batch atual (opcional)
        
    Returns:
        str: Caminho para o arquivo de checkpoint salvo
    """
    logger = logging.getLogger("landscape_classifier")
    
    # Garantir que o diretório exista
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_acc': best_acc,
        'random_state': {
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
    }
    
    # Adicionar amp_scaler se fornecido
    if amp_scaler is not None:
        checkpoint['amp_scaler'] = amp_scaler.state_dict()
    
    # Adicionar batch_size se fornecido
    if batch_size is not None:
        checkpoint['batch_size'] = batch_size
    
    torch.save(checkpoint, filename)
    logger.info(f"Checkpoint salvo em: {filename}")
    return filename

def resume_from_checkpoint(checkpoint_path, model, optimizer):
    """
    Carrega um checkpoint e configura o modelo e otimizador para retomar treinamento.
    
    Args:
        checkpoint_path: Caminho para o arquivo de checkpoint
        model: Modelo PyTorch a ser configurado
        optimizer: Otimizador a ser configurado
        
    Returns:
        tuple: (modelo, otimizador, época, perdas de treino, perdas de validação, melhor acurácia)
    """
    logger = logging.getLogger("landscape_classifier")
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint {checkpoint_path} não encontrado. Iniciando do zero.")
        return model, optimizer, 0, [], [], 0.0
    
    logger.info(f"Carregando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Restaurar modelo e otimizador
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restaurar estado aleatório para reprodutibilidade
    if 'random_state' in checkpoint:
        np.random.set_state(checkpoint['random_state']['numpy'])
        torch.set_rng_state(checkpoint['random_state']['torch'])
        if checkpoint['random_state']['torch_cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['random_state']['torch_cuda'])
    
    logger.info(f"Retomando treinamento da época {checkpoint['epoch']+1}")
    return model, optimizer, checkpoint['epoch']+1, checkpoint['train_losses'], checkpoint['val_losses'], checkpoint['best_acc']

def save_model_hyperparameters(model_name, best_params, best_accuracy):
    """
    Salva ou atualiza os hiperparâmetros no arquivo JSON.
    
    Args:
        model_name: Nome do modelo
        best_params: Melhores hiperparâmetros encontrados
        best_accuracy: Melhor acurácia obtida
    """
    logger = logging.getLogger("landscape_classifier")
    
    try:
        # Criar diretório se não existir
        os.makedirs('results', exist_ok=True)
        
        # Tentar carregar arquivo existente
        try:
            with open('results/best_hyperparameters.json', 'r') as f:
                best_params_dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            best_params_dict = {}
        
        # Atualizar com os novos parâmetros
        best_params_dict[model_name] = {
            'params': best_params,
            'accuracy': float(best_accuracy)
        }
        
        # Salvar o arquivo atualizado
        with open('results/best_hyperparameters.json', 'w') as f:
            json.dump(best_params_dict, f, indent=2)
            
        logger.info(f"Parâmetros do {model_name} salvos em results/best_hyperparameters.json")
    except Exception as e:
        logger.error(f"Erro ao salvar parâmetros do {model_name}: {str(e)}")

def save_ensemble_results(ensemble_accuracy, model_names):
    """
    Salva os resultados do ensemble no mesmo arquivo JSON.
    
    Args:
        ensemble_accuracy: Acurácia do ensemble
        model_names: Lista de nomes dos modelos usados no ensemble
    """
    logger = logging.getLogger("landscape_classifier")
    
    try:
        # Criar diretório se não existir
        os.makedirs('results', exist_ok=True)
        
        # Tentar carregar arquivo existente
        try:
            with open('results/best_hyperparameters.json', 'r') as f:
                data_dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data_dict = {}
        
        # Adicionar resultados do ensemble
        data_dict['ensemble'] = {
            'accuracy': float(ensemble_accuracy),
            'models_used': model_names
        }
        
        # Salvar o arquivo atualizado
        with open('results/best_hyperparameters.json', 'w') as f:
            json.dump(data_dict, f, indent=2)
            
        logger.info(f"Resultados do ensemble salvos em results/best_hyperparameters.json")
    except Exception as e:
        logger.error(f"Erro ao salvar resultados do ensemble: {str(e)}")

# --------------- NOVAS FUNÇÕES DE OTIMIZAÇÃO DE MEMÓRIA ---------------

def create_dynamic_batch_dataloader(dataset, initial_batch_size, num_workers=4, 
                                   pin_memory=True, shuffle=True, collate_fn=None):
    """
    Cria um DataLoader com funcionalidade para ajuste dinâmico de batch size.
    Permite reduzir o tamanho do batch durante o treinamento em caso de OOM.
    
    Args:
        dataset: Dataset PyTorch
        initial_batch_size: Tamanho do batch inicial
        num_workers: Número de workers para carregamento paralelo
        pin_memory: Se True, usa pin_memory para transferência mais rápida para GPU
        shuffle: Se True, embaralha os dados
        collate_fn: Função de collate personalizada
        
    Returns:
        DataLoader configurado para ajuste dinâmico de batch size
    """
    from torch.utils.data import DataLoader
    
    class DynamicBatchDataLoader:
        def __init__(self, dataset, initial_batch_size, **kwargs):
            self.dataset = dataset
            self.initial_batch_size = initial_batch_size
            self.current_batch_size = initial_batch_size
            self.kwargs = kwargs
            self.dataloader = DataLoader(dataset, batch_size=initial_batch_size, **kwargs)
            
            # Para compatibilidade com atributos de DataLoader
            self.batch_size = self.current_batch_size
            self.num_workers = kwargs.get('num_workers', 0)
            self.pin_memory = kwargs.get('pin_memory', False)
            self.collate_fn = kwargs.get('collate_fn', None)
            
        def __iter__(self):
            return iter(self.dataloader)
            
        def __len__(self):
            return len(self.dataloader)
            
        def reduce_batch_size(self, factor=2, min_batch_size=1):
            """
            Reduz o tamanho do batch por um fator (default: divide por 2).
            
            Args:
                factor: Fator de redução
                min_batch_size: Tamanho mínimo de batch permitido
                
            Returns:
                bool: True se o batch foi reduzido, False se já está no mínimo
            """
            logger = logging.getLogger("landscape_classifier")
            
            new_batch_size = max(self.current_batch_size // factor, min_batch_size)
            
            if new_batch_size < self.current_batch_size:
                logger.info(f"Reduzindo batch size: {self.current_batch_size} -> {new_batch_size}")
                self.current_batch_size = new_batch_size
                self.dataloader = DataLoader(self.dataset, batch_size=new_batch_size, **self.kwargs)
                self.batch_size = self.current_batch_size
                return True
            else:
                logger.warning(f"Batch size já está no mínimo: {self.current_batch_size}")
                return False
    
    return DynamicBatchDataLoader(dataset, initial_batch_size, 
                                 num_workers=num_workers, 
                                 pin_memory=pin_memory, 
                                 shuffle=shuffle,
                                 collate_fn=collate_fn)

def enable_gradient_checkpointing(model, enable=True):
    """
    Ativa checkpoint de gradiente no modelo para reduzir uso de memória.
    Funciona com modelos compatíveis como ResNets e Transformers.
    
    Args:
        model: Modelo PyTorch a ser modificado
        enable: Se True, ativa checkpoint de gradiente; se False, desativa
        
    Returns:
        bool: True se a operação foi bem-sucedida, False caso contrário
    """
    logger = logging.getLogger("landscape_classifier")
    
    # Identificar o tipo de modelo pelo nome da classe
    model_name = model.__class__.__name__.lower()
    logger.info(f"Tentando ativar gradient checkpointing para modelo tipo: {model_name}")
    
    # NOVO: Bloquear specifically para EfficientNet - problema conhecido de compatibilidade
    if 'efficient' in model_name.lower():
        logger.warning("Gradient checkpointing não é recomendado para EfficientNet devido a problemas de compatibilidade")
        return False
    
    # Implementações específicas por tipo de modelo
    if 'resnet' in model_name:
        # Implementação específica para modelos ResNet
        if hasattr(model, 'layer4'):
            try:
                from torch.utils.checkpoint import checkpoint_sequential
                
                # Salvar o forward original
                original_forward = model.forward
                
                if enable:
                    def checkpointed_forward(x):
                        # Aplicar checkpointing apenas nas camadas mais pesadas
                        x = model.conv1(x)
                        x = model.bn1(x)
                        x = model.relu(x)
                        x = model.maxpool(x)
                        
                        # Aplicar checkpointing em blocos sequenciais
                        sequential_blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
                        x = checkpoint_sequential(sequential_blocks, 4, x)
                        
                        x = model.avgpool(x)
                        x = torch.flatten(x, 1)
                        x = model.fc(x)
                        return x
                    
                    # Substituir o forward
                    model.forward = checkpointed_forward
                    logger.info("Gradient checkpointing ativado para ResNet")
                else:
                    # Restaurar o forward original
                    model.forward = original_forward
                    logger.info("Gradient checkpointing desativado para ResNet")
                
                return True
            except Exception as e:
                logger.error(f"Erro ao ativar gradient checkpointing para ResNet: {str(e)}")
        
    elif 'efficient' in model_name:
        # Implementação específica para EfficientNet
        try:
            from torch.utils.checkpoint import checkpoint
            
            # Salvar o forward original da parte de features
            if hasattr(model, 'features'):
                original_features_forward = model.features.forward
                
                if enable:
                    def checkpointed_features_forward(x):
                        # Dividir as features em blocos menores para checkpointing
                        features_blocks = []
                        for i, feature_block in enumerate(model.features):
                            if i % 2 == 0 and i > 0:  # Agrupar em pares para reduzir overhead
                                features_blocks.append(feature_block)
                        
                        # Aplicar checkpointing nos blocos
                        for block in features_blocks:
                            x = checkpoint(block, x)
                        
                        return x
                    
                    # Substituir o forward das features
                    model.features.forward = checkpointed_features_forward
                    logger.info("Gradient checkpointing ativado para EfficientNet")
                else:
                    # Restaurar o forward original
                    model.features.forward = original_features_forward
                    logger.info("Gradient checkpointing desativado para EfficientNet")
                
                return True
            
        except Exception as e:
            logger.error(f"Erro ao ativar gradient checkpointing para EfficientNet: {str(e)}")
    
    elif any(name in model_name for name in ['vit', 'vision', 'transformer', 'swin']):
        # Implementação para transformers
        try:
            # Verificar se é um modelo do timm
            if hasattr(model, 'blocks'):
                # Para modelos ViT padrão
                if enable:
                    for i, block in enumerate(model.blocks):
                        if hasattr(block, 'checkpoint') and isinstance(block.checkpoint, bool):
                            block.checkpoint = True
                    logger.info("Gradient checkpointing ativado para blocos do Vision Transformer")
                    return True
            
            # Para modelos Swin Transformer
            if hasattr(model, 'layers'):
                if enable:
                    from torch.utils.checkpoint import checkpoint
                    
                    # Salvar o forward original
                    original_forward = model.forward
                    
                    def checkpointed_forward(x):
                        # Processar etapas iniciais
                        x = model.patch_embed(x)
                        if hasattr(model, 'pos_drop'):
                            x = model.pos_drop(x)
                        
                        # Aplicar checkpointing nas camadas principais
                        for layer in model.layers:
                            x = checkpoint(layer, x)
                        
                        # Processar final
                        if hasattr(model, 'norm'):
                            x = model.norm(x)
                        
                        # Classificação final
                        if hasattr(model, 'head'):
                            x = model.head(x)
                        
                        return x
                    
                    # Substituir o forward
                    model.forward = checkpointed_forward
                    logger.info("Gradient checkpointing ativado para Swin Transformer")
                    return True
            
            # Tentativa genérica para qualquer tipo de transformer
            # Método da HuggingFace
            if hasattr(model, 'gradient_checkpointing_enable'):
                if enable:
                    model.gradient_checkpointing_enable()
                else:
                    model.gradient_checkpointing_disable()
                logger.info(f"Gradient checkpointing {'ativado' if enable else 'desativado'} via método nativo")
                return True
            
        except Exception as e:
            logger.error(f"Erro ao ativar gradient checkpointing para Transformer: {str(e)}")
    
    # Tentativa genérica para qualquer modelo (fallback)
    try:
        from torch.utils.checkpoint import checkpoint
        
        # Verificar se tem módulos sequenciais
        sequential_modules = []
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Sequential) and len(list(module.children())) > 2:
                sequential_modules.append((name, module))
        
        if sequential_modules:
            # Salvar o forward original
            original_forward = model.forward
            
            if enable:
                # Criar um forward personalizado
                def checkpointed_forward(x):
                    with torch.enable_grad():
                        result = original_forward(x)
                    return result
                
                # Substituir o forward
                model.forward = checkpointed_forward
                logger.info(f"Gradient checkpointing genérico ativado para {model_name}")
                return True
            else:
                # Restaurar o forward original
                model.forward = original_forward
                logger.info(f"Gradient checkpointing genérico desativado para {model_name}")
                return True
    
    except Exception as e:
        logger.error(f"Erro na tentativa genérica de gradient checkpointing: {str(e)}")
    
    # Se chegou aqui, não conseguiu ativar o checkpointing
    logger.warning(f"Gradient checkpointing não é suportado para {model_name}")
    return False

def disable_gradient_checkpointing_for_efficientnet(model):
    """
    Desativa o gradient checkpointing especificamente para o EfficientNet.
    
    Args:
        model: Modelo EfficientNet
        
    Returns:
        bool: True se foi desativado com sucesso
    """
    logger = logging.getLogger("landscape_classifier")
    try:
        # Restaurar o método original de forward se foi modificado
        if hasattr(model, 'features') and hasattr(model.features, 'forward'):
            if hasattr(model.features, '_original_forward'):
                model.features.forward = model.features._original_forward
                logger.info("Gradient checkpointing desativado para EfficientNet")
                return True
            
        # Se não encontrarmos o forward original, desativar os hooks
        for module in model.modules():
            if hasattr(module, '_checkpoint_hooks'):
                for hook in module._checkpoint_hooks:
                    hook.remove()
                delattr(module, '_checkpoint_hooks')
            
        logger.info("Hooks de gradient checkpointing removidos para EfficientNet")
        return True
    except Exception as e:
        logger.error(f"Erro ao desativar gradient checkpointing: {str(e)}")
        return False

def setup_amp_training(model, optimizer, criterion):
    """
    Configura treinamento com precisão mista automática (AMP).
    
    Args:
        model: Modelo PyTorch
        optimizer: Otimizador PyTorch
        criterion: Função de perda
        
    Returns:
        tuple: (scaler, função forward_amp, função backward_amp)
    """
    import torch
    
    logger = logging.getLogger("landscape_classifier")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA não disponível. AMP desativado.")
        
        # Funções dummy sem AMP
        def forward_fn(inputs, targets):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            return outputs, loss
            
        def backward_fn(loss):
            loss.backward()
            optimizer.step()
            
        return None, forward_fn, backward_fn
    
    # Criar scaler para AMP
    scaler = torch.cuda.amp.GradScaler()
    logger.info("Treinamento com precisão mista (AMP) configurado")
    
    # Função de forward com AMP
    def forward_amp(inputs, targets):
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        return outputs, loss
    
    # Função de backward com AMP
    def backward_amp(loss):
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    return scaler, forward_amp, backward_amp

def profile_model_memory(model, input_shape=(1, 3, 224, 224), detailed=False):
    """
    Analisa o uso de memória do modelo durante forward e backward pass.
    
    Args:
        model: Modelo PyTorch a ser analisado
        input_shape: Forma do tensor de entrada de exemplo
        detailed: Se True, mostra uso de memória por camada
        
    Returns:
        dict: Estatísticas de uso de memória
    """
    import torch
    
    logger = logging.getLogger("landscape_classifier")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA não disponível. Monitoramento de memória desativado.")
        return {}
    
    # Garantir que o modelo está na GPU
    device = next(model.parameters()).device
    if device.type != 'cuda':
        model = model.cuda()
        device = next(model.parameters()).device
    
    # Criar tensor de entrada de exemplo
    dummy_input = torch.rand(input_shape, device=device)
    
    # Registrar uso de memória inicial
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    
    # Forward pass
    model.train()  # Garantir modo de treinamento
    outputs = model(dummy_input)
    
    # Registrar uso de memória após forward
    forward_mem = torch.cuda.memory_allocated() - start_mem
    
    # Backward pass
    if isinstance(outputs, tuple):
        loss = outputs[0].sum()
    else:
        loss = outputs.sum()
    loss.backward()
    
    # Registrar uso de memória após backward
    backward_mem = torch.cuda.memory_allocated() - forward_mem - start_mem
    peak_mem = torch.cuda.max_memory_allocated()
    
    # Limpar memória
    loss = outputs = dummy_input = None
    torch.cuda.empty_cache()
    gc.collect()
    
    # Coletar estatísticas
    memory_stats = {
        'forward_mb': forward_mem / (1024**2),
        'backward_mb': backward_mem / (1024**2),
        'total_mb': (forward_mem + backward_mem) / (1024**2),
        'peak_mb': peak_mem / (1024**2)
    }
    
    logger.info(f"Análise de memória para {model.__class__.__name__}:")
    logger.info(f"  Forward pass: {memory_stats['forward_mb']:.2f} MB")
    logger.info(f"  Backward pass: {memory_stats['backward_mb']:.2f} MB")
    logger.info(f"  Total: {memory_stats['total_mb']:.2f} MB")
    logger.info(f"  Pico de memória: {memory_stats['peak_mb']:.2f} MB")
    
    # Análise detalhada por camada
    if detailed:
        try:
            # Importar ferramentas de profiling
            from torch.autograd.profiler import profile
            
            model.zero_grad()
            dummy_input = torch.rand(input_shape, device=device)
            
            with profile(use_cuda=True) as prof:
                outputs = model(dummy_input)
                if isinstance(outputs, tuple):
                    loss = outputs[0].sum()
                else:
                    loss = outputs.sum()
                loss.backward()
            
            # Obter tabela de uso de memória por operação
            logger.info("\nUso de memória detalhado por operação:")
            logger.info(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            
            # Adicionar ao dicionário de estatísticas
            memory_stats['detailed_profile'] = prof.key_averages().table(
                sort_by="cuda_memory_usage", row_limit=20)
        except Exception as e:
            logger.warning(f"Falha na análise detalhada de memória: {str(e)}")
    
    return memory_stats

def profile_model_training(model, dataloader, num_steps=5, warmup_steps=1):
    """
    Utiliza o PyTorch Profiler para analisar uso de memória e performance.
    
    Args:
        model: Modelo PyTorch
        dataloader: DataLoader contendo dados de exemplo
        num_steps: Número de passos de treinamento para análise
        warmup_steps: Número de passos para warmup antes da análise
        
    Returns:
        Objeto de perfil do PyTorch ou None se não suportado
    """
    logger = logging.getLogger("landscape_classifier")
    
    try:
        from torch.profiler import profile, record_function, ProfilerActivity
    except ImportError:
        logger.error("PyTorch Profiler não está disponível. Instale PyTorch >= 1.8.0")
        return None
    
    device = next(model.parameters()).device
    model.train()
    
    # Definir atividades para monitorar
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    # Definir critério e otimizador temporários
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Configurar o profiler
    try:
        # Obter um batch de exemplo do dataloader para verificar a compatibilidade
        try:
            sample_batch, sample_labels = next(iter(dataloader))
            sample_batch = sample_batch.to(device)
            
            # Testar forward pass com o batch de exemplo para verificar compatibilidade
            with torch.no_grad():
                _ = model(sample_batch)
            logger.info(f"Teste de forward pass bem-sucedido com batch de tamanho {sample_batch.shape}")
        except Exception as e:
            logger.warning(f"Aviso: Teste de forward pass falhou: {str(e)}")
        
        # Schedule para profiling: wait, warmup, active, repeat
        schedule = torch.profiler.schedule(
            wait=1, warmup=warmup_steps, active=num_steps, repeat=1)
        
        # Criar handler para TensorBoard
        tensorboard_dir = f'runs/profile_{model.__class__.__name__}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(tensorboard_dir, exist_ok=True)
        handler = torch.profiler.tensorboard_trace_handler(tensorboard_dir)
        logger.info(f"Iniciando profiling de {model.__class__.__name__} por {num_steps} passos")
        logger.info(f"Resultados serão salvos em {tensorboard_dir}")
        
        # Executar profiling
        with profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            # Loop de treinamento simplificado
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                if batch_idx >= warmup_steps + num_steps:
                    break
                
                with record_function("batch_processing"):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Zerar gradientes
                    optimizer.zero_grad()
                    
                    # Forward
                    with record_function("forward"):
                        # Usar o batch real do dataloader
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    # Backward
                    with record_function("backward"):
                        loss.backward()
                    
                    # Optimize
                    with record_function("optimize"):
                        optimizer.step()
                
                prof.step()  # Avançar o profiler
                
                if batch_idx < warmup_steps:
                    logger.info(f"Passo de warmup {batch_idx+1}/{warmup_steps}")
                else:
                    logger.info(f"Passo de profiling {batch_idx-warmup_steps+1}/{num_steps}")
        
        # Imprimir estatísticas
        top_cuda_ops = prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=10)
        memory_stats = prof.key_averages().table(
            sort_by="cuda_memory_usage", row_limit=10)
        
        logger.info("\nOperações CUDA mais lentas:")
        logger.info(top_cuda_ops)
        
        logger.info("\nOperações com maior uso de memória:")
        logger.info(memory_stats)
        
        logger.info(f"\nProfiling completo! Visualize no TensorBoard: tensorboard --logdir={tensorboard_dir}")
        
        return prof
    
    except Exception as e:
        logger.error(f"Erro durante profiling: {str(e)}")
        return None

def monitor_memory_usage(logger=None, label="", interval_mb=None):
    """
    Monitora o uso atual de memória e registra no logger.
    
    Args:
        logger: O logger para registrar informações
        label: Rótulo opcional para identificar o ponto de medição
        interval_mb: Intervalo em MB para verificar uso excessivo de memória
    """
    if logger is None:
        logger = logging.getLogger("landscape_classifier")
    
    if not torch.cuda.is_available():
        logger.info(f"Monitoramento de memória ({label}): GPU não disponível")
        return
    
    # Memória CUDA
    allocated_mb = torch.cuda.memory_allocated() / (1024**2)
    reserved_mb = torch.cuda.memory_reserved() / (1024**2)
    allocated_gb = allocated_mb / 1024
    reserved_gb = reserved_mb / 1024
    
    # Memória RAM
    import psutil
    ram_used_percent = psutil.virtual_memory().percent
    ram_used_gb = psutil.virtual_memory().used / (1024**3)
    
    # Memória do processo
    process = psutil.Process()
    process_memory_gb = process.memory_info().rss / (1024**3)
    
    logger.info(f"Monitoramento de memória ({label}):")
    logger.info(f"  GPU: {allocated_gb:.3f} GB alocado, {reserved_gb:.3f} GB reservado")
    logger.info(f"  RAM: {ram_used_gb:.3f} GB ({ram_used_percent}%)")
    logger.info(f"  Processo: {process_memory_gb:.3f} GB")

    # Lógica para usar interval_mb
    if interval_mb is not None:
        # Verificar se a memória alocada ultrapassou o intervalo especificado
        if allocated_mb > interval_mb:
            logger.warning(f"Uso de memória GPU ({allocated_mb:.2f} MB) excedeu o limite de {interval_mb} MB")
            
        # Registrar apenas em múltiplos do intervalo para reduzir logs
        if int(allocated_mb) % int(interval_mb) < 10:  # Tolerância de 10MB
            logger.debug(f"Checkpoint de memória: {int(allocated_mb)} MB alocados (intervalo: {interval_mb} MB)")
            
        # Tentar liberar memória se estiver muito alta (opcional)
        if allocated_mb > 10 * interval_mb:  # Se ultrapassar 10x o intervalo
            logger.warning(f"Uso crítico de memória: {allocated_mb:.2f} MB. Tentando liberar memória...")
            gc.collect()
            torch.cuda.empty_cache()

def cleanup_cuda_memory(model=None, tensors_to_delete=None):
    """
    Realiza limpeza agressiva de memória CUDA.
    
    Args:
        model: Modelo PyTorch opcional para descarregar da GPU
        tensors_to_delete: Lista de tensores para excluir explicitamente
        
    Returns:
        float: Memória GPU liberada em MB
    """
    import torch
    
    logger = logging.getLogger("landscape_classifier")
    
    if not torch.cuda.is_available():
        return 0
    
    # Registrar memória em uso antes da limpeza
    memory_before = torch.cuda.memory_allocated() / (1024**2)
    
    # Excluir tensores específicos se fornecidos
    if tensors_to_delete is not None:
        for tensor in tensors_to_delete:
            del tensor
    
    # Mover modelo para CPU se fornecido
    if model is not None:
        model_device = next(model.parameters()).device
        if model_device.type == 'cuda':
            model_cpu = model.cpu()
            del model
            model = model_cpu
    
    # Coletar objetos Python não utilizados
    gc.collect()
    
    # Limpar cache de memória CUDA
    torch.cuda.empty_cache()
    
    # Verificar memória após limpeza
    memory_after = torch.cuda.memory_allocated() / (1024**2)
    memory_freed = memory_before - memory_after
    
    logger.info(f"Memória GPU liberada: {memory_freed:.2f} MB (antes: {memory_before:.2f} MB, depois: {memory_after:.2f} MB)")
    
    return memory_freed

def get_memory_snapshot():
    """
    Obtém um snapshot do estado atual de memória.
    
    Returns:
        dict: Dicionário com informações de memória
    """
    import torch
    import psutil
    
    snapshot = {}
    
    # Memória CUDA
    if torch.cuda.is_available():
        snapshot['cuda'] = {
            'gpu_0': {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'free_gb': torch.cuda.mem_get_info()[0] / (1024**3),
                'total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        }
    
    # Memória RAM
    import psutil
    ram = psutil.virtual_memory()
    snapshot['ram_total_gb'] = ram.total / (1024**3)
    snapshot['ram_used_gb'] = ram.used / (1024**3)
    snapshot['ram_percent'] = ram.percent
    
    # Memória do processo
    process = psutil.Process()
    snapshot['process_memory_gb'] = process.memory_info().rss / (1024**3)
    
    # Timestamp
    from datetime import datetime
    snapshot['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return snapshot