"""
Módulo para criação e configuração de modelos no projeto de classificação de paisagens.

Este módulo contém funções para criar e configurar diferentes arquiteturas de 
modelos (ResNet, EfficientNet, Vision Transformer, MobileNet) e otimizadores.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import timm
import logging
import numpy as np
from torch.utils.data import DataLoader
import time
from torch.utils.tensorboard import SummaryWriter

# Importar funções do módulo training para usar no safe_mobilenet_training
# Esta importação será feita após a definição das funções relevantes no módulo training.py
# from training import train_model, evaluate_model

# Obter logger
logger = logging.getLogger("landscape_classifier")

def enable_gradient_checkpointing(model, enable=True):
    """
    Ativa checkpoint de gradiente no modelo para reduzir uso de memória.
    Funciona com modelos compatíveis como ResNets e Transformers.
    
    Args:
        model: Modelo PyTorch a ser configurado
        enable: Se True, ativa checkpoint de gradiente; se False, desativa
        
    Returns:
        bool: True se o checkpoint de gradiente foi aplicado com sucesso, False caso contrário
    """
    try:
        if hasattr(model, 'gradient_checkpointing_enable'):
            # Para modelos HuggingFace/Transformers
            if enable:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            logger.info(f"Checkpoint de gradiente {'ativado' if enable else 'desativado'} usando método do HuggingFace")
            return True
        
        # Para torchvision ResNets e outros modelos compatíveis
        checkpoint_enabled = False
        for module in model.modules():
            if hasattr(module, 'checkpoint') and callable(getattr(module, 'checkpoint', None)):
                module.checkpoint = enable
                checkpoint_enabled = True
                logger.info(f"Checkpoint de gradiente {'ativado' if enable else 'desativado'} para módulo {type(module).__name__}")
        
        if checkpoint_enabled:
            return True
        
        # Para implementações personalizadas
        if hasattr(model, 'apply_gradient_checkpointing'):
            model.apply_gradient_checkpointing()
            logger.info("Checkpoint de gradiente ativado usando método personalizado")
            return True
        
        # Para modelos timm
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(enable=enable)
            logger.info(f"Checkpoint de gradiente {'ativado' if enable else 'desativado'} usando método do timm")
            return True
            
        logger.warning(f"Não foi possível aplicar checkpoint de gradiente para o modelo {type(model).__name__}")
        return False
    except Exception as e:
        logger.error(f"Erro ao configurar checkpoint de gradiente: {str(e)}", exc_info=True)
        return False

def create_model_with_best_params(best_params, model_name='resnet50'):
    """
    Cria um modelo usando os melhores hiperparâmetros encontrados pelo Optuna
    
    Args:
        best_params: Dicionário com hiperparâmetros
        model_name: Nome do modelo ('resnet50', 'efficientnet', 'mobilenet')
        
    Returns:
        torch.nn.Module: Modelo configurado com os parâmetros otimizados
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_name == 'resnet50':
            # Código original para ResNet50...
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
            # Determinar número de camadas congeladas
            freeze_layers = best_params.get('freeze_layers', 0)
            logger.info(f"ResNet50 com {freeze_layers} camadas congeladas")
            
            if freeze_layers > 0:
                layers_to_freeze = list(model.children())[:freeze_layers]
                for layer in layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
            
            # Modificar a camada final
            num_features = model.fc.in_features
            dropout_rate = best_params.get('dropout_rate', 0.0)
            fc_layers = []
            
            # Número de camadas fully connected
            n_layers = best_params.get('n_layers', 1)
            
            prev_size = num_features
            for i in range(n_layers - 1):
                fc_size = best_params.get(f'fc_size_{i}', 512)
                fc_layers.append(nn.Linear(prev_size, fc_size))
                fc_layers.append(nn.ReLU())
                if dropout_rate > 0:
                    fc_layers.append(nn.Dropout(dropout_rate))
                prev_size = fc_size
            
            fc_layers.append(nn.Linear(prev_size, 6))  # 6 classes
            model.fc = nn.Sequential(*fc_layers)
        
        elif model_name == 'efficientnet':
            # CORREÇÃO PARA O EFFICIENTNET
            logger.info("Criando EfficientNet com correção para compatibilidade de canais de entrada")

            # Usar diretamente os pesos pré-treinados em vez de tentar sem pesos primeiro
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            logger.info("EfficientNet criado com pesos pré-treinados")

            # Verificar e corrigir a primeira camada convolucional
            try:
                # Abordagem direta - sabemos que a primeira camada está em features[0][0]
                if hasattr(model, 'features') and len(model.features) > 0:
                    # O EfficientNet B0 tem uma estrutura específica model.features[0][0]
                    first_conv = model.features[0][0]
        
                    if isinstance(first_conv, nn.Conv2d):
                        in_channels = first_conv.in_channels
                        logger.info(f"EfficientNet: primeira camada convolucional tem {in_channels} canais de entrada")
            
                        # Se a primeira camada não estiver configurada para 3 canais, corrija
                        if in_channels != 3:
                            logger.info(f"Corrigindo primeira camada convolucional de {in_channels} para 3 canais")
                
                            # Criar uma nova camada convolucional com os mesmos parâmetros, mas 3 canais de entrada
                            new_conv = nn.Conv2d(
                                in_channels=3,
                                out_channels=first_conv.out_channels,
                                kernel_size=first_conv.kernel_size,
                                stride=first_conv.stride,
                                padding=first_conv.padding,
                                bias=False if not hasattr(first_conv, 'bias') or first_conv.bias is None else True
                            )
                
                            # Inicializar os pesos da nova camada
                            import torch.nn.init as init
                            init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                
                            # Substituir a primeira camada pela nova
                            model.features[0][0] = new_conv
                            logger.info("Primeira camada convolucional substituída com sucesso")
                    else:
                        logger.warning(f"Primeira camada não é Conv2d, é {type(first_conv).__name__}")
                else:
                    logger.warning("Não foi possível encontrar a primeira camada convolucional")
            except Exception as e:
                logger.error(f"Erro ao corrigir a primeira camada do EfficientNet: {str(e)}")
                # Em caso de erro, crie um modelo completamente novo com 3 canais de entrada
                logger.info("Criando um novo modelo EfficientNet do zero")
                model = models.efficientnet_b0(weights=None, num_classes=6)
            
            # Congelar camadas iniciais (código original)
            freeze_percent = best_params.get('freeze_percent', 0.0)
            logger.info(f"EfficientNet com {freeze_percent*100:.1f}% de parâmetros congelados")
            
            if freeze_percent > 0:
                params = list(model.parameters())
                freeze_params = int(len(params) * freeze_percent)
                for param in params[:freeze_params]:
                    param.requires_grad = False
            
            # Substituir a última camada (código original)
            dropout_rate = best_params.get('dropout_rate', 0.2)
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features=1280, out_features=6)
            )
            
            # Verificar se o modelo agora aceita entrada com 3 canais
            logger.info("Verificando configuração final do EfficientNet")
            # Encontrar a primeira camada convolucional novamente para verificar
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    logger.info(f"Camada {name}: entrada={module.in_channels}, saída={module.out_channels}")
                    break
        
        elif model_name == 'mobilenet':
            # Código original para MobileNet...
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    
            # Congelar camadas iniciais
            freeze_percent = best_params.get('freeze_percent', 0.0)
            logger.info(f"MobileNet com {freeze_percent*100:.1f}% de parâmetros congelados")
    
            if freeze_percent > 0:
                params = list(model.parameters())
                freeze_params = int(len(params) * freeze_percent)
                for param in params[:freeze_params]:
                    param.requires_grad = False
    
            # Substituir a última camada
            in_features = model.classifier[-1].in_features
            dropout_rate = best_params.get('dropout_rate', 0.2)
    
            # Modificar apenas a última camada para ser seguro
            model.classifier[-1] = nn.Linear(in_features, 6)
        
        else:
            logger.error(f"Modelo {model_name} não suportado")
            raise ValueError(f"Modelo {model_name} não suportado")

        # Tentar ativar checkpoint de gradiente se especificado nos parâmetros
        if best_params.get('use_gradient_checkpointing', False):
            checkpoint_success = enable_gradient_checkpointing(model, enable=True)
            if checkpoint_success:
                logger.info(f"Checkpoint de gradiente ativado para o modelo {model_name}")
            else:
                logger.warning(f"Não foi possível ativar checkpoint de gradiente para o modelo {model_name}")

        return model.to(device)

    except Exception as e:
        logger.error(f"Erro ao criar modelo {model_name}: {str(e)}", exc_info=True)
        raise

def create_optimizer_and_scheduler(model, best_params, train_loader=None):
    """
    Configura otimizador e scheduler com base nos melhores parâmetros
    
    Args:
        model: Modelo PyTorch
        best_params: Dicionário com hiperparâmetros
        train_loader: DataLoader de treinamento (necessário para alguns schedulers)
        
    Returns:
        tuple: (optimizer, scheduler, scheduler_name)
    """
    try:
        # Configurar otimizador
        lr = best_params.get('lr', 0.001)
        optimizer_name = best_params.get('optimizer', 'Adam')
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
            logger.info(f"Otimizador Adam configurado com lr={lr}")
        elif optimizer_name == 'SGD':
            momentum = best_params.get('momentum', 0.9)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            logger.info(f"Otimizador SGD configurado com lr={lr}, momentum={momentum}")
        else:  # AdamW
            weight_decay = best_params.get('weight_decay', 0.01)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            logger.info(f"Otimizador AdamW configurado com lr={lr}, weight_decay={weight_decay}")
        
        # Configurar scheduler
        scheduler_name = best_params.get('scheduler', 'StepLR')
        
        if scheduler_name == 'StepLR':
            step_size = best_params.get('step_size', 7)
            gamma = best_params.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            logger.info(f"Scheduler StepLR configurado com step_size={step_size}, gamma={gamma}")
        elif scheduler_name == 'CosineAnnealingLR':
            T_max = best_params.get('T_max', 10)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
            logger.info(f"Scheduler CosineAnnealingLR configurado com T_max={T_max}")
        elif scheduler_name == 'ReduceLROnPlateau':
            factor = best_params.get('factor', 0.1)
            patience = best_params.get('patience_scheduler', 3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
            logger.info(f"Scheduler ReduceLROnPlateau configurado com factor={factor}, patience={patience}")
        elif scheduler_name == 'OneCycleLR':
            if train_loader is None:
                logger.warning("Train loader não fornecido. Scheduler OneCycleLR não pode ser configurado.")
                scheduler = None
            else:
                max_lr = best_params.get('max_lr', lr*10)
                epochs = best_params.get('epochs', 10)
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs
                )
                logger.info(f"Scheduler OneCycleLR configurado com max_lr={max_lr}")
        else:
            logger.warning(f"Scheduler {scheduler_name} não reconhecido. Usando StepLR padrão.")
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            scheduler_name = 'StepLR'
        
        return optimizer, scheduler, scheduler_name
    except Exception as e:
        logger.error(f"Erro ao configurar otimizador e scheduler: {str(e)}", exc_info=True)
        raise

def create_vit_model(vit_params=None):
    """
    Cria um modelo Vision Transformer com configurações otimizadas
    
    Args:
        vit_params: Dicionário com parâmetros para o ViT
            {
                'model_name': Nome da arquitetura (ex: 'vit_base_patch16_224'),
                'pretrained': Se deve usar pesos pré-treinados,
                'dropout_rate': Taxa de dropout,
                'use_gradient_checkpointing': Se deve ativar checkpoint de gradiente
            }
    
    Returns:
        torch.nn.Module: Modelo Vision Transformer configurado
    """
    logger.info("Criando modelo Vision Transformer...")
    
    if vit_params is None:
        vit_params = {
            'model_name': 'vit_base_patch16_224',
            'pretrained': True,
            'dropout_rate': 0.1,
            'use_gradient_checkpointing': False,
        }
    
    try:
        # Obter dispositivo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Usar timm para carregar o modelo pré-treinado
        model_name = vit_params.get('model_name', 'vit_base_patch16_224')
        pretrained = vit_params.get('pretrained', True)
        
        # Carregar o modelo base
        model = timm.create_model(model_name, pretrained=pretrained)
        
        # Modificar a cabeça de classificação para o número de classes
        dropout_rate = vit_params.get('dropout_rate', 0.1)
        
        # A modificação da cabeça de classificação varia dependendo da arquitetura
        if 'vit' in model_name or 'deit' in model_name:
            num_features = model.head.in_features
            model.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 6)  # 6 classes
            )
        elif 'swin' in model_name:
            num_features = model.head.in_features
            model.head = nn.Sequential(
                nn.LayerNorm(num_features),
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 6)
            )
        elif 'pit' in model_name:
            num_features = model.head.in_features
            model.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 6)
            )
        elif 'efficientformer' in model_name:
            num_features = model.head.fc.in_features
            model.head.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 6)
            )
        else:
            # Tentar uma abordagem mais genérica
            if hasattr(model, 'head'):
                if hasattr(model.head, 'in_features'):
                    num_features = model.head.in_features
                    model.head = nn.Sequential(
                        nn.Dropout(dropout_rate),
                        nn.Linear(num_features, 6)
                    )
                elif hasattr(model.head, 'fc'):
                    num_features = model.head.fc.in_features
                    model.head.fc = nn.Linear(num_features, 6)
            elif hasattr(model, 'fc'):
                num_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(num_features, 6)
                )
            else:
                logger.error(f"Não foi possível modificar a cabeça de classificação para {model_name}.")
                raise ValueError(f"Arquitetura não suportada: {model_name}")
        
        # Ativar checkpoint de gradiente se especificado
        if vit_params.get('use_gradient_checkpointing', False):
            checkpoint_success = enable_gradient_checkpointing(model, enable=True)
            if checkpoint_success:
                logger.info(f"Checkpoint de gradiente ativado para o modelo ViT {model_name}")
            else:
                logger.warning(f"Não foi possível ativar checkpoint de gradiente para o modelo ViT {model_name}")
        
        # Contar parâmetros treináveis
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Modelo {model_name} criado com {trainable_params:,} parâmetros treináveis de {total_params:,} total")
        
        return model.to(device)
    
    except Exception as e:
        logger.error(f"Erro ao criar modelo ViT: {str(e)}", exc_info=True)
        raise

def verify_model_dimensions(model, input_shape=(1, 3, 224, 224)):
    """
    Verifica as dimensões de saída de todas as camadas de um modelo.
    Ajuda a identificar incompatibilidades antes do treinamento.
    
    Args:
        model: Modelo PyTorch a ser verificado
        input_shape: Forma do tensor de entrada para teste
    
    Returns:
        dict: Dicionário com nomes de camadas e dimensões de saída
    """
    model.eval()
    device = next(model.parameters()).device
    logger.info(f"Verificando dimensões do modelo em {device}")
    
    # Criar dicionário para armazenar as dimensões
    layer_dimensions = {}
    hooks = []
    
    # Função para registrar saída de cada camada
    def hook_fn(name):
        def hook(module, input, output):
            layer_dimensions[name] = output.shape
        return hook
    
    # Registrar hooks para cada módulo
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Apenas camadas folha
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Executar forward pass
    try:
        dummy_input = torch.randn(input_shape).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        logger.error(f"Erro durante verificação de dimensões: {str(e)}")
    finally:
        # Remover todos os hooks
        for hook in hooks:
            hook.remove()
    
    # Registrar dimensões importantes
    for name, dim in layer_dimensions.items():
        if "classifier" in name or "fc" in name or "conv" in name:
            logger.info(f"Camada {name}: {dim}")
    
    return layer_dimensions

# A função safe_mobilenet_training depende de funções do módulo training, 
# então será implementada após a definição dessas funções
def safe_mobilenet_training(train_dataset, test_dataset, mobilenet_best_params, model_type=None, 
                           num_epochs=10, use_amp=False, use_gradient_checkpointing=False):
    """
    Função segura para treinar o MobileNet com tratamento adicional de erros de shape
    
    Args:
        train_dataset: Dataset de treinamento
        test_dataset: Dataset de teste
        mobilenet_best_params: Dicionário com os melhores hiperparâmetros para o MobileNet
        model_type: Tipo do modelo (cnn, vit, etc.) - ignorado, mas adicionado por compatibilidade
        num_epochs: Número de épocas para treinar (padrão: 10)
        use_amp: Se True, usa precisão mista automática
        use_gradient_checkpointing: Se True, usa gradient checkpointing
    
    Returns:
        tuple: (modelo, perdas de treino, acurácias de treino, perdas de val, acurácias de val)
    """
    logger.info(f"Iniciando treinamento seguro do MobileNet (tipo: {model_type if model_type else 'padrão'})...")
    logger.info(f"Configurações: AMP={use_amp}, gradient_checkpointing={use_gradient_checkpointing}")
    
    try:
        # Obter dispositivo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configurar DataLoaders
        batch_size = mobilenet_best_params.get('batch_size', 32)
        mobile_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        mobile_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # CORREÇÃO: Criar modelo com abordagem mais segura
        mobile_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Registrar a estrutura do modelo para diagnóstico
        logger.info(f"MobileNet - Arquitetura completa: {mobile_model}")
        
        # Verificar estrutura original do classificador
        logger.info(f"MobileNet - Estrutura original do classificador: {mobile_model.classifier}")
        
        # Obter dimensão de entrada da última camada para diagnóstico
        last_layer = mobile_model.classifier[-1]
        in_features = last_layer.in_features
        logger.info(f"MobileNet - Dimensão de entrada da última camada: {in_features}")
        logger.info(f"MobileNet - Número de classes: 6")
        
        # Congelar camadas iniciais
        if 'freeze_percent' in mobilenet_best_params:
            freeze_percent = mobilenet_best_params.get('freeze_percent', 0.5)
            if freeze_percent > 0:
                params = list(mobile_model.parameters())
                freeze_params = int(len(params) * freeze_percent)
                for param in params[:freeze_params]:
                    param.requires_grad = False
                logger.info(f"Congelados {freeze_percent*100:.1f}% dos parâmetros do MobileNet")
        
        # CORREÇÃO: Usar uma abordagem minimalista para modificar o modelo
        # Substituir apenas a última camada linear para manter a compatibilidade
        dropout_rate = mobilenet_best_params.get('dropout_rate', 0.2)
        mobile_model.classifier[-1] = nn.Linear(in_features, 6)
        
        # Ativar checkpoint de gradiente se especificado
        if mobilenet_best_params.get('use_gradient_checkpointing', False):
            checkpoint_success = enable_gradient_checkpointing(mobile_model, enable=True)
            if checkpoint_success:
                logger.info("Checkpoint de gradiente ativado para o MobileNet")
            else:
                logger.warning("Não foi possível ativar checkpoint de gradiente para o MobileNet")
        
        logger.info(f"MobileNet - Nova estrutura do classificador: {mobile_model.classifier}")
        
        # Mover para o dispositivo apropriado
        mobile_model = mobile_model.to(device)
        logger.info("Verificando dimensões do modelo MobileNet")
        verify_model_dimensions(mobile_model)
        
        # Fazer um forward pass de teste antes do treinamento
        logger.info("Realizando teste de forward pass para verificar compatibilidade...")
        try:
            sample_batch = next(iter(mobile_train_loader))[0][:2].to(device)
            with torch.no_grad():
                test_output = mobile_model(sample_batch)
                logger.info(f"Teste de forward pass bem-sucedido: entrada {sample_batch.shape} → saída {test_output.shape}")
        except Exception as e:
            logger.error(f"Erro no teste inicial de forward pass: {str(e)}", exc_info=True)
            # Se falhar, tentar uma abordagem ainda mais simples
            logger.info("Tentando reconstruir o modelo com configuração mínima...")
            mobile_model = models.mobilenet_v3_small(weights=None, num_classes=6)
            mobile_model = mobile_model.to(device)
        
        # Configurar otimizador com taxa de aprendizado conservadora
        lr = mobilenet_best_params.get('lr', 0.0003)
        mobile_optimizer = optim.Adam(mobile_model.parameters(), lr=lr)
        
        # Usar um scheduler simples
        mobile_scheduler = optim.lr_scheduler.StepLR(mobile_optimizer, step_size=5, gamma=0.1)
        
        # Configurar parâmetros adicionais de treinamento
        criterion = nn.CrossEntropyLoss()
        use_mixup = False  # Desativado para maior segurança
        use_cutmix = False  # Desativado para maior segurança
        early_stopping = True
        patience = mobilenet_best_params.get('patience_stopping', 3)
        
        # Criar writer específico para MobileNet
        import os
        mobile_writer = SummaryWriter(log_dir=os.path.join("tensorboard_logs", "mobilenet"))
        
        # Treinar modelo com menos épocas para evitar problemas
        epochs = min(num_epochs, mobilenet_best_params.get('epochs', 10))
        logger.info(f"Iniciando treinamento do MobileNet com {epochs} épocas...")
        
        # Importar função de treinamento do módulo training
        from training import train_model, evaluate_model
        
        mobile_model, train_losses, train_accs, val_losses, val_accs = train_model(
            mobile_model, mobile_train_loader, mobile_test_loader, criterion, mobile_optimizer,
            mobile_scheduler, epochs, mobile_writer, "mobilenet", use_mixup=use_mixup,
            use_cutmix=use_cutmix, early_stopping=early_stopping, patience=patience,
            model_type=model_type if model_type else 'cnn'  # Usar o model_type fornecido ou 'cnn' como padrão
        )
        
        logger.info("Treinamento seguro do MobileNet concluído com sucesso!")
        return mobile_model, train_losses, train_accs, val_losses, val_accs
    
    except Exception as e:
        # Caso ocorra qualquer erro, tentamos uma abordagem de fallback com configuração mínima
        logger.error(f"Erro durante o treinamento do MobileNet: {str(e)}", exc_info=True)
        logger.info("Tentando treinamento com configuração de fallback mínima...")
        
        try:
            # Importar função de treinamento do módulo training
            from training import train_model, evaluate_model
            
            # Obter dispositivo
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Criar modelo limpo com configuração mínima e robusta
            fallback_model = models.mobilenet_v3_small(num_classes=6)
            
            fallback_model = fallback_model.to(device)
            
            # Otimizador simples com taxa de aprendizado conservadora
            fallback_optimizer = optim.Adam(fallback_model.parameters(), lr=0.0001)
            fallback_scheduler = optim.lr_scheduler.StepLR(fallback_optimizer, step_size=5, gamma=0.1)
            
            # DataLoaders com batch size menor
            fallback_train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
            fallback_test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
            
            # Criar writer para o modelo de fallback
            import os
            fallback_writer = SummaryWriter(log_dir=os.path.join("tensorboard_logs", "mobilenet_fallback"))
            
            logger.info("Iniciando treinamento com configuração de fallback...")
            fallback_model, train_losses, train_accs, val_losses, val_accs = train_model(
                fallback_model, fallback_train_loader, fallback_test_loader,
                nn.CrossEntropyLoss(), fallback_optimizer, fallback_scheduler,
                3, fallback_writer, "mobilenet_fallback",
                use_mixup=False, use_cutmix=False, early_stopping=True, patience=2,
                model_type=model_type if model_type else 'cnn'  # Usar o model_type fornecido ou 'cnn' como padrão
            )
            
            logger.info("Treinamento de fallback concluído!")
            return fallback_model, train_losses, train_accs, val_losses, val_accs
            
        except Exception as fallback_error:
            logger.critical(f"Erro no treinamento de fallback: {str(fallback_error)}", exc_info=True)
            # Retornar valores vazios ou nulos para não quebrar o fluxo do pipeline
            logger.warning("Retornando modelo vazio para continuar o pipeline")
            empty_model = models.mobilenet_v3_small(num_classes=6).to(device)
            return empty_model, [], [], [], []