"""
Módulo para treinar e avaliar modelos de classificação de imagens.

Este módulo contém as funções principais para treinamento de modelos, avaliação
de performance e análise de erros. Integra-se com TensorBoard para visualização
de métricas durante o treinamento.

Incorpora otimizações avançadas de memória e performance:
- Ajuste automático de batch size
- Precisão mista automática (AMP)
- Checkpoint de gradientes
- Monitoramento avançado de memória
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
import logging
import gc
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from dataset_utils import custom_collate_fn
import torchvision

# Importar decorador para verificação de GPU
from utils import gpu_memory_check, cleanup_cuda_memory, get_memory_snapshot, monitor_memory_usage

# Obter o logger configurado
logger = logging.getLogger("landscape_classifier")

class DynamicBatchSizeLoader:
    """
    Wrapper para DataLoader que suporta ajuste dinâmico de batch size.
    """
    def __init__(self, dataset, initial_batch_size, num_workers=4, 
                 pin_memory=True, shuffle=True, collate_fn=None):
        """
        Inicializa o DataLoader com suporte a batch size dinâmico.
        
        Args:
            dataset: Dataset a ser carregado
            initial_batch_size: Tamanho inicial de batch
            num_workers: Número de workers para carregamento paralelo
            pin_memory: Se True, usa pin_memory para transferência mais rápida para GPU
            shuffle: Se True, embaralha os dados a cada época
            collate_fn: Função opcional para processamento de batch
        """
        self._dataset = dataset
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        
        # Criar dataloader inicial
        self._create_dataloader()
    
    def _create_dataloader(self):
        """Cria um novo dataloader com o batch size atual."""
        self.dataloader = DataLoader(
            self._dataset,
            batch_size=self.current_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            multiprocessing_context='spawn'
        )
    
    def reduce_batch_size(self, min_batch_size=1):
        """
        Reduz o batch size para metade, mas não abaixo do mínimo especificado.
        
        Args:
            min_batch_size: Tamanho mínimo permitido para o batch
            
        Returns:
            bool: True se o batch size foi reduzido, False se já está no mínimo
        """
        if self.current_batch_size <= min_batch_size:
            return False
        
        new_batch_size = max(self.current_batch_size // 2, min_batch_size)
        if new_batch_size == self.current_batch_size:
            return False
        
        self.current_batch_size = new_batch_size
        self._create_dataloader()
        return True
    
    def __iter__(self):
        """Retorna o iterador do dataloader interno."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Retorna o tamanho do dataloader interno."""
        return len(self.dataloader)
    
    @property
    def batch_size(self):
        """Retorna o tamanho de batch atual."""
        return self.current_batch_size
    
    @property
    def dataset(self):
        """Retorna o dataset do dataloader interno."""
        return self.dataloader.dataset


def setup_amp_training(model, optimizer, criterion):
    """
    Configura treinamento com precisão mista automática (AMP).
    
    Args:
        model: Modelo PyTorch
        optimizer: Otimizador PyTorch
        criterion: Função de perda
        
    Returns:
        tuple: (scaler, forward_amp_fn, backward_amp_fn)
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA não disponível, AMP não será utilizado")
        scaler = None
        
        def forward_amp(inputs, labels):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            return outputs, loss
        
        def backward_amp(loss):
            loss.backward()
            optimizer.step()
            
        return scaler, forward_amp, backward_amp
    
    # Configurar GradScaler para AMP
    scaler = torch.cuda.amp.GradScaler()
    
    # Função forward com AMP
    def forward_amp(inputs, labels):
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        return outputs, loss
    
    # Função backward com AMP
    def backward_amp(loss):
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    return scaler, forward_amp, backward_amp


def enable_gradient_checkpointing(model, enable=True):
    """
    Ativa ou desativa checkpoint de gradiente para economizar memória.
    
    Args:
        model: Modelo PyTorch a ser modificado
        enable: Se True, ativa checkpoint de gradiente; se False, desativa
        
    Returns:
        bool: True se a operação foi bem-sucedida, False caso contrário
    """
    # Verificar se o modelo suporta gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        if enable:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        return True
    
    # Para modelos transformers da biblioteca huggingface
    elif hasattr(model, 'config') and hasattr(model.config, 'gradient_checkpointing'):
        model.config.gradient_checkpointing = enable
        # Tentar encontrar método específico para ativar
        if hasattr(model, 'gradient_checkpointing_enable') and enable:
            model.gradient_checkpointing_enable()
        elif hasattr(model, 'gradient_checkpointing_disable') and not enable:
            model.gradient_checkpointing_disable()
        return True
    
    # Para torchvision e outros modelos com módulos
    else:
        success = False
        # Tentar aplicar para módulos específicos que têm suporte
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = enable
                success = True
            elif hasattr(module, 'gradient_checkpointing_enable'):
                if enable:
                    module.gradient_checkpointing_enable()
                else:
                    module.gradient_checkpointing_disable()
                success = True
        
        return success


def create_dynamic_batch_dataloader(dataset, initial_batch_size, num_workers=4, 
                                   pin_memory=True, shuffle=True, collate_fn=None):
    """
    Cria um DataLoader com suporte a ajuste dinâmico de batch size.
    
    Args:
        dataset: Dataset a ser carregado
        initial_batch_size: Tamanho inicial de batch
        num_workers: Número de workers para carregamento paralelo
        pin_memory: Se True, usa pin_memory para transferência mais rápida para GPU
        shuffle: Se True, embaralha os dados a cada época
        collate_fn: Função opcional para processamento de batch
        
    Returns:
        DynamicBatchSizeLoader: DataLoader com suporte a batch size dinâmico
    """
    return DynamicBatchSizeLoader(
        dataset, 
        initial_batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


@gpu_memory_check(threshold_mb=1000)
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, writer, model_name="model", use_mixup=False, 
                use_cutmix=False, early_stopping=True, patience=3, alpha=0.2, model_type='cnn'):
    """
    Treina o modelo com logging completo e visualização via TensorBoard.
    
    Args:
        model: Modelo PyTorch a ser treinado
        train_loader: DataLoader para treinamento
        val_loader: DataLoader para validação
        criterion: Função de perda (ex: CrossEntropyLoss)
        optimizer: Otimizador (ex: Adam, SGD)
        scheduler: Scheduler para ajustar learning rate
        num_epochs: Número de épocas para treinar
        writer: SummaryWriter do TensorBoard
        model_name: Nome do modelo para logging
        use_mixup: Se True, aplica técnica Mixup durante treinamento
        use_cutmix: Se True, aplica técnica CutMix durante treinamento
        early_stopping: Se True, aplica early stopping
        patience: Número de épocas sem melhoria para early stopping
        alpha: Parâmetro alpha para técnicas de augmentation
        model_type: Tipo de modelo ('cnn', 'vit', 'swin', etc.)
        
    Returns:
        tuple: (modelo_treinado, train_losses, train_accs, val_losses, val_accs)
    """
    logger.info(f"Iniciando treinamento: {model_name}")
    logger.info(f"Parâmetros: mixup={use_mixup}, cutmix={use_cutmix}, early_stopping={early_stopping}, paciência={patience}")

    # Liberar memória explicitamente antes de iniciar o treinamento
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Memória GPU liberada antes do treinamento de {model_name}")
        logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        logger.info(f"Memória GPU reservada: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
    
    # Registrar número de parâmetros
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parâmetros: {trainable_params:,} treináveis de {total_params:,} total")
    
    # Escrever configuração no TensorBoard
    writer.add_text(f"{model_name}/config", 
                  f"Modelo: {model_name}\n" +
                  f"Batch Size: {train_loader.batch_size}\n" +
                  f"Otimizador: {type(optimizer).__name__}\n" +
                  f"LR inicial: {optimizer.param_groups[0]['lr']}\n" +
                  f"Técnicas: mixup={use_mixup}, cutmix={use_cutmix}, early_stopping={early_stopping}")
    
    # Visualizar modelo no TensorBoard
    try:
        example_batch = next(iter(train_loader))[0][:1].to(next(model.parameters()).device)
        writer.add_graph(model, example_batch)
        logger.info("Grafo do modelo adicionado ao TensorBoard")
    except Exception as e:
        logger.warning(f"Não foi possível adicionar o grafo do modelo ao TensorBoard: {str(e)}")
    
    # Registrar algumas imagens de exemplo
    example_images, example_labels = next(iter(train_loader))
    img_grid = torchvision.utils.make_grid(example_images[:16])
    writer.add_image(f"{model_name}/training_samples", img_grid, 0)
    
    # Prepara para capturar o melhor modelo
    device = next(model.parameters()).device
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0
    no_improve_epochs = 0
    
    # Listas para armazenar as métricas de treinamento
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Registrar hora de início
    start_time = time.time()
    
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            logger.info(f"Iniciando época {epoch+1}/{num_epochs}")
            
            # ===== Fase de treinamento =====
            model.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0
            
            # Logar pesos de algumas camadas antes do treinamento nesta época
            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    writer.add_histogram(f"{model_name}/{name}", param, epoch)
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                try:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Log de progresso
                    if batch_idx % 20 == 0:
                        logger.debug(f"Treinamento: Batch {batch_idx}/{len(train_loader)}")
                        
                        # Log de uso de memória a cada 20 batches
                        if torch.cuda.is_available():
                            logger.debug(f"Memória GPU durante treinamento: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                    
                    # Zerar os gradientes
                    optimizer.zero_grad()
                    
                    # Aplicar mixup se ativado
                    if use_mixup:
                        from data_processing import mixup_data, mixup_criterion
                        
                        # Detectar o tipo de modelo com base no nome ou no parâmetro
                        detected_model_type = model_type
                        if model_type == 'cnn':  # Se não foi especificado, tentar detectar pelo nome
                            if 'swin' in model_name.lower():
                                detected_model_type = 'swin'
                            elif 'vit' in model_name.lower() or 'deit' in model_name.lower():
                                detected_model_type = 'vit'
                        
                        # Chamar mixup_data com o tipo de modelo
                        mixup_result = mixup_data(inputs, labels, alpha, detected_model_type)
                        
                        # Forward pass
                        with torch.set_grad_enabled(True):
                            # Verificar o tipo de resultado retornado pelo mixup_data
                            if detected_model_type.lower() in ['swin', 'swin_transformer'] or len(mixup_result) == 2:
                                mixed_inputs, original_labels = mixup_result
                                outputs = model(mixed_inputs)
                                # Processar a saída para diferentes tipos de modelos
                                outputs = process_model_outputs(outputs, model_type)
                                loss = criterion(outputs, original_labels)
                                # Acurácia calculada normalmente
                                _, preds = torch.max(outputs, 1)
                                running_corrects += torch.sum(preds == original_labels.data)
                            else:  # Para outros modelos: (mixed_x, y_a, y_b, lam)
                                mixed_inputs, labels_a, labels_b, lam = mixup_result
                                outputs = model(mixed_inputs)
                                # Processar a saída para diferentes tipos de modelos
                                outputs = process_model_outputs(outputs, model_type)
                                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam, detected_model_type)
                                # Para mixup tradicional, calcular acurácia ponderada
                                _, preds = torch.max(outputs, 1)
                                corrects_a = torch.sum(preds == labels_a.data)
                                corrects_b = torch.sum(preds == labels_b.data)
                                running_corrects += lam * corrects_a + (1 - lam) * corrects_b
                    else:
                        # Forward pass sem mixup
                        with torch.set_grad_enabled(True):
                            outputs = model(inputs)
                            # Processar a saída para diferentes tipos de modelos
                            outputs = process_model_outputs(outputs, model_type)
                            
                            # Logar a forma da saída para depuração
                            if batch_idx == 0 and epoch == 0:
                                logger.info(f"Forma da saída do modelo: {outputs.shape}, Forma dos rótulos: {labels.shape}")
                            
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            running_corrects += torch.sum(preds == labels.data)
                    
                    # Backward pass + otimização
                    loss.backward()
                    optimizer.step()
                    
                    # Estatísticas
                    running_loss += loss.item() * inputs.size(0)
                    total += labels.size(0)
                    
                    # Liberar tensores no final de cada batch para economia de memória
                    del inputs, labels, outputs, loss, preds
                    if 'mixed_inputs' in locals():
                        del mixed_inputs
                    if 'original_labels' in locals():
                        del original_labels
                    if 'labels_a' in locals():
                        del labels_a, labels_b
                    
                    # Coleta de lixo periódica
                    if batch_idx % 100 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"CUDA OOM no batch {batch_idx}. Pulando batch.", exc_info=True)
                        torch.cuda.empty_cache()  # Limpar memória
                        
                        # Reduzir batch size dinamicamente se possível
                        if hasattr(train_loader, 'reduce_batch_size'):
                            new_batch_size = train_loader.reduce_batch_size(min_batch_size=1)
                            if new_batch_size:
                                logger.warning(f"Reduzindo batch size para {train_loader.batch_size}")
                                continue
                        elif hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'batch_size'):
                            new_batch_size = max(1, train_loader.batch_sampler.batch_size // 2)
                            logger.warning(f"Reduzindo batch size para {new_batch_size}")
                            # A redução de batch size dinâmica é complexa e pode exigir recriação do DataLoader
                        
                        # Limpar variáveis locais para recuperar memória
                        if 'inputs' in locals():
                            del inputs
                        if 'labels' in locals():
                            del labels
                        if 'outputs' in locals():
                            del outputs
                        if 'loss' in locals():
                            del loss
                        if 'preds' in locals():
                            del preds
                        if 'mixed_inputs' in locals():
                            del mixed_inputs
                        if 'original_labels' in locals():
                            del original_labels
                        if 'labels_a' in locals():
                            del labels_a, labels_b
                        
                        continue
                    else:
                        logger.error(f"Erro no batch {batch_idx} durante treinamento", exc_info=True)
                        raise
            
            # Calcular métricas da época
            epoch_loss = running_loss / total if total > 0 else float('inf')
            epoch_acc = running_corrects.double() / total if total > 0 else 0
            
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc.item())
            
            logger.info(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Adicionar ao TensorBoard
            writer.add_scalar(f"{model_name}/Loss/train", epoch_loss, epoch)
            writer.add_scalar(f"{model_name}/Accuracy/train", epoch_acc, epoch)
            
            # Limpar memória explicitamente após o treinamento
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.debug(f"Memória liberada após fase de treinamento, antes da validação")
                logger.debug(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            
            # ===== Fase de validação =====
            model.eval()
            running_loss = 0.0
            running_corrects = 0
            total = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(val_loader):
                    try:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        if batch_idx == 0 and epoch == 0:
                            logger.info(f"Formato de labels no primeiro batch: {labels.shape}, tipo: {labels.dtype}")
                        
                        # Forward pass
                        outputs = model(inputs)
                        # Processar a saída para diferentes tipos de modelos
                        outputs = process_model_outputs(outputs, model_type)
                        
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Estatísticas
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        total += labels.size(0)
                        
                        # Coletar para matriz de confusão
                        val_preds.extend(preds.cpu().numpy())
                        val_targets.extend(labels.cpu().numpy())
                        
                        # Liberar tensores no final de cada batch para economia de memória
                        del inputs, labels, outputs, loss, preds
                        
                    except Exception as e:
                        logger.error(f"Erro no batch {batch_idx} durante validação", exc_info=True)
            
            # Calcular métricas de validação
            epoch_loss = running_loss / total if total > 0 else float('inf')
            epoch_acc = running_corrects.double() / total if total > 0 else 0
            
            val_losses.append(epoch_loss)
            val_accs.append(epoch_acc.item())
            
            # Calcular matriz de confusão para esta época
            if len(val_targets) > 0 and len(val_preds) > 0:
                num_classes = len(set(val_targets)) if hasattr(val_loader.dataset, 'classes') else None
                if num_classes is None:
                    if hasattr(val_loader.dataset, 'dataset') and hasattr(val_loader.dataset.dataset, 'classes'):
                        num_classes = len(val_loader.dataset.dataset.classes)
                    else:
                        num_classes = max(max(val_targets), max(val_preds)) + 1
                
                conf_matrix = confusion_matrix(val_targets, val_preds, labels=range(num_classes))
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Obter nomes das classes para os labels
                if hasattr(val_loader.dataset, 'classes'):
                    class_names = val_loader.dataset.classes
                elif hasattr(val_loader.dataset, 'dataset') and hasattr(val_loader.dataset.dataset, 'classes'):
                    class_names = val_loader.dataset.dataset.classes
                else:
                    class_names = [str(i) for i in range(num_classes)]
                
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=class_names,
                           yticklabels=class_names)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'Confusion Matrix - Epoch {epoch+1}')
                writer.add_figure(f"{model_name}/confusion_matrix", fig, epoch)
                plt.close(fig)
            
            val_time = time.time() - epoch_start_time
            logger.info(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Tempo: {val_time:.1f}s")
            
            # Adicionar ao TensorBoard
            writer.add_scalar(f"{model_name}/Loss/val", epoch_loss, epoch)
            writer.add_scalar(f"{model_name}/Accuracy/val", epoch_acc, epoch)
            writer.add_scalar(f"{model_name}/Learning_rate", optimizer.param_groups[0]['lr'], epoch)
            
            # Ajustar o learning rate (de acordo com o tipo de scheduler)
            old_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                
                if old_lr != new_lr:
                    logger.info(f"Learning rate ajustado: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Salvar o melhor modelo
            if epoch_acc > best_acc:
                logger.info(f"Melhor acurácia melhorou: {best_acc:.4f} -> {epoch_acc:.4f}")
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())
                no_improve_epochs = 0
                
                # Salvar melhor modelo até agora
                try:
                    model_save_path = os.path.join("models", f"{model_name}_best.pth")
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model_wts,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': best_acc,
                        'loss': epoch_loss
                    }, model_save_path)
                    logger.info(f"Melhor modelo salvo como {model_save_path}")
                except Exception as e:
                    logger.error(f"Erro ao salvar o melhor modelo: {str(e)}")
            else:
                no_improve_epochs += 1
                logger.info(f"Sem melhoria há {no_improve_epochs} épocas. Melhor: {best_acc:.4f}")

            # Salvar checkpoint para retomada (a cada 3 épocas ou na última)
            if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
                checkpoint_path = os.path.join("models", f"{model_name}_checkpoint_e{epoch+1}.pt")
                from utils import create_training_checkpoint
                create_training_checkpoint(
                    model, optimizer, epoch, train_losses, val_losses, 
                    best_acc, filename=checkpoint_path
                )

            # Early stopping
            if early_stopping and no_improve_epochs >= patience:
                logger.info(f"Early stopping ativado após {epoch+1} épocas")
                break
            
            # Limpar memória no final de cada época
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.debug(f"Memória liberada após época {epoch+1}")
                logger.debug(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        
        # Tempo total de treinamento
        total_time = time.time() - start_time
        logger.info(f"Treinamento concluído em {total_time/60:.2f} minutos")
        logger.info(f"Melhor acurácia de validação: {best_acc:.4f}")
        
        # Adicionar resumo ao TensorBoard
        writer.add_hparams(
            {"model": model_name, "batch_size": train_loader.batch_size, "epochs": epoch+1, 
             "mixup": use_mixup, "early_stopping": early_stopping},
            {"hparam/accuracy": best_acc}
        )
        
        # Carregar o melhor modelo
        model.load_state_dict(best_model_wts)
        
        # Limpar memória explicitamente após o treinamento completo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Memória liberada após treinamento completo")
            logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        
        return model, train_losses, train_accs, val_losses, val_accs
    
    except Exception as e:
        logger.critical(f"Erro crítico durante o treinamento de {model_name}", exc_info=True)
        # Tentar salvar o modelo parcial para recuperação
        try:
            recovery_path = os.path.join("models", f"{model_name}_recovery.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses, 'val_losses': val_losses
            }, recovery_path)
            logger.info(f"Estado de recuperação salvo como {recovery_path}")
        except:
            logger.critical("Falha ao salvar o estado de recuperação")
        
        # Limpar memória explicitamente após o erro
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Memória GPU liberada após erro")
        
        # Re-lançar a exceção para ser tratada pelo chamador
        raise


def train_model_optimized(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                         num_epochs, writer, model_name="model", use_mixup=False, 
                         use_cutmix=False, early_stopping=True, patience=3, alpha=0.2, 
                         model_type='cnn', use_amp=True, use_gradient_checkpointing=True, 
                         dynamic_batch=True, min_batch_size=4):
    """
    Versão otimizada da função train_model que incorpora todas as otimizações de memória.
    
    Args adicionais:
        use_amp: Se True, usa precisão mista automática para treinamento
        use_gradient_checkpointing: Se True, ativa checkpoint de gradiente para economizar memória
        dynamic_batch: Se True, permite redução dinâmica do batch size em caso de OOM
        min_batch_size: Tamanho mínimo de batch permitido para redução dinâmica
    
    Returns:
        tuple: (modelo_treinado, train_losses, train_accs, val_losses, val_accs)
    """
    logger.info(f"Iniciando treinamento otimizado: {model_name}")
    logger.info(f"Configurações: mixup={use_mixup}, cutmix={use_cutmix}, early_stopping={early_stopping}, AMP={use_amp}, Gradient checkpointing={use_gradient_checkpointing}")
    
    # Snapshot de memória inicial
    memory_snapshot = get_memory_snapshot()
    logger.info(f"Memória inicial: GPU={memory_snapshot.get('cuda', {}).get('gpu_0', {}).get('allocated_gb', 0):.2f} GB, RAM={memory_snapshot['process_memory_gb']:.2f} GB")
    
    # Liberar memória antes de iniciar
    cleanup_cuda_memory()
    
    # Ativar gradient checkpointing se solicitado
    if use_gradient_checkpointing:
        checkpoint_enabled = enable_gradient_checkpointing(model, enable=True)
        if checkpoint_enabled:
            logger.info(f"Gradient checkpointing ativado para {model_name}")
        else:
            logger.info(f"Gradient checkpointing não suportado para {model_name}")
    
    # Configurar AMP (Automatic Mixed Precision)
    if use_amp and torch.cuda.is_available():
        scaler, forward_amp, backward_amp = setup_amp_training(model, optimizer, criterion)
        logger.info(f"Treinamento com precisão mista (AMP) ativado para {model_name}")
    else:
        # Funções sem AMP
        scaler = None
        
        def forward_amp(inputs, labels):
            outputs = model(inputs)
            outputs = process_model_outputs(outputs, model_type)
            loss = criterion(outputs, labels)
            return outputs, loss
        
        def backward_amp(loss):
            loss.backward()
            optimizer.step()
    
    # Converter para DataLoader dinâmico se solicitado
    if dynamic_batch and hasattr(train_loader, 'dataset'):
        # Salvar as propriedades originais do dataloader
        original_batch_size = train_loader.batch_size
        dataset = train_loader.dataset
        num_workers = getattr(train_loader, 'num_workers', 4)
        pin_memory = getattr(train_loader, 'pin_memory', True)
        collate_fn = getattr(train_loader, 'collate_fn', None)
        
        # Criar DataLoader dinâmico
        train_loader = create_dynamic_batch_dataloader(
            dataset, 
            original_batch_size, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
        logger.info(f"DataLoader dinâmico criado com batch size inicial: {original_batch_size}")
    
    # Registro de número de parâmetros
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parâmetros: {trainable_params:,} treináveis de {total_params:,} total")
    
    # Escrever configuração no TensorBoard
    writer.add_text(f"{model_name}/config", 
                  f"Modelo: {model_name}\n" +
                  f"Batch Size inicial: {train_loader.batch_size}\n" +
                  f"Otimizador: {type(optimizer).__name__}\n" +
                  f"LR inicial: {optimizer.param_groups[0]['lr']}\n" +
                  f"Técnicas: mixup={use_mixup}, cutmix={use_cutmix}, AMP={use_amp}, " +
                  f"Gradient checkpointing={use_gradient_checkpointing}")
    
    # Resto da implementação da função train_model, adaptada para usar as otimizações
    device = next(model.parameters()).device
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0
    no_improve_epochs = 0
    
    # Listas para armazenar as métricas de treinamento
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Registrar hora de início
    start_time = time.time()
    
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            logger.info(f"Iniciando época {epoch+1}/{num_epochs}")
            
            # Monitorar memória no início da época
            monitor_memory_usage(logger)
            
            # ===== Fase de treinamento =====
            model.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0
            
            # Logar pesos de algumas camadas antes do treinamento
            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    writer.add_histogram(f"{model_name}/{name}", param, epoch)
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                try:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Log de progresso
                    if batch_idx % 20 == 0:
                        logger.debug(f"Treinamento: Batch {batch_idx}/{len(train_loader)}")
                        monitor_memory_usage(logger, interval_mb=50)
                    
                    # Zerar os gradientes
                    optimizer.zero_grad()
                    
                    # Aplicar mixup se ativado
                    if use_mixup:
                        from data_processing import mixup_data, mixup_criterion
                        
                        # Detectar o tipo de modelo com base no nome ou no parâmetro
                        detected_model_type = model_type
                        if model_type == 'cnn':  # Se não foi especificado, tentar detectar pelo nome
                            if 'swin' in model_name.lower():
                                detected_model_type = 'swin'
                            elif 'vit' in model_name.lower() or 'deit' in model_name.lower():
                                detected_model_type = 'vit'
                        
                        # Chamar mixup_data com o tipo de modelo
                        mixup_result = mixup_data(inputs, labels, alpha, detected_model_type)
                        
                        # Forward pass com AMP se ativado
                        if use_amp and torch.cuda.is_available():
                            with torch.cuda.amp.autocast():
                                # Verificar o tipo de resultado retornado pelo mixup_data
                                if detected_model_type.lower() in ['swin', 'swin_transformer'] or len(mixup_result) == 2:
                                    mixed_inputs, original_labels = mixup_result
                                    outputs = model(mixed_inputs)
                                    # Processar a saída para diferentes tipos de modelos
                                    outputs = process_model_outputs(outputs, model_type)
                                    loss = criterion(outputs, original_labels)
                                    # Acurácia calculada normalmente
                                    _, preds = torch.max(outputs, 1)
                                    running_corrects += torch.sum(preds == original_labels.data)
                                else:  # Para outros modelos: (mixed_x, y_a, y_b, lam)
                                    mixed_inputs, labels_a, labels_b, lam = mixup_result
                                    outputs = model(mixed_inputs)
                                    # Processar a saída para diferentes tipos de modelos
                                    outputs = process_model_outputs(outputs, model_type)
                                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam, detected_model_type)
                                    # Para mixup tradicional, calcular acurácia ponderada
                                    _, preds = torch.max(outputs, 1)
                                    corrects_a = torch.sum(preds == labels_a.data)
                                    corrects_b = torch.sum(preds == labels_b.data)
                                    running_corrects += lam * corrects_a + (1 - lam) * corrects_b
                            
                            # Backward pass com AMP
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # Implementação padrão sem AMP
                            # Verificar o tipo de resultado retornado pelo mixup_data
                            if detected_model_type.lower() in ['swin', 'swin_transformer'] or len(mixup_result) == 2:
                                mixed_inputs, original_labels = mixup_result
                                outputs = model(mixed_inputs)
                                # Processar a saída para diferentes tipos de modelos
                                outputs = process_model_outputs(outputs, model_type)
                                loss = criterion(outputs, original_labels)
                                # Acurácia calculada normalmente
                                _, preds = torch.max(outputs, 1)
                                running_corrects += torch.sum(preds == original_labels.data)
                            else:  # Para outros modelos: (mixed_x, y_a, y_b, lam)
                                mixed_inputs, labels_a, labels_b, lam = mixup_result
                                outputs = model(mixed_inputs)
                                # Processar a saída para diferentes tipos de modelos
                                outputs = process_model_outputs(outputs, model_type)
                                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam, detected_model_type)
                                # Para mixup tradicional, calcular acurácia ponderada
                                _, preds = torch.max(outputs, 1)
                                corrects_a = torch.sum(preds == labels_a.data)
                                corrects_b = torch.sum(preds == labels_b.data)
                                running_corrects += lam * corrects_a + (1 - lam) * corrects_b
                            
                            # Backward pass padrão
                            loss.backward()
                            optimizer.step()
                    else:
                        # Forward e backward pass com AMP
                        outputs, loss = forward_amp(inputs, labels)
                        
                        # Calcular acurácia
                        _, preds = torch.max(outputs, 1)
                        running_corrects += torch.sum(preds == labels.data)
                        
                        # Backward e otimização com AMP
                        backward_amp(loss)
                    
                    # Estatísticas
                    running_loss += loss.item() * inputs.size(0)
                    total += labels.size(0)
                    
                    # Liberar tensores para economia de memória
                    del inputs, labels, outputs, loss, preds
                    if 'mixed_inputs' in locals():
                        del mixed_inputs
                    if 'original_labels' in locals():
                        del original_labels
                    if 'labels_a' in locals() and 'labels_b' in locals():
                        del labels_a, labels_b
                    
                    # Coleta de lixo periódica
                    if batch_idx % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"CUDA OOM no batch {batch_idx}. Tentando reduzir batch size.", exc_info=True)
                        torch.cuda.empty_cache()
                        gc.collect()
        
                        # Limpar variáveis locais
                        if 'inputs' in locals(): del inputs
                        if 'labels' in locals(): del labels
                        if 'outputs' in locals(): del outputs
                        if 'loss' in locals(): del loss
                        if 'preds' in locals(): del preds
                        if 'mixed_inputs' in locals(): del mixed_inputs
                        if 'original_labels' in locals(): del original_labels
                        if 'labels_a' in locals() and 'labels_b' in locals(): del labels_a, labels_b
        
                        # Reduzir batch size se dataloader for dinâmico
                        if dynamic_batch and hasattr(train_loader, 'reduce_batch_size'):
                            reduced = train_loader.reduce_batch_size(min_batch_size=min_batch_size)
                            if reduced:
                                logger.info(f"Batch size reduzido para {train_loader.batch_size}")
                                continue  # Continuar com novo batch size
                            else:
                                logger.error("Não foi possível reduzir mais o batch size.")
                                # Tentar criar um dataloader de fallback com configurações mínimas
                                try:
                                    reduced_batch_size = min_batch_size
                                    logger.info(f"Criando dataloader de fallback com batch size {reduced_batch_size}")
                    
                                    # Quando o DataLoader é recriado para tamanho de batch reduzido
                                    fallback_train_loader = DataLoader(
                                        train_loader.dataset, 
                                        batch_size=reduced_batch_size, 
                                        shuffle=True, 
                                        num_workers=2,
                                        pin_memory=True if torch.cuda.is_available() else False,
                                        collate_fn=getattr(train_loader, 'collate_fn', None),
                                        multiprocessing_context='spawn'
                                    )
                    
                                    # Substituir o dataloader atual pelo fallback
                                    train_loader = fallback_train_loader
                                    logger.info(f"Dataloader substituído com sucesso. Continuando treinamento.")
                                    continue
                                except Exception as fallback_error:
                                    logger.error(f"Falha ao criar dataloader de fallback: {str(fallback_error)}")
                                    raise
                        else:
                            logger.error("OOM sem possibilidade de redução de batch size. Tentando criar dataloader emergencial.")
                            try:
                                # Tentar criar um dataloader de emergência com configurações mínimas
                                emergency_batch_size = 1
                                emergency_train_loader = DataLoader(
                                    train_loader.dataset, 
                                    batch_size=emergency_batch_size, 
                                    shuffle=True, 
                                    num_workers=1,
                                    pin_memory=False,
                                    collate_fn=getattr(train_loader, 'collate_fn', None),
                                    multiprocessing_context='spawn'
                                )
                                train_loader = emergency_train_loader
                                logger.warning(f"Criado dataloader de emergência com batch size={emergency_batch_size}")
                                continue
                            except Exception as emergency_error:
                                logger.error(f"Falha total na recuperação: {str(emergency_error)}")
                                raise
                    else:
                        logger.error(f"Erro no batch {batch_idx} durante treinamento", exc_info=True)
                        raise
            
            # Calcular métricas da época
            epoch_loss = running_loss / total if total > 0 else float('inf')
            epoch_acc = running_corrects.double() / total if total > 0 else 0
            
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc.item())
            
            logger.info(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Adicionar ao TensorBoard
            writer.add_scalar(f"{model_name}/Loss/train", epoch_loss, epoch)
            writer.add_scalar(f"{model_name}/Accuracy/train", epoch_acc, epoch)
            
            # Liberar memória após fase de treinamento
            cleanup_cuda_memory()
            
            # ===== Fase de validação =====
            model.eval()
            running_loss = 0.0
            running_corrects = 0
            total = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(val_loader):
                    try:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        # Forward pass
                        if use_amp and torch.cuda.is_available():
                            with torch.cuda.amp.autocast():
                                outputs = model(inputs)
                                outputs = process_model_outputs(outputs, model_type)
                                loss = criterion(outputs, labels)
                        else:
                            outputs = model(inputs)
                            outputs = process_model_outputs(outputs, model_type)
                            loss = criterion(outputs, labels)
                        
                        _, preds = torch.max(outputs, 1)
                        
                        # Estatísticas
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        total += labels.size(0)
                        
                        # Coletar para matriz de confusão
                        val_preds.extend(preds.cpu().numpy())
                        val_targets.extend(labels.cpu().numpy())
                        
                        # Liberar tensores para economia de memória
                        del inputs, labels, outputs, loss, preds
                        
                    except Exception as e:
                        logger.error(f"Erro no batch {batch_idx} durante validação", exc_info=True)
            
            # Calcular métricas de validação
            epoch_loss = running_loss / total if total > 0 else float('inf')
            epoch_acc = running_corrects.double() / total if total > 0 else 0
            
            val_losses.append(epoch_loss)
            val_accs.append(epoch_acc.item())
            
            # Calcular matriz de confusão para esta época
            if len(val_targets) > 0 and len(val_preds) > 0:
                num_classes = len(set(val_targets)) if hasattr(val_loader.dataset, 'classes') else None
                if num_classes is None:
                    if hasattr(val_loader.dataset, 'dataset') and hasattr(val_loader.dataset.dataset, 'classes'):
                        num_classes = len(val_loader.dataset.dataset.classes)
                    else:
                        num_classes = max(max(val_targets), max(val_preds)) + 1
                
                conf_matrix = confusion_matrix(val_targets, val_preds, labels=range(num_classes))
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Obter nomes das classes para os labels
                if hasattr(val_loader.dataset, 'classes'):
                    class_names = val_loader.dataset.classes
                elif hasattr(val_loader.dataset, 'dataset') and hasattr(val_loader.dataset.dataset, 'classes'):
                    class_names = val_loader.dataset.dataset.classes
                else:
                    class_names = [str(i) for i in range(num_classes)]
                
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=class_names,
                           yticklabels=class_names)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'Confusion Matrix - Epoch {epoch+1}')
                writer.add_figure(f"{model_name}/confusion_matrix", fig, epoch)
                plt.close(fig)
            
            val_time = time.time() - epoch_start_time
            logger.info(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Tempo: {val_time:.1f}s")
            
            # Adicionar ao TensorBoard
            writer.add_scalar(f"{model_name}/Loss/val", epoch_loss, epoch)
            writer.add_scalar(f"{model_name}/Accuracy/val", epoch_acc, epoch)
            writer.add_scalar(f"{model_name}/Learning_rate", optimizer.param_groups[0]['lr'], epoch)
            
            # Ajustar o learning rate (de acordo com o tipo de scheduler)
            old_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                
                if old_lr != new_lr:
                    logger.info(f"Learning rate ajustado: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Snapshot de memória a cada época
            memory_snapshot = get_memory_snapshot()
            logger.info(f"Memória após época {epoch+1}: GPU={memory_snapshot.get('cuda', {}).get('gpu_0', {}).get('allocated_gb', 0):.3f} GB")
            
            # Salvar o melhor modelo
            if epoch_acc > best_acc:
                logger.info(f"Melhor acurácia melhorou: {best_acc:.4f} -> {epoch_acc:.4f}")
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())
                no_improve_epochs = 0
                
                # Salvar melhor modelo até agora
                try:
                    model_save_path = os.path.join("models", f"{model_name}_best.pth")
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model_wts,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': best_acc,
                        'loss': epoch_loss,
                        'amp_scaler': scaler.state_dict() if scaler else None,  # Salvar estado do scaler AMP
                        'batch_size': train_loader.batch_size,  # Salvar batch size atual para retomada
                    }, model_save_path)
                    logger.info(f"Melhor modelo salvo como {model_save_path}")
                except Exception as e:
                    logger.error(f"Erro ao salvar o melhor modelo: {str(e)}")
            else:
                no_improve_epochs += 1
                logger.info(f"Sem melhoria há {no_improve_epochs} épocas. Melhor: {best_acc:.4f}")

            # Salvar checkpoint para retomada (a cada 3 épocas ou na última)
            if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
                checkpoint_path = os.path.join("models", f"{model_name}_checkpoint_e{epoch+1}.pt")
                from utils import create_training_checkpoint
                create_training_checkpoint(
                    model, optimizer, epoch, train_losses, val_losses, 
                    best_acc, filename=checkpoint_path,
                    amp_scaler=scaler,  # Incluir o scaler AMP no checkpoint
                    batch_size=train_loader.batch_size  # Incluir batch size atual
                )

            # Early stopping
            if early_stopping and no_improve_epochs >= patience:
                logger.info(f"Early stopping ativado após {epoch+1} épocas")
                break
            
            # Liberar memória no final de cada época
            cleanup_cuda_memory()
        
        # Tempo total de treinamento
        total_time = time.time() - start_time
        logger.info(f"Treinamento concluído em {total_time/60:.2f} minutos")
        logger.info(f"Melhor acurácia de validação: {best_acc:.4f}")
        
        # Adicionar resumo ao TensorBoard
        writer.add_hparams(
            {"model": model_name, "batch_size": getattr(train_loader, 'batch_size', min_batch_size), 
             "epochs": epoch+1, "mixup": use_mixup, "early_stopping": early_stopping,
             "amp": use_amp, "gradient_checkpointing": use_gradient_checkpointing},
            {"hparam/accuracy": best_acc}
        )
        
        # Desativar gradient checkpointing se estiver ativado
        if use_gradient_checkpointing:
            enable_gradient_checkpointing(model, enable=False)
        
        # Carregar o melhor modelo
        model.load_state_dict(best_model_wts)
        
        # Limpeza final de memória
        cleanup_cuda_memory()
        
        return model, train_losses, train_accs, val_losses, val_accs
    
    except Exception as e:
        logger.critical(f"Erro crítico durante o treinamento de {model_name}", exc_info=True)
        
        # Tentar salvar o modelo parcial para recuperação
        try:
            recovery_path = os.path.join("models", f"{model_name}_recovery.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'amp_scaler': scaler.state_dict() if scaler else None,
                'batch_size': train_loader.batch_size
            }, recovery_path)
            logger.info(f"Estado de recuperação salvo como {recovery_path}")
        except:
            logger.critical("Falha ao salvar o estado de recuperação")
        
        # Limpeza de memória após erro
        cleanup_cuda_memory()
        
        # Re-lançar a exceção para ser tratada pelo chamador
        raise


def dynamic_batch_size_training(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                               num_epochs, writer, model_name="model", initial_batch_size=None,
                               min_batch_size=4, **kwargs):
    """
    Treina o modelo com ajuste dinâmico de batch size em caso de erros OOM.
    
    Args:
        model: Modelo PyTorch a ser treinado
        train_loader: DataLoader para treinamento
        val_loader: DataLoader para validação
        criterion: Função de perda (ex: CrossEntropyLoss)
        optimizer: Otimizador (ex: Adam, SGD)
        scheduler: Scheduler para ajustar learning rate
        num_epochs: Número de épocas para treinar
        writer: SummaryWriter do TensorBoard
        model_name: Nome do modelo para logging
        initial_batch_size: Tamanho inicial de batch (se None, usa o do train_loader)
        min_batch_size: Tamanho mínimo de batch permitido
        **kwargs: Argumentos adicionais para train_model
        
    Returns:
        tuple: (modelo_treinado, train_losses, train_accs, val_losses, val_accs)
    """
    if initial_batch_size is None:
        initial_batch_size = train_loader.batch_size
    
    current_batch_size = initial_batch_size
    
    logger.info(f"Iniciando treinamento com ajuste dinâmico de batch size: {model_name}")
    logger.info(f"Batch size inicial: {current_batch_size}, batch size mínimo: {min_batch_size}")
    
    while current_batch_size >= min_batch_size:
        try:
            # Criar um novo dataloader com o tamanho de batch atual
            new_train_loader = DataLoader(
                train_loader.dataset,
                batch_size=current_batch_size,
                shuffle=True,
                num_workers=getattr(train_loader, 'num_workers', 4),
                pin_memory=getattr(train_loader, 'pin_memory', True),
                collate_fn=custom_collate_fn,
                multiprocessing_context='spawn'
            )
            
            logger.info(f"Tentando treinar com batch size = {current_batch_size}")
            
            # Chamar a função de treinamento normal
            return train_model(model, new_train_loader, val_loader, criterion, 
                              optimizer, scheduler, num_epochs, writer, 
                              model_name=model_name, **kwargs)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Reduzir o batch size e tentar novamente
                prev_batch_size = current_batch_size
                current_batch_size = max(current_batch_size // 2, min_batch_size)
                
                # Limpar memória
                cleanup_cuda_memory()
                
                logger.warning(f"OOM com batch size {prev_batch_size}. Reduzindo para {current_batch_size}.")
                
                # Se já está no batch size mínimo e falhou, desistir
                if current_batch_size == min_batch_size and prev_batch_size == min_batch_size:
                    logger.error("Falha mesmo com batch size mínimo. Impossível treinar este modelo.")
                    raise RuntimeError("Não foi possível treinar o modelo nem com o menor batch size.")
            else:
                # Se o erro não for OOM, repassar a exceção
                raise
    
    logger.error("Não foi possível treinar o modelo nem com o menor batch size.")
    return None, [], [], [], []


def train_model_with_mixed_precision(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                                     num_epochs, writer, model_name="model", use_amp=True, **kwargs):
    """
    Treina o modelo com precisão mista (Automatic Mixed Precision) se suportado.
    
    Args:
        model: Modelo PyTorch a ser treinado
        train_loader: DataLoader para treinamento
        val_loader: DataLoader para validação
        criterion: Função de perda (ex: CrossEntropyLoss)
        optimizer: Otimizador (ex: Adam, SGD)
        scheduler: Scheduler para ajustar learning rate
        num_epochs: Número de épocas para treinar
        writer: SummaryWriter do TensorBoard
        model_name: Nome do modelo para logging
        use_amp: Se True, usa precisão mista automática para treinamento
        **kwargs: Argumentos adicionais para train_model
        
    Returns:
        tuple: (modelo_treinado, train_losses, train_accs, val_losses, val_accs)
    """
    # Verificar se AMP é suportado
    if use_amp and not torch.cuda.is_available():
        logger.warning("AMP solicitado mas CUDA não está disponível. Desabilitando AMP.")
        use_amp = False
    
    logger.info(f"Iniciando treinamento com {'' if use_amp else 'sem '}precisão mista: {model_name}")
    
    # Definir scaler para AMP
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Resto da implementação específica para AMP
    device = next(model.parameters()).device
    
    # Modificar função train_model para usar AMP
    try:
        # Contador para o número de tentativas de recuperação AMP
        amp_retries = 0
        max_amp_retries = 3
        
        while True:
            try:
                # Preparar para treinamento
                if use_amp:
                    logger.info(f"Treinamento com precisão mista (AMP) ativado para {model_name}")
                
                # Chamar o train_model_optimized que já tem suporte AMP incorporado
                return train_model_optimized(
                    model, train_loader, val_loader, criterion, optimizer, scheduler,
                    num_epochs, writer, model_name=model_name,
                    use_amp=use_amp, use_gradient_checkpointing=False, dynamic_batch=False,
                    **kwargs
                )
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and use_amp and amp_retries < max_amp_retries:
                    # Se ocorrer OOM com AMP ativado, tentar reduzir escala AMP
                    amp_retries += 1
                    # Redefinir scaler com fator de crescimento menor
                    new_growth_factor = 1.0 - (0.1 * amp_retries)  # Reduzir crescimento gradualmente
                    scaler = torch.cuda.amp.GradScaler(growth_factor=new_growth_factor)
                    
                    cleanup_cuda_memory()
                    logger.warning(f"OOM com AMP. Tentativa {amp_retries}/{max_amp_retries} com growth_factor={new_growth_factor}")
                    continue
                    
                elif "CUDA error" in str(e).lower() and use_amp and amp_retries < max_amp_retries:
                    # Erros CUDA podem ocorrer com AMP, tentar ajustar
                    amp_retries += 1
                    cleanup_cuda_memory()
                    logger.warning(f"Erro CUDA com AMP. Tentativa {amp_retries}/{max_amp_retries}")
                    continue
                    
                elif amp_retries >= max_amp_retries and use_amp:
                    # Se todas as tentativas com AMP falharam, tentar sem AMP
                    use_amp = False
                    cleanup_cuda_memory()
                    logger.warning(f"AMP falhou após {max_amp_retries} tentativas. Tentando sem AMP.")
                    continue
                    
                else:
                    # Outros erros ou AMP já desabilitado, repassar exceção
                    raise
    
    except Exception as e:
        logger.error(f"Erro durante o treinamento com precisão mista: {str(e)}", exc_info=True)
        # Limpar memória e repassar exceção
        cleanup_cuda_memory()
        raise


def process_model_outputs(outputs, model_type):
    """
    Processa as saídas de diferentes tipos de modelos para garantir formato consistente.
    
    Args:
        outputs: Saídas do modelo
        model_type: Tipo de modelo ('cnn', 'vit', 'swin', etc.)
        
    Returns:
        torch.Tensor: Saídas processadas no formato correto para cálculo de perda
    """
    # Para modelos do Hugging Face ou modelos tipo ViT que retornam dicionários
    if isinstance(outputs, dict):
        if 'logits' in outputs:
            return outputs['logits']
        elif 'last_hidden_state' in outputs:
            # Alguns modelos podem retornar apenas estados ocultos
            logger.warning("Modelo retornando apenas hidden states, sem logits diretos")
            return outputs['last_hidden_state'].mean(dim=1)  # Média pooling sobre tokens
        else:
            # Tentar encontrar uma chave que pareça um tensor de saída
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                    return value
            
            # Falha segura - retornar o primeiro tensor de saída
            logger.warning(f"Formato de saída desconhecido: {outputs.keys()}")
            return list(outputs.values())[0]
    
    # Para modelos que retornam tuplas (comum em alguns modelos PyTorch)
    elif isinstance(outputs, tuple):
        # Assumir que o primeiro elemento é o tensor de logits
        return outputs[0]
    
    # Processamento para tensores 4D [batch_size, altura, largura, classes] ou similar
    if isinstance(outputs, torch.Tensor) and len(outputs.shape) == 4:
        # Log para debug da forma do tensor
        logger.info(f"Processando saída com forma 4D: {outputs.shape}")
        
        # Verificar qual dimensão provavelmente contém as classes
        batch_size = outputs.shape[0]
        
        # Se o último canal tiver tamanho compatível com o número de classes (geralmente < 20)
        if outputs.shape[3] < 20:
            # Reorganizar de [batch, h, w, classes] para [batch, classes, h, w]
            outputs = outputs.permute(0, 3, 1, 2)
            logger.info(f"Reorganizando tensor para forma: {outputs.shape}")
        
        # Se ainda tiver 4 dimensões, fazer pooling global para obter tensor 2D
        if len(outputs.shape) == 4:
            # Global Average Pooling nas dimensões espaciais
            outputs = outputs.mean(dim=[2, 3])
            logger.info(f"Aplicando pooling global, nova forma: {outputs.shape}")
    
    # Para modelos ViT e Swin específicos que podem precisar de processamento especial
    if model_type.lower() in ['vit', 'swin', 'swin_transformer']:
        # Verificar se a saída é um tensor
        if isinstance(outputs, torch.Tensor):
            # Se não for 2D, tentar converter para o formato [batch_size, num_classes]
            if len(outputs.shape) != 2:
                logger.warning(f"Saída {model_type} tem forma {outputs.shape}, tentando reshape")
                batch_size = outputs.shape[0]
                
                # Se for 4D, primeiro aplicar pooling global
                if len(outputs.shape) == 4:
                    outputs = outputs.mean(dim=[2, 3])
                    logger.info(f"Após pooling: {outputs.shape}")
                
                # Então garantir formato 2D
                outputs = outputs.view(batch_size, -1)
                logger.info(f"Após reshape: {outputs.shape}")
                
                # Se a dimensão de features for muito grande, pode precisar de redução adicional
                if outputs.shape[1] > 1000:
                    logger.warning(f"Dimensão de features muito grande: {outputs.shape[1]}")
                    # Considerar implementar redução de dimensionalidade se necessário
    
    # Saída padrão para outros modelos ou quando o formato já é adequado
    return outputs


def evaluate_model(model, test_loader, classes, writer=None, model_name="model", use_amp=False):
    """
    Avalia o modelo com logging detalhado e visualizações.
    
    Args:
        model: Modelo PyTorch treinado
        test_loader: DataLoader para o conjunto de teste
        classes: Lista com nomes das classes
        writer: SummaryWriter do TensorBoard (opcional)
        model_name: Nome do modelo para logging
        use_amp: Se True, usa precisão mista automática para inferência
        
    Returns:
        tuple: (acurácia, relatório, matriz_confusão, predições, labels, probabilidades)
    """
    # Detectar o tipo de modelo com base no nome
    model_type = 'cnn'
    if 'vit' in model_name.lower():
        model_type = 'vit'
    elif 'swin' in model_name.lower():
        model_type = 'swin'
    
    logger.info(f"Iniciando avaliação de {model_name} (tipo: {model_type})")
    
    # Liberar memória explicitamente antes da avaliação
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Memória GPU liberada antes da avaliação de {model_name}")
        logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    
    try:
        device = next(model.parameters()).device
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Tempo de inferência acumulado
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                # Log de progresso
                if batch_idx % 10 == 0:
                    logger.debug(f"Avaliando batch {batch_idx}/{len(test_loader)}")
                
                try:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Medir tempo de inferência
                    start_time = time.time()
                    
                    # Forward pass com AMP se ativado
                    if use_amp and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            # Processar saídas para diferentes tipos de modelos
                            outputs = process_model_outputs(outputs, model_type)
                    else:
                        outputs = model(inputs)
                        # Processar saídas para diferentes tipos de modelos
                        outputs = process_model_outputs(outputs, model_type)
                    
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
                    # Liberar memória explicitamente
                    del inputs, labels, outputs, probs, preds
                    
                    # Liberação periódica de memória
                    if batch_idx % 20 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Erro no batch {batch_idx} durante avaliação", exc_info=True)
        
        # Calcular métricas e tempos
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        logger.info(f"Tempo médio de inferência por batch: {avg_inference_time*1000:.2f} ms")
        logger.info(f"Throughput estimado: {test_loader.batch_size/avg_inference_time:.1f} imagens/s")
        
        # Calcular métricas
        if len(all_labels) > 0 and len(all_preds) > 0:
            accuracy = accuracy_score(all_labels, all_preds)
            report_dict = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
            report = classification_report(all_labels, all_preds, target_names=classes)
            conf_matrix = confusion_matrix(all_labels, all_preds)
            
            logger.info(f"Acurácia do modelo: {accuracy:.4f}")
            logger.info(f"Relatório de classificação:\n{report}")
            
            # Logar métricas por classe
            logger.info("Métricas por classe:")
            for i, class_name in enumerate(classes):
                precision = report_dict[class_name]['precision']
                recall = report_dict[class_name]['recall']
                f1 = report_dict[class_name]['f1-score']
                support = report_dict[class_name]['support']
                
                logger.info(f"Classe '{class_name}': Precisão={precision:.4f}, "
                           f"Recall={recall:.4f}, F1={f1:.4f}, Suporte={support}")
                
                if writer:
                    writer.add_scalar(f"{model_name}/Metrics/{class_name}/precision", precision, 0)
                    writer.add_scalar(f"{model_name}/Metrics/{class_name}/recall", recall, 0)
                    writer.add_scalar(f"{model_name}/Metrics/{class_name}/f1", f1, 0)
            
            # Identificar classes problemáticas
            problematic_classes = []
            for i, class_name in enumerate(classes):
                f1 = report_dict[class_name]['f1-score']
                if f1 < 0.7:  # Limiar ajustável
                    problematic_classes.append((class_name, f1))
            
            if problematic_classes:
                logger.warning("Classes com baixo desempenho:")
                for class_name, f1 in problematic_classes:
                    logger.warning(f"  - {class_name}: F1={f1:.4f}")
            
            # Visualizar a matriz de confusão
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                      xticklabels=classes, yticklabels=classes)
            plt.title(f'Matriz de Confusão - {model_name}')
            plt.xlabel('Classe Prevista')
            plt.ylabel('Classe Real')
            plt.tight_layout()
            
            # Salvar matriz de confusão
            conf_matrix_path = os.path.join('confusion_matrices', f'confusion_matrix_{model_name}.png')
            os.makedirs(os.path.dirname(conf_matrix_path), exist_ok=True)
            plt.savefig(conf_matrix_path)
            logger.info(f"Matriz de confusão salva em: {conf_matrix_path}")
            plt.close()
            
            # Adicionar ao TensorBoard
            if writer:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=classes, yticklabels=classes)
                ax.set_title(f'Matriz de Confusão - {model_name}')
                ax.set_xlabel('Classe Prevista')
                ax.set_ylabel('Classe Real')
                writer.add_figure(f"{model_name}/confusion_matrix", fig, 0)
                plt.close(fig)
                
                # Adicionar accuracy global
                writer.add_scalar(f"{model_name}/Metrics/accuracy", accuracy, 0)
                
                # Adicionar exemplos de inferência
                try:
                    example_batch, example_labels = next(iter(test_loader))
                    example_batch = example_batch[:8].to(device)  # Limitar a algumas imagens
                    example_labels = example_labels[:8].to(device)
                    
                    with torch.no_grad():
                        if use_amp and torch.cuda.is_available():
                            with torch.cuda.amp.autocast():
                                example_outputs = model(example_batch)
                                # Processar saídas para diferentes tipos de modelos
                                example_outputs = process_model_outputs(example_outputs, model_type)
                        else:
                            example_outputs = model(example_batch)
                            # Processar saídas para diferentes tipos de modelos
                            example_outputs = process_model_outputs(example_outputs, model_type)
                            
                        _, example_preds = torch.max(example_outputs, 1)
                    
                    # Preparar figura com predições
                    fig = plt.figure(figsize=(12, 6))
                    for i, (img, label, pred) in enumerate(zip(example_batch, example_labels, example_preds)):
                        # Normalizar imagem para visualização
                        img = img.cpu().numpy().transpose(1, 2, 0)
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img = std * img + mean
                        img = np.clip(img, 0, 1)
                        
                        plt.subplot(2, 4, i + 1)
                        plt.imshow(img)
                        plt.title(
                            f"Real: {classes[label]}\\nPred: {classes[pred]}",
                            color="green" if pred == label else "red"
                        )
                        plt.axis("off")
                    
                    plt.tight_layout()
                    writer.add_figure(f"{model_name}/example_predictions", fig, 0)
                    plt.close(fig)
                    
                    # Liberar memória
                    del example_batch, example_labels, example_outputs, example_preds
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Erro ao adicionar exemplos de inferência ao TensorBoard: {str(e)}")
            
            # Limpar memória após avaliação
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info(f"Memória GPU liberada após avaliação de {model_name}")
                logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            
            return accuracy, report_dict, conf_matrix, all_preds, all_labels, all_probs
        else:
            logger.error("Não foi possível calcular métricas: dados de saída vazios")
            return 0, {}, np.array([]), [], [], []
    
    except Exception as e:
        logger.critical(f"Erro crítico durante a avaliação do modelo {model_name}", exc_info=True)
        # Limpar memória após erro
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Memória GPU liberada após erro na avaliação")
        # Retornar valores vazios ou nulos
        return 0, {}, np.array([]), [], [], []


def analyze_error_cases(model, test_loader, classes, num_samples=10, writer=None, model_name="model", use_amp=False):
    """
    Analisa e visualiza os casos em que o modelo comete erros mais graves.
    
    Args:
        model: Modelo PyTorch treinado
        test_loader: DataLoader para o conjunto de teste
        classes: Lista com nomes das classes
        num_samples: Número de amostras de erro para visualizar
        writer: SummaryWriter do TensorBoard (opcional)
        model_name: Nome do modelo para logging
        use_amp: Se True, usa precisão mista automática para inferência
        
    Returns:
        list: Lista de dicionários com informações dos erros encontrados
    """
    logger.info(f"Analisando casos de erro para o modelo {model_name}")
    
    # Liberar memória antes de iniciar a análise
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Memória GPU liberada antes da análise de erros de {model_name}")
        logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    
    device = next(model.parameters()).device
    model.eval()
    errors = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass com AMP se ativado
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
            else:
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
            confidences, preds = torch.max(probs, 1)
            
            # Garantir que os tensores tenham dimensões compatíveis
            if preds.shape != labels.shape:
                # Redimensionar para compatibilidade
                if len(preds.shape) > len(labels.shape):
                    # Se o modelo retornar tensores com dimensões extras, aplicar squeeze ou seleção
                    if preds.shape[0] == labels.shape[0]:
                        # Se o batch size é compatível, mas há dimensões extras
                        preds = preds.view(labels.shape)
                # Se ainda houver incompatibilidade, registrar e retornar
                if preds.shape != labels.shape:
                    logger.error(f"Incompatibilidade de dimensões: preds {preds.shape}, labels {labels.shape}")
                    # Limpar memória antes de retornar
                    del inputs, labels, outputs, probs, preds, confidences
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return []  # Retornar lista vazia em caso de erro

            # Encontrar erros
            error_mask = (preds != labels)
            
            if error_mask.any():
                error_indices = error_mask.nonzero(as_tuple=True)[0]
                for idx in error_indices:
                    # Calcular o quão confiante o modelo estava no erro
                    confidence = confidences[idx].item()
                    prediction = preds[idx].item()
                    true_label = labels[idx].item()
                    true_class_confidence = probs[idx, true_label].item()
                    
                    # Calcular um "error score" - maior valor significa erro mais grave
                    # (alta confiança na classe errada, baixa confiança na classe correta)
                    error_score = confidence * (1 - true_class_confidence)
                    
                    errors.append({
                        'input': inputs[idx].cpu(),
                        'true_label': true_label,
                        'predicted': prediction,
                        'confidence': confidence,
                        'true_class_confidence': true_class_confidence,
                        'error_score': error_score
                    })
            
            # Liberar memória após processar cada batch
            del inputs, labels, outputs, probs, preds, confidences
            
            # Limpeza periódica de memória
            if len(errors) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Ordenar erros pelo error_score (erros mais graves primeiro)
    errors.sort(key=lambda x: x['error_score'], reverse=True)
    
    # Limitar ao número de amostras
    errors = errors[:num_samples]
    
    # Visualizar os erros mais graves
    if errors:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6)) if num_samples >= 10 else plt.subplots(1, num_samples, figsize=(15, 3))
        axes = axes.flatten() if num_samples >= 10 else [axes] if num_samples == 1 else axes
        
        for i, error in enumerate(errors):
            if i >= len(axes):
                break
                
            # Normalizar imagem para visualização
            img = error['input'].numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            axes[i].set_title(
                f"Real: {classes[error['true_label']]}\n"\
                f"Pred: {classes[error['predicted']]} ({error['confidence']:.2f})\n"\
                f"Conf. real: {error['true_class_confidence']:.2f}", 
                color='red'
            )
            axes[i].axis('off')
        
        plt.tight_layout()
        error_analysis_path = os.path.join('error_analysis', f'{model_name}_error_analysis.png')
        os.makedirs(os.path.dirname(error_analysis_path), exist_ok=True)
        plt.savefig(error_analysis_path)

        if writer:
            writer.add_figure(f"{model_name}/error_analysis", fig, 0)

        plt.close(fig)

        logger.info(f"Análise de erros salva em {error_analysis_path}")
        
        # Resumo dos erros mais comuns (matriz de confusão de erros)
        error_pairs = [(e['true_label'], e['predicted']) for e in errors]
        error_counts = {}
        for true_label, pred_label in error_pairs:
            key = (classes[true_label], classes[pred_label])
            error_counts[key] = error_counts.get(key, 0) + 1
        
        # Ordenar e registrar os tipos de erros mais comuns
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        logger.info("Erros mais comuns:")
        for (true_class, pred_class), count in sorted_errors:
            logger.info(f"  {true_class} classificado como {pred_class}: {count} ocorrências")
        
        # Liberar memória após análise de erros
        for error in errors:
            del error['input']  # Remover os tensores de imagem para liberar memória
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Memória GPU liberada após análise de erros de {model_name}")
        
        return errors
    else:
        logger.info(f"Nenhum erro encontrado para o modelo {model_name} no conjunto de teste")
        # Liberar memória mesmo sem erros
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return []


def perform_cross_validation(model_fn, dataframe, transform, n_folds=5, n_epochs=3, 
                            batch_size=32, use_amp=False, use_gradient_checkpointing=False):
    """
    Realiza validação cruzada K-fold para avaliar a performance do modelo, com otimizações de memória.
    
    Args:
        model_fn: Função que retorna um modelo PyTorch não inicializado
        dataframe: DataFrame pandas com os dados
        transform: Transformações a serem aplicadas
        n_folds: Número de folds para validação cruzada
        n_epochs: Número de épocas para treinar cada fold
        batch_size: Tamanho do batch
        use_amp: Se True, usa precisão mista automática
        use_gradient_checkpointing: Se True, ativa checkpoint de gradiente
        
    Returns:
        tuple: (média das acurácias, desvio padrão, lista de acurácias por fold)
    """
    logger.info(f"Iniciando validação cruzada com {n_folds} folds")
    logger.info(f"Configurações: AMP={use_amp}, Gradient checkpointing={use_gradient_checkpointing}")
    
    # Liberar memória antes de iniciar a validação cruzada
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Memória GPU liberada antes da validação cruzada")
        logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    
    # Configurar modelo
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Configurar dataset base
    from dataset import IntelImageDataset
    dataset = IntelImageDataset(dataframe, transform=transform)
    
    # Listas para resultados
    fold_results = []
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Para cada fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
        logger.info(f"Treinando fold {fold+1}/{n_folds}")
        
        # Limpar memória antes de iniciar cada fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Memória GPU liberada antes do fold {fold+1}")
            logger.info(f"Memória GPU alocada: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        
        # Samplers para datasets
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        # Dataloaders
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)
        
        # Tentar criar modelo com batch size dinâmico se necessário
        try:
            # Criar modelo, critério, otimizador e scheduler
            model = model_fn().to(device)
            
            # Ativar gradient checkpointing se solicitado
            if use_gradient_checkpointing:
                checkpoint_enabled = enable_gradient_checkpointing(model, enable=True)
                if checkpoint_enabled:
                    logger.info(f"Gradient checkpointing ativado para fold {fold+1}")
            
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
            # Criar writer específico para este fold
            from torch.utils.tensorboard import SummaryWriter
            fold_writer = SummaryWriter(log_dir=os.path.join("tensorboard_logs", f"fold_{fold+1}"))
            
            # Treinar modelo para este fold com otimizações
            if use_amp:
                # Usar função com precisão mista
                model, _, _, _, _ = train_model_with_mixed_precision(
                    model, train_loader, val_loader, criterion, optimizer, scheduler,
                    n_epochs, fold_writer, f"fold_{fold+1}", use_amp=True, 
                    early_stopping=True, patience=2
                )
            else:
                # Usar função padrão
                model, _, _, _, _ = train_model(
                    model, train_loader, val_loader, criterion, optimizer, scheduler,
                    n_epochs, fold_writer, f"fold_{fold+1}", early_stopping=True, patience=2
                )
            
            # Avaliar modelo
            accuracy, _, _, _, _, _ = evaluate_model(
                model, val_loader, dataset.classes, fold_writer, f"fold_{fold+1}",
                use_amp=use_amp
            )
            
            # Adicionar ao TensorBoard
            fold_writer.add_scalar("cross_validation/accuracy", accuracy, fold)
            
            # Desativar gradient checkpointing se estiver ativado
            if use_gradient_checkpointing:
                enable_gradient_checkpointing(model, enable=False)
            
            # Fechar writer
            fold_writer.close()
            
            # Salvar resultado
            fold_results.append(accuracy)
            logger.info(f"Fold {fold+1} concluído com acurácia: {accuracy:.4f}")
            
            # Limpar memória
            del model, optimizer, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info(f"Memória GPU liberada após fold {fold+1}")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM durante o treinamento do fold {fold+1}. Tentando reduzir batch size.")
                
                # Tentar novamente com batch size reduzido
                reduced_batch_size = batch_size // 2
                logger.info(f"Tentando novamente com batch size reduzido: {reduced_batch_size}")
                
                # Limpar memória
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Recriar dataloaders com batch size reduzido
                train_loader = DataLoader(dataset, batch_size=reduced_batch_size, sampler=train_sampler, num_workers=2)
                val_loader = DataLoader(dataset, batch_size=reduced_batch_size, sampler=val_sampler, num_workers=2)
                
                # Tentar novamente
                try:
                    # Criar modelo novamente
                    model = model_fn().to(device)
                    
                    # Ativar gradient checkpointing se solicitado
                    if use_gradient_checkpointing:
                        checkpoint_enabled = enable_gradient_checkpointing(model, enable=True)
                    
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
                    
                    # Criar writer específico para este fold
                    from torch.utils.tensorboard import SummaryWriter
                    fold_writer = SummaryWriter(log_dir=os.path.join("tensorboard_logs", f"fold_{fold+1}_retry"))
                    
                    # Tentar treinar com dynamic_batch_size_training para ajuste automático
                    model, _, _, _, _ = dynamic_batch_size_training(
                        model, train_loader, val_loader, criterion, optimizer, scheduler,
                        n_epochs, fold_writer, f"fold_{fold+1}_retry", 
                        initial_batch_size=reduced_batch_size, min_batch_size=2,
                        early_stopping=True, patience=2, use_amp=use_amp
                    )
                    
                    # Avaliar modelo
                    accuracy, _, _, _, _, _ = evaluate_model(
                        model, val_loader, dataset.classes, fold_writer, f"fold_{fold+1}_retry",
                        use_amp=use_amp
                    )
                    
                    # Adicionar ao TensorBoard
                    fold_writer.add_scalar("cross_validation/accuracy", accuracy, fold)
                    
                    # Desativar gradient checkpointing se estiver ativado
                    if use_gradient_checkpointing:
                        enable_gradient_checkpointing(model, enable=False)
                    
                    # Fechar writer
                    fold_writer.close()
                    
                    # Salvar resultado
                    fold_results.append(accuracy)
                    logger.info(f"Fold {fold+1} (retry) concluído com acurácia: {accuracy:.4f}")
                    
                    # Limpar memória
                    del model, optimizer, scheduler
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                except Exception as retry_e:
                    logger.error(f"Falha na segunda tentativa do fold {fold+1}: {str(retry_e)}")
                    # Adicionar um resultado vazio para este fold
                    fold_results.append(0.0)
            else:
                logger.error(f"Erro não relacionado a OOM no fold {fold+1}: {str(e)}")
                # Adicionar um resultado vazio para este fold
                fold_results.append(0.0)
    
    # Calcular média e desvio padrão (apenas para resultados válidos)
    valid_results = [r for r in fold_results if r > 0]
    if valid_results:
        cross_val_mean = np.mean(valid_results)
        cross_val_std = np.std(valid_results)
    else:
        cross_val_mean = 0
        cross_val_std = 0
    
    logger.info(f"Validação cruzada concluída com {len(valid_results)} de {n_folds} folds válidos")
    logger.info(f"Acurácia média: {cross_val_mean:.4f} ± {cross_val_std:.4f}")
    logger.info(f"Resultados por fold: {fold_results}")
    
    # Adicionar métricas de validação cruzada ao TensorBoard
    tb_writer = SummaryWriter(log_dir="tensorboard_logs/cross_validation")
    tb_writer.add_scalar("cross_validation/mean_accuracy", cross_val_mean, 0)
    tb_writer.add_scalar("cross_validation/std_accuracy", cross_val_std, 0)
    
    # Adicionar resultados individuais de cada fold
    for i, result in enumerate(fold_results):
        tb_writer.add_scalar("cross_validation/fold_accuracy", result, i)
    
    tb_writer.close()
    
    # Limpar memória final
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Memória GPU liberada após validação cruzada completa")
    
    return cross_val_mean, cross_val_std, fold_results


# Exemplo de uso das funções otimizadas
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Treinamento otimizado de modelos de classificação")
    parser.add_argument('--model', type=str, default='resnet50', help='Modelo a ser treinado')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamanho do batch')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas')
    parser.add_argument('--amp', action='store_true', help='Usar precisão mista automática')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Usar gradient checkpointing')
    parser.add_argument('--dynamic_batch', action='store_true', help='Usar ajuste dinâmico de batch size')
    
    args = parser.parse_args()
    
    # Configurar logs
    import logging
    from utils import setup_logging
    
    setup_logging()
    logger = logging.getLogger("landscape_classifier")
    
    logger.info(f"Iniciando treinamento com modelo {args.model}")
    logger.info(f"Parâmetros: batch_size={args.batch_size}, epochs={args.epochs}, amp={args.amp}, " +
               f"gradient_checkpointing={args.gradient_checkpointing}, dynamic_batch={args.dynamic_batch}")
    
    # Carregar dados
    # Aqui você importaria seu código para carregar os dados e criar transformações
    # Por exemplo:
    from data_processing import load_data, get_transforms
    
    train_df, val_df, test_df = load_data()
    train_transform, val_transform = get_transforms()
    
    # Criar datasets e dataloaders
    from dataset import IntelImageDataset
    from torch.utils.data import DataLoader
    
    train_dataset = IntelImageDataset(train_df, transform=train_transform)
    val_dataset = IntelImageDataset(val_df, transform=val_transform)
    test_dataset = IntelImageDataset(test_df, transform=val_transform)
    
    if args.dynamic_batch:
        # Usar DataLoader dinâmico para treinamento
        train_loader = create_dynamic_batch_dataloader(
            train_dataset, args.batch_size, num_workers=4, pin_memory=True
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Importar modelo
    from models import get_model
    
    model = get_model(args.model, num_classes=len(train_dataset.classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Definir critério, otimizador e scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Configurar TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    
    writer = SummaryWriter(log_dir=f"tensorboard_logs/{args.model}")
    
    # Treinar modelo com as otimizações apropriadas
    if args.amp and args.gradient_checkpointing and args.dynamic_batch:
        # Usar todas as otimizações
        model, train_losses, train_accs, val_losses, val_accs = train_model_optimized(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            args.epochs, writer, model_name=args.model,
            use_amp=True, use_gradient_checkpointing=True, dynamic_batch=True,
            min_batch_size=4
        )
    elif args.amp:
        # Usar apenas precisão mista
        model, train_losses, train_accs, val_losses, val_accs = train_model_with_mixed_precision(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            args.epochs, writer, model_name=args.model,
            use_amp=True
        )
    elif args.dynamic_batch:
        # Usar apenas batch size dinâmico
        model, train_losses, train_accs, val_losses, val_accs = dynamic_batch_size_training(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            args.epochs, writer, model_name=args.model,
            initial_batch_size=args.batch_size, min_batch_size=4
        )
    else:
        # Usar treinamento padrão
        model, train_losses, train_accs, val_losses, val_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            args.epochs, writer, model_name=args.model
        )
    
    # Avaliar modelo
    accuracy, report, conf_matrix, _, _, _ = evaluate_model(
        model, test_loader, train_dataset.classes, writer, args.model,
        use_amp=args.amp
    )
    
    logger.info(f"Acurácia final no conjunto de teste: {accuracy:.4f}")
    
    # Analisar casos de erro
    analyze_error_cases(
        model, test_loader, train_dataset.classes, num_samples=10, 
        writer=writer, model_name=args.model, use_amp=args.amp
    )
    
    # Salvar modelo final
    model_save_path = os.path.join("models", f"{args.model}_final.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy
    }, model_save_path)
    
    logger.info(f"Modelo final salvo em {model_save_path}")
    
    # Fechar writer
    writer.close()