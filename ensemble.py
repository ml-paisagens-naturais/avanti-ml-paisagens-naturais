"""
Ensemble Module

Este módulo contém funções para criar e avaliar ensembles de modelos.
Os ensembles combinam as previsões de múltiplos modelos para melhorar a acurácia.
"""

import torch
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Obter o logger
logger = logging.getLogger("landscape_classifier")

def evaluate_ensemble(models, dataloaders, classes, model_names=None, writer=None):
    """
    Avalia o ensemble de modelos.
    
    Args:
        models: Lista de modelos PyTorch
        dataloaders: Lista de DataLoaders (um para cada modelo)
        classes: Lista de nomes de classes
        model_names: Lista de nomes dos modelos (para logging)
        writer: SummaryWriter do TensorBoard para visualizações
    
    Returns:
        tuple: (acurácia, relatório_classificação, matriz_confusão, predições, labels)
    """
    logger.info("Iniciando avaliação do ensemble de modelos...")
    
    try:
        # Verificar se todos os dataloaders têm o mesmo tamanho
        if len(set(len(dl) for dl in dataloaders)) > 1:
            logger.warning("Dataloaders têm tamanhos diferentes. Usando o menor para ensemble.")
        
        if model_names is None:
            model_names = [f"Model_{i}" for i in range(len(models))]
        
        # Colocar modelos em modo de avaliação
        for model in models:
            model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []  # Para cada classe
        
        # Rastrear acurácia individual de cada modelo para comparação
        individual_correct = [0] * len(models)
        individual_total = 0
        
        # Tempo para inferência
        ensemble_times = []
        
        logger.info("Processando batches para avaliação do ensemble...")
        
        # Obter iteradores para cada dataloader
        dataloader_iters = [iter(dl) for dl in dataloaders]
        num_batches = min(len(dl) for dl in dataloaders)
        
        for batch_idx in range(num_batches):
            try:
                # Obter batch de cada dataloader
                batch_data = []
                for dl_iter in dataloader_iters:
                    inputs, labels = next(dl_iter)
                    inputs = inputs.to(next(models[0].parameters()).device)
                    batch_data.append((inputs, labels))
                
                # Usar labels do primeiro dataloader
                labels = batch_data[0][1].to(next(models[0].parameters()).device)
                individual_total += labels.size(0)
                
                # Medir tempo de inferência do ensemble
                start_time = time.time()
                
                # Prever com cada modelo individualmente
                model_outputs = []
                model_probs = []
                
                with torch.no_grad():
                    for i, (model, (inputs, _)) in enumerate(zip(models, batch_data)):
                        # Inferência individual
                        outputs = model(inputs)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        
                        # Registrar predições individuais para comparação
                        _, model_preds = torch.max(outputs, 1)
                        individual_correct[i] += torch.sum(model_preds == labels.data).item()
                        
                        model_outputs.append(outputs)
                        model_probs.append(probs)
                    
                    # Combinar probabilidades (média)
                    ensemble_probs = torch.mean(torch.stack(model_probs), dim=0)
                    _, ensemble_preds = torch.max(ensemble_probs, 1)
                
                # Medir tempo total
                inference_time = time.time() - start_time
                ensemble_times.append(inference_time)
                
                # Coletar resultados
                all_preds.extend(ensemble_preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(ensemble_probs.cpu().numpy())
                
                # Log de progresso
                if batch_idx % 10 == 0:
                    logger.debug(f"Ensemble: Batch {batch_idx}/{num_batches} processado")
            
            except Exception as e:
                logger.error(f"Erro no batch {batch_idx} do ensemble", exc_info=True)
        
        # Verificar e corrigir número de classes antes de calcular o relatório
        if len(all_preds) > 0:
            # Encontrar o número real de classes nos dados
            unique_pred_classes = np.unique(all_preds)
            unique_label_classes = np.unique(all_labels)
            unique_classes = np.unique(np.concatenate((unique_pred_classes, unique_label_classes)))
            
            if len(unique_classes) != len(classes):
                logger.warning(f"Incompatibilidade no número de classes: encontradas {len(unique_classes)}, esperadas {len(classes)}")
                
                # Criar subconjunto de classes usadas nas predições e labels
                used_classes = []
                used_indices = []
                for i, class_name in enumerate(classes):
                    if i in unique_classes:
                        used_classes.append(class_name)
                        used_indices.append(i)
                
                # Filtrar predições e labels para incluir apenas classes usadas
                mask = np.isin(all_labels, used_indices)
                filtered_labels = np.array(all_labels)[mask]
                filtered_preds = np.array(all_preds)[mask]
                
                # Calcular métricas com classes filtradas
                if len(filtered_labels) > 0:
                    accuracy = accuracy_score(filtered_labels, filtered_preds)
                    report = classification_report(filtered_labels, filtered_preds, 
                                                 target_names=used_classes, output_dict=True)
                    report_text = classification_report(filtered_labels, filtered_preds, 
                                                      target_names=used_classes)
                    conf_matrix = confusion_matrix(filtered_labels, filtered_preds)
                    
                    # Atualizar os arrays originais para continuar o processamento
                    all_labels = filtered_labels
                    all_preds = filtered_preds
                else:
                    logger.error("Após filtragem, não há exemplos para calcular métricas")
                    accuracy = 0
                    report = {}
                    report_text = "Sem dados para gerar relatório"
                    conf_matrix = np.array([])
            else:
                # Cálculo normal se o número de classes corresponder
                accuracy = accuracy_score(all_labels, all_preds)
                report = classification_report(all_labels, all_preds, 
                                             target_names=classes, output_dict=True)
                report_text = classification_report(all_labels, all_preds, 
                                                  target_names=classes)
                conf_matrix = confusion_matrix(all_labels, all_preds)
        else:
            # Default para caso de falha
            logger.error("Não há predições para calcular métricas")
            accuracy = 0
            report = {}
            report_text = "Sem dados para gerar relatório"
            conf_matrix = np.array([])
        
        logger.info(f"Acurácia do ensemble: {accuracy:.4f}")
        logger.info(f"Relatório de classificação do ensemble:\n{report_text}")
        
        # Calcular acurácia individual de cada modelo
        individual_accuracies = [correct / individual_total if individual_total > 0 else 0 
                               for correct in individual_correct]
        
        # Criar comparativo
        logger.info("Comparação de acurácias:")
        for i, (model_name, acc) in enumerate(zip(model_names, individual_accuracies)):
            logger.info(f"  {model_name}: {acc:.4f}")
        logger.info(f"  Ensemble: {accuracy:.4f}")
        
        # Média do tempo de inferência
        if ensemble_times:
            avg_time = np.mean(ensemble_times)
            logger.info(f"Tempo médio de inferência do ensemble: {avg_time*1000:.2f} ms por batch")
        
        # Visualizar a matriz de confusão
        if conf_matrix.size > 0:
            plt.figure(figsize=(10, 8))
            display_classes = classes
            if len(classes) != conf_matrix.shape[0]:
                # Usar subset de classes se houver incompatibilidade
                display_classes = [classes[i] for i in used_indices] if 'used_indices' in locals() else []
                if not display_classes:
                    display_classes = [str(i) for i in range(conf_matrix.shape[0])]
            
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                      xticklabels=display_classes, yticklabels=display_classes)
            plt.title('Matriz de Confusão - Ensemble')
            plt.xlabel('Classe Prevista')
            plt.ylabel('Classe Real')
            plt.tight_layout()
            
            # Salvar matriz de confusão
            conf_matrix_path = os.path.join('ensemble_results', 'confusion_matrix_ensemble.png')
            os.makedirs(os.path.dirname(conf_matrix_path), exist_ok=True)
            plt.savefig(conf_matrix_path)
            logger.info(f"Matriz de confusão do ensemble salva em: {conf_matrix_path}")
            plt.close()
            
            # Adicionar ao TensorBoard
            if writer:
                # Adicionar matriz de confusão
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=display_classes, yticklabels=display_classes)
                ax.set_title('Matriz de Confusão - Ensemble')
                ax.set_xlabel('Classe Prevista')
                ax.set_ylabel('Classe Real')
                writer.add_figure("Ensemble/confusion_matrix", fig, 0)
                plt.close(fig)
                
                # Adicionar acurácias comparativas
                for i, (model_name, acc) in enumerate(zip(model_names, individual_accuracies)):
                    writer.add_scalar("Ensemble/individual_accuracy", acc, i)
                    writer.add_text("Ensemble/model_comparison", f"{model_name}: {acc:.4f}", i)
                
                writer.add_scalar("Ensemble/ensemble_accuracy", accuracy, 0)
                
                # Visualizar comparação em barras
                fig, ax = plt.subplots(figsize=(10, 6))
                all_models = model_names + ["Ensemble"]
                all_accs = individual_accuracies + [accuracy]
                
                bars = ax.bar(range(len(all_models)), all_accs, color='skyblue')
                bars[-1].set_color('red')  # Destacar o ensemble
                
                # Adicionar valores nas barras
                for i, v in enumerate(all_accs):
                    ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
                
                ax.set_xticks(range(len(all_models)))
                ax.set_xticklabels(all_models, rotation=45, ha='right')
                ax.set_ylim(0, 1.0)
                ax.set_ylabel('Acurácia')
                ax.set_title('Comparação de Acurácia: Modelos Individuais vs Ensemble')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                writer.add_figure("Ensemble/accuracy_comparison", fig, 0)
                plt.close(fig)
        
        return accuracy, report, conf_matrix, all_preds, all_labels
    
    except Exception as e:
        logger.critical("Erro crítico durante avaliação do ensemble", exc_info=True)
        # Retornar valores padrão/vazios
        return 0, {}, np.array([]), [], []
    
def evaluate_ensemble_with_memory_management(models, data_loaders, classes, model_names=None, writer=None, 
                                            batch_size=None, use_amp=False):
    """
    Avalia um ensemble de modelos com otimizações de memória.
    
    Args:
        models: Lista de modelos PyTorch
        data_loaders: Lista de DataLoaders ou um DataLoader único
        classes: Lista com nomes das classes
        model_names: Lista de nomes dos modelos (opcional)
        writer: SummaryWriter do TensorBoard (opcional)
        batch_size: Tamanho do batch para avaliação (opcional para reduzir)
        use_amp: Se True, usa precisão mista (AMP) para avaliação
        
    Returns:
        tuple: (acurácia, relatório, matriz_confusão, predições, labels)
    """
    logger.info("Iniciando avaliação do ensemble com otimizações de memória")
    
    # Verificar se data_loaders é uma lista ou um DataLoader único
    if not isinstance(data_loaders, list):
        data_loaders = [data_loaders] * len(models)
    
    if len(models) != len(data_loaders):
        raise ValueError(f"Número de modelos ({len(models)}) e dataloaders ({len(data_loaders)}) não correspondem")
    
    # Verificar se model_names é uma lista ou None
    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(models))]
    
    if len(models) != len(model_names):
        raise ValueError(f"Número de modelos ({len(models)}) e nomes ({len(model_names)}) não correspondem")
    
    # Determinar o device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configurar AMP se solicitado
    if use_amp and torch.cuda.is_available():
        logger.info("Usando precisão mista (AMP) para avaliação do ensemble")
        amp_enabled = True
    else:
        amp_enabled = False
    
    # Obter um dataloader representativo para iteração principal
    main_dataloader = data_loaders[0]
    
    # Se batch_size for especificado, criar novo dataloader com batch menor
    if batch_size is not None and batch_size < main_dataloader.batch_size:
        logger.info(f"Reduzindo batch size de {main_dataloader.batch_size} para {batch_size} para avaliação")
        
        # Criar novo dataloader com batch menor
        main_dataloader = DataLoader(
            main_dataloader.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=getattr(main_dataloader, 'pin_memory', False)
        )
    
    # SOLUÇÃO COMPLETA: Rastreamento de índices e predições por modelo
    # Dicionários para rastrear índices e predições de cada modelo
    all_processed_indices = {model_name: set() for model_name in model_names}
    all_model_predictions = {model_name: {} for model_name in model_names}
    
    # Dicionário para armazenar rótulos verdadeiros para cada índice
    all_labels_dict = {}
    
    # Contador global para índices das amostras
    sample_idx_offset = 0
    
    # Processar batches
    for batch_idx, batch_data in enumerate(main_dataloader):
        try:
            # Verificar o formato do batch primeiro
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                inputs, labels = batch_data[0], batch_data[1]
            else:
                logger.error(f"Formato inesperado dos dados no batch {batch_idx}")
                continue
                
            # Calcular índices globais para este batch
            batch_size_actual = labels.size(0)
            batch_indices = list(range(sample_idx_offset, sample_idx_offset + batch_size_actual))
            sample_idx_offset += batch_size_actual
            
            # Armazenar rótulos verdadeiros para cada índice
            for idx, label in zip(batch_indices, labels.cpu().numpy()):
                all_labels_dict[idx] = label
            
            # Liberar memória antes de processar cada batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Processar cada modelo sequencialmente (para economizar memória)
            for i, (model, model_name) in enumerate(zip(models, model_names)):
                try:
                    # Colocar modelo em modo de avaliação
                    model.eval()
                    
                    # Garantir que temos inputs específicos para este modelo
                    # usando o dataloader correspondente se disponível
                    model_inputs = inputs
                    
                    # Se estivermos usando dataloaders específicos por modelo, obter os dados correspondentes
                    if main_dataloader != data_loaders[i]:
                        try:
                            # Tentar encontrar dados no mesmo índice/posição em outro dataloader
                            model_batch_data = None
                            for j, data in enumerate(data_loaders[i]):
                                if j == batch_idx:
                                    model_batch_data = data
                                    break
                                    
                            if model_batch_data is not None and isinstance(model_batch_data, (list, tuple)) and len(model_batch_data) >= 1:
                                model_inputs = model_batch_data[0]
                        except Exception as e:
                            logger.warning(f"Não foi possível obter dados específicos para {model_name}, usando dados padrão: {str(e)}")
                    
                    # Garantir que model_inputs está definido neste ponto
                    if model_inputs is None:
                        logger.error(f"Inputs não definidos para modelo {model_name} no batch {batch_idx}")
                        continue
                    
                    # Mover inputs para o device
                    model_inputs = model_inputs.to(device)
                    
                    # Forward pass com AMP se ativado
                    with torch.no_grad():
                        if amp_enabled:
                            with torch.cuda.amp.autocast():
                                outputs = model(model_inputs)
                        else:
                            outputs = model(model_inputs)
                    
                    # Processar saídas (para lidar com diferentes formatos de saída)
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        outputs = outputs['logits']
                    elif isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Pegar probabilidades
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # Obter predições
                    _, preds = torch.max(probs, 1)
                    
                    # Mover para CPU e converter para numpy
                    probs_np = probs.cpu().numpy()
                    preds_np = preds.cpu().numpy()
                    
                    # Armazenar predições e probabilidades para cada amostra deste modelo
                    for idx, (pred, prob) in zip(batch_indices, zip(preds_np, probs_np)):
                        all_model_predictions[model_name][idx] = (pred, prob)
                        all_processed_indices[model_name].add(idx)
                    
                    # Log do progresso
                    if batch_idx % 10 == 0 and i == 0:
                        logger.info(f"Avaliando batch {batch_idx}/{len(main_dataloader)}")
                    
                    # Liberar memória após processar cada modelo
                    del model_inputs, outputs, probs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Erro ao avaliar modelo {model_name} no batch {batch_idx}: {str(e)}")
                    # Continuar para o próximo modelo, este modelo não terá registros para este batch
            
            # Liberar memória após processar cada batch
            del batch_data, inputs, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            logger.error(f"Erro no processamento do batch {batch_idx}: {str(e)}")
            # Continuar para o próximo batch
            continue
    
    # Após o processamento, encontrar índices comuns a todos os modelos
    if not all_processed_indices or not all(all_processed_indices.values()):
        logger.error("Nenhum índice processado por qualquer modelo. Verificar se os modelos estão funcionando corretamente.")
        return 0, {}, np.array([]), [], []
    
    common_indices = set.intersection(*[indices for indices in all_processed_indices.values()])
    
    # Registrar estatísticas sobre as amostras processadas
    logger.info(f"Estatísticas de processamento:")
    for model_name, indices in all_processed_indices.items():
        logger.info(f"  Modelo {model_name}: {len(indices)} amostras processadas")
    logger.info(f"Amostras processadas por todos os modelos: {len(common_indices)} de {sample_idx_offset} total")
    
    if len(common_indices) == 0:
        logger.error("Nenhuma amostra foi processada por todos os modelos. Não é possível calcular métricas de ensemble.")
        return 0, {}, np.array([]), [], []
    
    # Função auxiliar para obter predição do ensemble para um índice específico
    def get_ensemble_prediction(idx, predictions_dict, model_names):
        # Média das probabilidades de todos os modelos para esta amostra
        all_probs = [predictions_dict[name][idx][1] for name in model_names]
        avg_probs = np.mean(all_probs, axis=0)
        return np.argmax(avg_probs)
    
    # Usar apenas índices comuns para calcular métricas
    all_ensemble_preds = []
    all_labels = []
    
    for idx in sorted(common_indices):
        # Obter predição do ensemble para este índice
        ensemble_pred = get_ensemble_prediction(idx, all_model_predictions, model_names)
        all_ensemble_preds.append(ensemble_pred)
        
        # Obter rótulo verdadeiro para este índice
        true_label = all_labels_dict.get(idx)
        if true_label is not None:
            all_labels.append(true_label)
        else:
            logger.warning(f"Rótulo não encontrado para o índice {idx}, ignorando esta amostra")
    
    # Verificar se ainda temos amostras válidas
    if not all_ensemble_preds or not all_labels:
        logger.error("Nenhuma amostra válida após filtragem. Não é possível calcular métricas.")
        return 0, {}, np.array([]), [], []
    
    # Calcular métricas
    logger.info(f"Calculando métricas com {len(all_ensemble_preds)} amostras válidas")
    
    # Converter para numpy arrays
    all_ensemble_preds = np.array(all_ensemble_preds)
    all_labels = np.array(all_labels)
    
    # Calcular acurácia
    accuracy = accuracy_score(all_labels, all_ensemble_preds)
    
    # Calcular relatório detalhado
    try:
        report = classification_report(all_labels, all_ensemble_preds, target_names=classes, output_dict=True)
        
        # Calcular matriz de confusão
        conf_matrix = confusion_matrix(all_labels, all_ensemble_preds)
    except Exception as e:
        logger.error(f"Erro ao calcular métricas detalhadas: {str(e)}")
        report = {}
        conf_matrix = np.array([])
    
    # Visualizar matriz de confusão
    if writer and conf_matrix.size > 0:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            ax.set_title(f'Matriz de Confusão - Ensemble')
            ax.set_xlabel('Predito')
            ax.set_ylabel('Real')
            writer.add_figure("Ensemble/confusion_matrix", fig, 0)
            plt.close(fig)
            
            # Adicionar acurácia ao TensorBoard
            writer.add_scalar("Ensemble/accuracy", accuracy, 0)
            
            # Adicionar métricas por classe ao TensorBoard
            for i, class_name in enumerate(classes):
                if class_name in report:
                    writer.add_scalar(f"Ensemble/precision_{class_name}", report[class_name]['precision'], 0)
                    writer.add_scalar(f"Ensemble/recall_{class_name}", report[class_name]['recall'], 0)
                    writer.add_scalar(f"Ensemble/f1_{class_name}", report[class_name]['f1-score'], 0)
        except Exception as e:
            logger.error(f"Erro ao visualizar matriz de confusão: {str(e)}")
    
    logger.info(f"Ensemble avaliado com sucesso. Acurácia: {accuracy:.4f}")
    
    return accuracy, report, conf_matrix, all_ensemble_preds, all_labels

def ensemble_predict_with_batches(models, dataset, batch_size=16, num_workers=2, device=None, use_amp=False):
    """
    Faz previsões com o ensemble em lotes, gerenciando memória cuidadosamente.
    
    Args:
        models: Lista de modelos PyTorch
        dataset: Dataset PyTorch
        batch_size: Tamanho do batch (menor para economizar memória)
        num_workers: Número de workers para o DataLoader
        device: Dispositivo para computação (None = auto-detect)
        use_amp: Se True, usa precisão mista automática
        
    Returns:
        tuple: (predições do ensemble, probabilidades)
    """
    logger.info("Iniciando predição do ensemble com gerenciamento de batches")
    
    # Determinar device
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Verificar se modelos estão na lista
    if not isinstance(models, list):
        models = [models]
    
    # Criar DataLoader com batch size pequeno para gerenciamento de memória
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Colocar modelos em modo de avaliação
    for model in models:
        model.eval()
    
    # Listas para armazenar resultados
    all_ensemble_preds = []
    all_ensemble_probs = []
    
    # Inferência batch por batch
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            # Liberar memória antes de cada batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            # Log de progresso
            if batch_idx % 10 == 0:
                logger.info(f"Processando batch {batch_idx}/{len(data_loader)}")
            
            # Mover inputs para o device
            inputs = inputs.to(device)
            
            # Lista para armazenar probabilidades de cada modelo
            batch_probs = []
            
            # Processar cada modelo do ensemble
            for model_idx, model in enumerate(models):
                try:
                    # Forward pass com AMP se ativado
                    if use_amp and device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                    else:
                        outputs = model(inputs)
                    
                    # Processar saídas (para lidar com diferentes formatos)
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        outputs = outputs['logits']
                    elif isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Obter probabilidades
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # Mover para CPU como numpy array
                    probs_np = probs.cpu().numpy()
                    
                    # Adicionar às probabilidades do batch
                    batch_probs.append(probs_np)
                    
                    # Liberar memória após cada modelo
                    del outputs, probs
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Erro no modelo {model_idx} para o batch {batch_idx}: {str(e)}")
                    # Criar array de zeros como fallback
                    if len(batch_probs) > 0:
                        zero_probs = np.zeros_like(batch_probs[0])
                    else:
                        # Obter o número de classes do modelo
                        num_classes = model.fc.out_features if hasattr(model, 'fc') else \
                                    model.classifier[-1].out_features if hasattr(model, 'classifier') else \
                                    model.head.out_features if hasattr(model, 'head') else 2
                        zero_probs = np.zeros((inputs.size(0), num_classes))
                    batch_probs.append(zero_probs)
            
            # Calcular média de probabilidades do ensemble
            if batch_probs:
                batch_ensemble_probs = np.mean(batch_probs, axis=0)
                batch_preds = np.argmax(batch_ensemble_probs, axis=1)
                
                # Adicionar aos resultados globais
                all_ensemble_probs.extend(batch_ensemble_probs)
                all_ensemble_preds.extend(batch_preds)
            
            # Liberar memória explicitamente
            del inputs, batch_probs
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
    
    # Converter para numpy arrays
    all_ensemble_preds = np.array(all_ensemble_preds)
    all_ensemble_probs = np.array(all_ensemble_probs)
    
    return all_ensemble_preds, all_ensemble_probs

def classify_images_with_ensemble(ensemble_models, image_paths, preprocessing_transform, classes, batch_size=8):
    """
    Classifica imagens usando um ensemble de modelos, com gerenciamento de memória otimizado.
    
    Args:
        ensemble_models: Lista de modelos treinados
        image_paths: Lista de caminhos para as imagens a classificar
        preprocessing_transform: Transformações a aplicar às imagens
        classes: Lista de nomes das classes
        batch_size: Tamanho do batch para processamento
        
    Returns:
        DataFrame com resultados da classificação
    """
    import pandas as pd
    from PIL import Image
    from torch.utils.data import Dataset
    
    logger.info(f"Classificando {len(image_paths)} imagens com ensemble de {len(ensemble_models)} modelos")
    
    # Dataset personalizado para carregar as imagens
    class ImageDataset(Dataset):
        def __init__(self, image_paths, transform):
            self.image_paths = image_paths
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, 0  # 0 é um label dummy
            except Exception as e:
                logger.error(f"Erro ao carregar imagem {img_path}: {str(e)}")
                # Criar tensor vazio como fallback
                return torch.zeros((3, 224, 224)), 0
    
    # Criar dataset e dataloader
    image_dataset = ImageDataset(image_paths, preprocessing_transform)
    
    # Fazer predições com o ensemble
    predictions, probabilities = ensemble_predict_with_batches(
        ensemble_models, 
        image_dataset, 
        batch_size=batch_size,
        use_amp=True
    )
    
    # Criar DataFrame com resultados
    results = []
    for i, (img_path, pred) in enumerate(zip(image_paths, predictions)):
        # Obter o nome da imagem do caminho
        img_name = os.path.basename(img_path)
        
        # Obter classe predita e confiança
        pred_class = classes[pred]
        confidence = probabilities[i][pred]
        
        # Adicionar segundo e terceiro lugares para completude
        sorted_indices = np.argsort(probabilities[i])[::-1]
        second_class = classes[sorted_indices[1]]
        second_conf = probabilities[i][sorted_indices[1]]
        third_class = classes[sorted_indices[2]]
        third_conf = probabilities[i][sorted_indices[2]]
        
        # Adicionar resultado
        results.append({
            'image': img_name,
            'path': img_path,
            'predicted_class': pred_class,
            'confidence': confidence,
            'second_class': second_class,
            'second_confidence': second_conf,
            'third_class': third_class,
            'third_confidence': third_conf
        })
    
    # Limpar memória
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Criar DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def create_weighted_ensemble(models, weights=None):
    """
    Cria um modelo de ensemble ponderado que combina as previsões 
    de múltiplos modelos com pesos específicos.
    
    Args:
        models: Lista de modelos PyTorch
        weights: Lista de pesos para cada modelo (opcional)
        
    Returns:
        Função que aceita um tensor de entrada e retorna a previsão do ensemble
    """
    if weights is None:
        weights = [1.0] * len(models)
    
    # Normalizar pesos
    weights = np.array(weights) / sum(weights)
    
    def ensemble_predict(x):
        # Colocar modelos em modo de avaliação
        for model in models:
            model.eval()
        
        # Obter previsões de cada modelo
        probs_list = []
        with torch.no_grad():
            for i, model in enumerate(models):
                output = model(x)
                probs = torch.nn.functional.softmax(output, dim=1)
                probs_list.append(probs * weights[i])
        
        # Combinar probabilidades ponderadas
        ensemble_probs = sum(probs_list)
        return ensemble_probs
    
    return ensemble_predict

def save_ensemble_metadata(models, model_names, weights=None, save_path='models/ensemble_metadata.pt'):
    """
    Salva metadados do ensemble para recriação posterior.
    
    Args:
        models: Lista de modelos PyTorch
        model_names: Lista de nomes dos modelos
        weights: Lista de pesos para cada modelo (opcional)
        save_path: Caminho para salvar os metadados
    """
    if weights is None:
        weights = [1.0] * len(models)
    
    metadata = {
        'model_paths': [f"models/{name}_optimized.pth" for name in model_names],
        'model_names': model_names,
        'weights': weights,
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(metadata, save_path)
    logger.info(f"Metadados do ensemble salvos em {save_path}")

def create_memory_efficient_ensemble(model_creators, model_paths, weights=None):
    """
    Cria um ensemble que carrega modelos sob demanda para economizar memória.
    
    Args:
        model_creators: Lista de funções que criam modelos
        model_paths: Lista de caminhos para arquivos de pesos de modelos
        weights: Lista de pesos para cada modelo (opcional)
    
    Returns:
        Objeto com método predict que executa cada modelo sequencialmente,
        liberando memória entre eles
    """
    if weights is None:
        weights = [1.0] * len(model_creators)
    
    # Normalizar pesos
    weights = np.array(weights) / sum(weights)
    
    class MemoryEfficientEnsemble:
        def __init__(self, model_creators, model_paths, weights):
            self.model_creators = model_creators
            self.model_paths = model_paths
            self.weights = weights
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        def predict(self, inputs, batch_size=4, use_amp=True):
            """
            Faz predição com ensemble, carregando cada modelo sequencialmente.
            
            Args:
                inputs: Tensor de entrada ou lista de tensores
                batch_size: Tamanho do batch para processamento em partes
                use_amp: Se True, usa precisão mista automática
                
            Returns:
                tuple: (predições do ensemble, probabilidades)
            """
            # Verificar se inputs é um DataLoader, um Dataset ou um tensor
            if isinstance(inputs, DataLoader):
                return self._predict_dataloader(inputs, use_amp)
            elif hasattr(inputs, '__getitem__') and hasattr(inputs, '__len__'):
                # É um Dataset ou uma lista
                return self._predict_dataset(inputs, batch_size, use_amp)
            else:
                # É um tensor único
                return self._predict_single(inputs, use_amp)
        
        def _predict_dataloader(self, dataloader, use_amp):
            """Predição usando um DataLoader"""
            all_probs = []
            all_labels = []
            
            # Para cada batch no dataloader
            for batch_idx, (batch_inputs, batch_labels) in enumerate(dataloader):
                # Processar cada batch
                if batch_idx % 10 == 0:
                    logger.info(f"Processando batch {batch_idx}/{len(dataloader)}")
                
                # Obter probabilidades para esse batch
                batch_probs = self._predict_batch_sequential(batch_inputs, use_amp)
                all_probs.append(batch_probs)
                all_labels.extend(batch_labels.numpy())
                
                # Liberar memória
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Concatenar todos os resultados
            all_probs = np.concatenate(all_probs, axis=0)
            all_preds = np.argmax(all_probs, axis=1)
            
            return all_preds, all_probs, all_labels
        
        def _predict_dataset(self, dataset, batch_size, use_amp):
            """Predição usando um Dataset, processando em batches"""
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            return self._predict_dataloader(dataloader, use_amp)
        
        def _predict_single(self, inputs, use_amp):
            """Predição para um único tensor de entrada"""
            # Processar como um batch de tamanho 1
            probs = self._predict_batch_sequential(inputs.unsqueeze(0), use_amp)
            return np.argmax(probs, axis=1)[0], probs[0]
        
        def _predict_batch_sequential(self, inputs, use_amp):
            """
            Processa um batch com cada modelo sequencialmente para economizar memória.
            """
            batch_size = inputs.size(0)
            accumulated_probs = None
            
            # Processar cada modelo sequencialmente
            for i, (creator_fn, model_path, weight) in enumerate(zip(
                self.model_creators, self.model_paths, self.weights)):
                try:
                    # Limpar memória antes de carregar o modelo
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Criar o modelo e carregar pesos
                    logger.debug(f"Carregando modelo {i+1}/{len(self.model_creators)}")
                    model = creator_fn()
                    model.load_state_dict(torch.load(model_path))
                    model = model.to(self.device).eval()
                    
                    # Mover inputs para o device
                    inputs_device = inputs.to(self.device)
                    
                    # Forward pass com/sem AMP
                    with torch.no_grad():
                        if use_amp and torch.cuda.is_available():
                            with torch.cuda.amp.autocast():
                                outputs = model(inputs_device)
                        else:
                            outputs = model(inputs_device)
                    
                    # Processar saídas
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        outputs = outputs['logits']
                    elif isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Obter probabilidades e aplicar peso
                    probs = torch.nn.functional.softmax(outputs, dim=1) * weight
                    
                    # Acumular probabilidades
                    if accumulated_probs is None:
                        accumulated_probs = probs.cpu().numpy()
                    else:
                        accumulated_probs += probs.cpu().numpy()
                    
                    # Descartar o modelo para liberar memória
                    del model, outputs, probs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                except Exception as e:
                    logger.error(f"Erro ao processar modelo {i}: {str(e)}")
                    # Se não conseguir processar este modelo, continuar com os outros
                    continue
            
            # Se nenhum modelo foi processado com sucesso
            if accumulated_probs is None:
                # Determinar número de classes
                dummy_model = self.model_creators[0]()
                if hasattr(dummy_model, 'fc'):
                    num_classes = dummy_model.fc.out_features
                elif hasattr(dummy_model, 'classifier') and hasattr(dummy_model.classifier[-1], 'out_features'):
                    num_classes = dummy_model.classifier[-1].out_features
                else:
                    num_classes = 1000  # Fallback para 1000 classes
                del dummy_model
                
                # Retornar probabilidades uniformes
                return np.ones((batch_size, num_classes)) / num_classes
            
            return accumulated_probs
    
    return MemoryEfficientEnsemble(model_creators, model_paths, weights)

def find_optimal_weights(models, val_loader, classes, initial_weights=None, num_iterations=50):
    """
    Encontra os pesos ótimos para cada modelo no ensemble usando um algoritmo iterativo.
    
    Args:
        models: Lista de modelos PyTorch
        val_loader: DataLoader para conjunto de validação
        classes: Lista de nomes das classes
        initial_weights: Pesos iniciais (opcional)
        num_iterations: Número de iterações para otimização
        
    Returns:
        Lista de pesos otimizados
    """
    if initial_weights is None:
        # Começar com pesos iguais
        weights = np.ones(len(models)) / len(models)
    else:
        weights = np.array(initial_weights)
        weights = weights / np.sum(weights)  # Normalizar
    
    device = next(models[0].parameters()).device
    
    # Colocar modelos em modo de avaliação
    for model in models:
        model.eval()
    
    best_accuracy = 0.0
    best_weights = weights.copy()
    
    logger.info("Iniciando otimização de pesos para ensemble...")
    
    # Para cada modelo, calcular acurácia individual
    individual_accuracies = []
    
    with torch.no_grad():
        for i, model in enumerate(models):
            correct = 0
            total = 0
            
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                
                # Processar saídas
                if isinstance(outputs, dict) and 'logits' in outputs:
                    outputs = outputs['logits']
                elif isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            acc = correct / total
            individual_accuracies.append(acc)
            logger.info(f"Modelo {i}: acurácia = {acc:.4f}")
    
    # Usar acurácias individuais como pesos iniciais
    weights = np.array(individual_accuracies)
    weights = weights / np.sum(weights)  # Normalizar
    
    logger.info(f"Pesos iniciais baseados em acurácias individuais: {weights}")
    
    # Armazenar todas as previsões para economizar computação
    all_model_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            batch_probs = []
            
            # Obter previsões de cada modelo
            for model in models:
                outputs = model(inputs)
                
                # Processar saídas
                if isinstance(outputs, dict) and 'logits' in outputs:
                    outputs = outputs['logits']
                elif isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probs = torch.nn.functional.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())
            
            all_model_probs.append(batch_probs)
            all_labels.append(labels.numpy())
    
    # Otimizar pesos
    for iteration in range(num_iterations):
        # Calcular acurácia com pesos atuais
        correct = 0
        total = 0
        
        for batch_idx, batch_probs in enumerate(all_model_probs):
            # Combinar probabilidades do batch com pesos atuais
            batch_ensemble_probs = np.zeros_like(batch_probs[0])
            for i, model_probs in enumerate(batch_probs):
                batch_ensemble_probs += model_probs * weights[i]
            
            # Obter predições
            batch_preds = np.argmax(batch_ensemble_probs, axis=1)
            
            # Comparar com ground truth
            correct += np.sum(batch_preds == all_labels[batch_idx])
            total += len(all_labels[batch_idx])
        
        accuracy = correct / total
        
        # Atualizar melhores pesos se necessário
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = weights.copy()
            logger.info(f"Iteração {iteration+1}: Nova melhor acurácia = {best_accuracy:.4f}, Pesos = {best_weights}")
        
        # Atualizar pesos para próxima iteração
        # Usar algoritmo hill climbing simples
        for i in range(len(weights)):
            # Testar aumentar o peso do modelo i
            new_weights = weights.copy()
            new_weights[i] += 0.05
            new_weights = new_weights / np.sum(new_weights)  # Normalizar
            
            # Avaliar com novos pesos
            correct = 0
            total = 0
            
            for batch_idx, batch_probs in enumerate(all_model_probs):
                # Combinar probabilidades com novos pesos
                batch_ensemble_probs = np.zeros_like(batch_probs[0])
                for j, model_probs in enumerate(batch_probs):
                    batch_ensemble_probs += model_probs * new_weights[j]
                
                # Obter predições
                batch_preds = np.argmax(batch_ensemble_probs, axis=1)
                
                # Comparar com ground truth
                correct += np.sum(batch_preds == all_labels[batch_idx])
                total += len(all_labels[batch_idx])
            
            new_accuracy = correct / total
            
            # Se melhorou, atualizar pesos
            if new_accuracy > accuracy:
                weights = new_weights
                accuracy = new_accuracy
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = weights.copy()
                    logger.info(f"Iteração {iteration+1}: Nova melhor acurácia = {best_accuracy:.4f}, Pesos = {best_weights}")
    
    logger.info(f"Otimização concluída. Melhor acurácia = {best_accuracy:.4f}")
    logger.info(f"Pesos ótimos: {best_weights}")
    
    return best_weights.tolist()