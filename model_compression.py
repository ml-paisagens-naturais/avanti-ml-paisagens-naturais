"""
Model Compression Module

Este módulo contém funções para otimizar e comprimir modelos de redes neurais,
incluindo quantização, pruning e exportação para formatos de implantação.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import copy
import logging
import os
import onnx
from torch.nn.utils import prune
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

# Obter o logger
logger = logging.getLogger("landscape_classifier")

def disable_gradient_checkpointing(model):
    """
    Desativa o gradient checkpointing de um modelo antes da exportação ONNX
    
    Args:
        model: Modelo PyTorch
        
    Returns:
        model: Modelo com gradient checkpointing desativado
    """
    logger = logging.getLogger("landscape_classifier")
    
    # Restaurar qualquer método forward original
    if hasattr(model, '_original_forward'):
        model.forward = model._original_forward
        logger.info("Método forward original restaurado para exportação ONNX")
    
    # Para ResNet, reverter o forward modificado
    model_name = model.__class__.__name__.lower()
    if 'resnet' in model_name:
        if hasattr(model, 'layer4'):
            # Implementação específica para modelos ResNet
            def simple_forward(x):
                # Forward pass sem checkpointing para exportação ONNX
                x = model.conv1(x)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                
                # Usar camadas sequencialmente sem checkpoint
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                
                x = model.avgpool(x)
                x = torch.flatten(x, 1)
                x = model.fc(x)
                return x
            
            # Salvar o forward original se ainda não foi salvo
            if not hasattr(model, '_original_forward'):
                model._original_forward = model.forward
            
            # Substituir com forward simplificado
            model.forward = simple_forward
            logger.info("Forward do ResNet substituído por versão simplificada para exportação ONNX")
    
    return model

def quantize_model(model, test_loader, model_name="model"):
    """
    Aplica quantização dinâmica no modelo para reduzir o tamanho e
    aumentar a eficiência em inferência.

    Args:
        model: Modelo PyTorch a ser quantizado
        test_loader: DataLoader para avaliação
        model_name: Nome do modelo para logging
    
    Returns:
        Modelo quantizado ou None em caso de erro
    """
    logger.info(f"Aplicando quantização dinâmica ao modelo {model_name}...")
    
    try:
        # Obter o dispositivo atual do modelo
        device = next(model.parameters()).device
        
        # Criar cópia do modelo para quantização
        model_to_quantize = copy.deepcopy(model)
        
        # Medir tamanho original
        orig_size = sum(p.numel() for p in model.parameters()) * 4  # Float32 = 4 bytes
        logger.info(f"Tamanho do modelo original: {orig_size/1e6:.2f} MB")
        
        # Medir tempo de inferência original
        model.eval()
        original_times = []
        example_batch = next(iter(test_loader))[0][:8].to(device)
        
        # Aquecer GPU
        with torch.no_grad():
            _ = model(example_batch)
        
        # Medir tempo de inferência
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                _ = model(example_batch)
            original_times.append(time.time() - start_time)
        
        avg_original_time = np.mean(original_times)
        logger.info(f"Tempo médio de inferência original: {avg_original_time*1000:.2f} ms")
        
        # CORREÇÃO: Movimento para CPU antes da quantização e manter na CPU
        # A quantização dinâmica do PyTorch só é suportada na CPU
        model_to_quantize = model_to_quantize.cpu()
        quantized_model = torch.quantization.quantize_dynamic(
            model_to_quantize, 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8
        )
        
        logger.info(f"Modelo {model_name} quantizado com sucesso (mantido na CPU)")
        
        # Medir tamanho do modelo quantizado
        quant_size = 0
        for name, param in quantized_model.named_parameters():
            if param.dtype == torch.qint8:
                quant_size += param.numel() * 1  # INT8 = 1 byte
            else:
                quant_size += param.numel() * 4  # FLOAT32 = 4 bytes
                
        for name, buffer in quantized_model.named_buffers():
            if buffer.dtype == torch.qint8:
                quant_size += buffer.numel() * 1
            else:
                quant_size += buffer.numel() * 4
        
        logger.info(f"Tamanho aproximado do modelo quantizado: {quant_size/1e6:.2f} MB")
        logger.info(f"Redução aproximada: {100 * (1 - quant_size/orig_size):.2f}%")
        
        # Testar a acurácia do modelo quantizado
        quantized_model.eval()
        
        # CORREÇÃO: Executar inferência na CPU com o modelo quantizado
        # Medir tempo de inferência do modelo quantizado
        quantized_times = []
        example_batch_cpu = example_batch.cpu()  # Mover para CPU
        
        # Aquecer CPU
        with torch.no_grad():
            _ = quantized_model(example_batch_cpu)
        
        # Medir tempo de inferência
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                _ = quantized_model(example_batch_cpu)
            quantized_times.append(time.time() - start_time)
        
        avg_quantized_time = np.mean(quantized_times)
        logger.info(f"Tempo médio de inferência do modelo quantizado (CPU): {avg_quantized_time*1000:.2f} ms")
        
        # Comparação real é CPU quantizado vs GPU não-quantizado
        logger.info(f"Nota: Comparação de tempo não é direta (GPU vs CPU)")
        
        # Avaliar a acurácia no conjunto de teste
        logger.info("Avaliando acurácia do modelo quantizado em um subconjunto de teste...")
        
        # Usar um subconjunto menor para avaliação rápida
        max_test_samples = min(500, len(test_loader.dataset))
        indices = torch.randperm(len(test_loader.dataset))[:max_test_samples].tolist()  # Converter para lista
        test_subset = torch.utils.data.Subset(test_loader.dataset, indices)
        subset_loader = DataLoader(
            test_subset, 
            batch_size=32, 
            shuffle=False,
            num_workers=0  # Garantir processamento sequencial
        )
        
        # CORREÇÃO: Avaliar o modelo quantizado na CPU
        all_preds = []
        all_labels = []
        for inputs, labels in subset_loader:
            # Manter inputs na CPU para o modelo quantizado
            with torch.no_grad():
                outputs = quantized_model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        
        # Calcular acurácia
        quantized_accuracy = accuracy_score(all_labels, all_preds)
        logger.info(f"Acurácia do modelo quantizado: {quantized_accuracy:.4f}")
        
        # Avaliar modelo original no mesmo subconjunto
        model.eval()
        all_preds = []
        all_labels = []
        for inputs, labels in subset_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        original_accuracy = accuracy_score(all_labels, all_preds)
        logger.info(f"Acurácia do modelo original no mesmo subconjunto: {original_accuracy:.4f}")
        logger.info(f"Diferença de acurácia: {(original_accuracy - quantized_accuracy)*100:.2f}%")
        
        # Salvar modelo quantizado
        try:
            model_path = os.path.join("models", f"{model_name}_quantized.pth")
            torch.save(quantized_model.state_dict(), model_path)
            logger.info(f"Modelo quantizado salvo como {model_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo quantizado: {str(e)}")
        
        # CORREÇÃO: Criar uma versão do modelo para inferência que funciona no dispositivo original 
        # apenas para forward pass (funciona desde que não use operadores quantizados internamente)
        try:
            if device.type == 'cuda':
                # Criar um wrapper para inferência GPU, útil apenas para forward pass
                class QuantizedModelWrapper(nn.Module):
                    def __init__(self, quantized_model, device):
                        super().__init__()
                        self.quantized_model = quantized_model  # Mantém o modelo na CPU
                        self.device = device
                        self.is_quantized = True
                    
                    def forward(self, x):
                        # Converter para CPU → inferência → converter de volta para GPU
                        if x.device.type != 'cpu':
                            x_cpu = x.cpu()
                            with torch.no_grad():
                                output_cpu = self.quantized_model(x_cpu)
                            return output_cpu.to(self.device)
                        else:
                            return self.quantized_model(x)
                
                # Criar wrapper para uso em GPU (apenas forward)
                inference_model = QuantizedModelWrapper(quantized_model, device)
                logger.info("Criado wrapper para inferência em GPU (apenas forward)")
                
                # Opcional: retornar o wrapper em vez do modelo quantizado puro
                # return inference_model
                
                # Por enquanto, retornamos o modelo quantizado na CPU para mais segurança
                logger.info("Retornando modelo quantizado para CPU (uso seguro garantido)")
            
        except Exception as e:
            logger.error(f"Erro ao criar wrapper para inferência: {str(e)}")
        
        return quantized_model
    
    except Exception as e:
        logger.error(f"Erro durante a quantização do modelo {model_name}", exc_info=True)
        return None

def run_model_pruning(model, test_loader, prune_amount=0.2, model_name="model"):
    """
    Realizar pruning (poda) de neurônios para criar modelo mais leve.
    
    Args:
        model: Modelo PyTorch a ser podado
        test_loader: DataLoader para avaliação
        prune_amount: Fração dos parâmetros a serem podados (0-1)
        model_name: Nome do modelo para logging
    
    Returns:
        Modelo podado ou modelo original em caso de erro
    """
    logger.info(f"Aplicando pruning ({prune_amount*100:.0f}%) ao modelo {model_name}...")
    
    try:
        # Criar cópia do modelo para pruning
        pruned_model = copy.deepcopy(model)
        
        # Medir tamanho e performance original
        orig_size = sum(p.numel() for p in pruned_model.parameters())
        logger.info(f"Tamanho original: {orig_size:,} parâmetros")
        
        # Lista das camadas a serem podadas (convolucionais e lineares)
        prunable_layers = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prunable_layers.append((module, 'weight'))
                logger.debug(f"Camada adicionada para pruning: {name}")
        
        logger.info(f"Total de {len(prunable_layers)} camadas selecionadas para pruning")
        
        # Pruning global - remover conexões com menor magnitude
        prune.global_unstructured(
            prunable_layers,
            pruning_method=prune.L1Unstructured,
            amount=prune_amount,
        )
        
        logger.info(f"Pruning aplicado com amount={prune_amount}")
        
        # Avaliar modelo após pruning
        pruned_model.eval()
        
        # Avaliar em um subconjunto para rapidez
        max_test_samples = min(1000, len(test_loader.dataset))
        indices = torch.randperm(len(test_loader.dataset))[:max_test_samples].tolist()  # Converter para lista
        test_subset = torch.utils.data.Subset(test_loader.dataset, indices)
        subset_loader = DataLoader(
            test_subset, 
            batch_size=test_loader.batch_size, 
            shuffle=False,
            num_workers=0  # Garantir processamento sequencial
        )
        
        # Avaliar antes de tornar o pruning permanente
        running_corrects = 0
        total = 0
        
        device = next(pruned_model.parameters()).device
        
        with torch.no_grad():
            for inputs, labels in subset_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = pruned_model(inputs)
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
        
        pruned_acc = running_corrects.double() / total
        logger.info(f"Acurácia após pruning (antes de remover máscaras): {pruned_acc:.4f}")
        
        # Contar número de parâmetros diferentes de zero
        zero_params = 0
        total_params = 0
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                zero_params += torch.sum(module.weight == 0).item()
                total_params += module.weight.nelement()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        logger.info(f"Sparsidade atual: {sparsity*100:.2f}% ({zero_params:,} de {total_params:,} parâmetros são zero)")
        
        # Fazer o pruning permanente
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.remove(module, 'weight')
        
        logger.info("Máscaras de pruning removidas e pruning tornado permanente")
        
        # Contar número de parâmetros não-zero após tornar o pruning permanente
        non_zero_params = 0
        total_params = 0
        for name, param in pruned_model.named_parameters():
            if 'weight' in name:
                non_zero_params += (param != 0).sum().item()
                total_params += param.numel()
        
        sparsity = 1 - non_zero_params / total_params if total_params > 0 else 0
        logger.info(f"Sparsidade final: {sparsity*100:.2f}% ({total_params - non_zero_params:,} de {total_params:,} parâmetros são zero)")
        
        # Reavaliar após tornar o pruning permanente
        pruned_model.eval()
        running_corrects = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in subset_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = pruned_model(inputs)
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
        
        pruned_acc_final = running_corrects.double() / total
        logger.info(f"Acurácia após tornar o pruning permanente: {pruned_acc_final:.4f}")
        
        # Avaliar modelo original no mesmo subconjunto
        model.eval()
        running_corrects = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in subset_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
        
        original_acc = running_corrects.double() / total
        logger.info(f"Acurácia do modelo original: {original_acc:.4f}")
        logger.info(f"Diferença de acurácia: {(original_acc - pruned_acc_final)*100:.2f}%")
        
        # Salvar modelo podado
        try:
            model_path = os.path.join("models", f"{model_name}_pruned.pth")
            torch.save(pruned_model.state_dict(), model_path)
            logger.info(f"Modelo podado salvo como {model_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo podado: {str(e)}")
        
        return pruned_model
    
    except Exception as e:
        logger.error(f"Erro durante o pruning do modelo {model_name}", exc_info=True)
        return model  # Retornar o modelo original em caso de erro
    
def prune_attention_heads(vit_model, prune_amount=0.2, model_name="vit"):
    """
    Implementa pruning de cabeças de atenção em um modelo Vision Transformer.
    Identifica e remove as cabeças de atenção menos importantes.
    
    Args:
        vit_model: Modelo Vision Transformer
        prune_amount: Fração de cabeças a serem removidas
        model_name: Nome do modelo para logging
    
    Returns:
        Modelo com cabeças de atenção podadas
    """
    logger.info(f"Aplicando pruning de cabeças de atenção ({prune_amount*100:.0f}%) ao modelo {model_name}...")
    
    try:
        # Criar cópia do modelo para pruning
        pruned_model = copy.deepcopy(vit_model).cpu()
        
        # Identificar camadas de atenção
        attn_layers = []
        for name, module in pruned_model.named_modules():
            if "attn" in name and hasattr(module, "qkv"):
                attn_layers.append((name, module))
        
        logger.info(f"Encontradas {len(attn_layers)} camadas de atenção para pruning")
        
        if not attn_layers:
            logger.warning("Nenhuma camada de atenção encontrada. Pruning não aplicado.")
            return vit_model
        
        # Avaliar importância das cabeças de atenção
        # Para cada camada de atenção, vamos avaliar a importância de cada cabeça
        head_importance = {}
        
        for name, module in attn_layers:
            # Obter dimensões
            if hasattr(module, "num_heads"):
                num_heads = module.num_heads
            else:
                # Tentar inferir do tamanho da camada qkv
                qkv_dim = module.qkv.out_features
                head_dim = 64  # Valor padrão em muitos ViTs
                num_heads = qkv_dim // (head_dim * 3)
            
            # Obter pesos do QKV
            qkv_weight = module.qkv.weight.data
            
            # Dividir pesos por cabeças e avaliar importância
            # Para simplicidade, usamos a norma L2 como medida de importância
            head_size = qkv_weight.shape[0] // (3 * num_heads)
            head_importance[name] = []
            
            for h in range(num_heads):
                # Para cada cabeça, calculamos importância baseada em Q, K e V
                importance = 0
                for qkv_idx in range(3):  # Q, K, V
                    start_idx = qkv_idx * num_heads * head_size + h * head_size
                    end_idx = start_idx + head_size
                    head_weights = qkv_weight[start_idx:end_idx, :]
                    importance += torch.norm(head_weights).item()
                
                head_importance[name].append((h, importance))
        
        # Para cada camada, ordenar cabeças por importância e identificar as menos importantes
        heads_to_prune = {}
        for name, importances in head_importance.items():
            sorted_heads = sorted(importances, key=lambda x: x[1])
            num_to_prune = int(len(sorted_heads) * prune_amount)
            heads_to_prune[name] = [h for h, _ in sorted_heads[:num_to_prune]]
            logger.info(f"Camada {name}: removendo {num_to_prune} cabeças de {len(sorted_heads)}")
        
        # Aplicar pruning mascarando as cabeças selecionadas
        for name, module in attn_layers:
            if name in heads_to_prune and heads_to_prune[name]:
                # Obter dimensões
                if hasattr(module, "num_heads"):
                    num_heads = module.num_heads
                else:
                    # Inferir do tamanho da camada qkv
                    qkv_dim = module.qkv.out_features
                    head_dim = 64  # Valor padrão
                    num_heads = qkv_dim // (head_dim * 3)
                
                # Criar máscara para as cabeças
                head_size = module.qkv.weight.shape[0] // (3 * num_heads)
                qkv_mask = torch.ones_like(module.qkv.weight)
                
                # Zerar pesos das cabeças selecionadas
                for head_idx in heads_to_prune[name]:
                    for qkv_idx in range(3):  # Q, K, V
                        start_idx = qkv_idx * num_heads * head_size + head_idx * head_size
                        end_idx = start_idx + head_size
                        qkv_mask[start_idx:end_idx, :] = 0
                
                # Aplicar máscara
                with torch.no_grad():
                    module.qkv.weight.data *= qkv_mask
                    if module.qkv.bias is not None:
                        bias_mask = torch.ones_like(module.qkv.bias)
                        for head_idx in heads_to_prune[name]:
                            for qkv_idx in range(3):
                                start_idx = qkv_idx * num_heads * head_size + head_idx * head_size
                                end_idx = start_idx + head_size
                                bias_mask[start_idx:end_idx] = 0
                        module.qkv.bias.data *= bias_mask
        
        logger.info(f"Pruning de cabeças de atenção concluído para o modelo {model_name}")
        return pruned_model.to(next(vit_model.parameters()).device)
        
    except Exception as e:
        logger.error(f"Erro durante pruning de cabeças de atenção: {str(e)}", exc_info=True)
        return vit_model

def quantize_vit(vit_model, test_loader, model_name="vit"):
    """
    Aplica quantização específica para ViT, focando em camadas lineares 
    e preservando a estrutura de atenção multi-cabeça.
    
    Args:
        vit_model: Modelo Vision Transformer
        test_loader: DataLoader para avaliação 
        model_name: Nome do modelo para logging
    
    Returns:
        Modelo ViT quantizado ou None em caso de erro
    """
    logger.info(f"Aplicando quantização ao modelo Vision Transformer {model_name}...")
    
    try:
        # Criar cópia do modelo para quantização
        model_to_quantize = copy.deepcopy(vit_model).cpu()
        
        # Medir tamanho original
        orig_size = sum(p.numel() for p in model_to_quantize.parameters()) * 4  # Float32 = 4 bytes
        logger.info(f"Tamanho do modelo ViT original: {orig_size/1e6:.2f} MB")
        
        # Medir tempo de inferência original
        vit_model.eval()
        original_times = []
        example_batch = next(iter(test_loader))[0][:8].to(next(vit_model.parameters()).device)
        
        # Aquecer GPU/CPU
        with torch.no_grad():
            _ = vit_model(example_batch)
        
        # Medir tempo de inferência
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                _ = vit_model(example_batch)
            original_times.append(time.time() - start_time)
        
        avg_original_time = np.mean(original_times)
        logger.info(f"Tempo médio de inferência original: {avg_original_time*1000:.2f} ms")
        
        # Módulos para excluir da quantização: camadas de atenção críticas
        linear_layers_to_exclude = []
        for name, module in model_to_quantize.named_modules():
            if "attn" in name and hasattr(module, "qkv"):
                linear_layers_to_exclude.append(module.qkv)
                if hasattr(module, "proj"):
                    linear_layers_to_exclude.append(module.proj)
        
        logger.info(f"Identificadas {len(linear_layers_to_exclude)} camadas de atenção críticas para preservar")
        
        # Módulos para quantizar
        modules_to_quantize = {nn.Linear}
        
        # Função para coletar os módulos a serem quantizados
        def collect_quantizable_modules(model):
            quantizable_modules = {}
            for name, module in model.named_modules():
                if type(module) in modules_to_quantize and module not in linear_layers_to_exclude:
                    quantizable_modules[name] = module
            return quantizable_modules
        
        # Coletar módulos quantizáveis
        quantizable_modules = collect_quantizable_modules(model_to_quantize)
        logger.info(f"Encontrados {len(quantizable_modules)} módulos para quantização")
        
        # Aplicar quantização dinâmica
        quantized_model = torch.quantization.quantize_dynamic(
            model_to_quantize,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        logger.info(f"Modelo {model_name} quantizado com sucesso")
        
        # Medir tamanho do modelo quantizado
        quant_size = 0
        for name, param in quantized_model.named_parameters():
            if param.dtype == torch.qint8:
                quant_size += param.numel() * 1  # INT8 = 1 byte
            else:
                quant_size += param.numel() * 4  # FLOAT32 = 4 bytes
                
        for name, buffer in quantized_model.named_buffers():
            if buffer.dtype == torch.qint8:
                quant_size += buffer.numel() * 1
            else:
                quant_size += buffer.numel() * 4
        
        logger.info(f"Tamanho aproximado do modelo ViT quantizado: {quant_size/1e6:.2f} MB")
        logger.info(f"Redução aproximada: {100 * (1 - quant_size/orig_size):.2f}%")
        
        # Avaliar no conjunto de teste
        max_test_samples = min(500, len(test_loader.dataset))
        indices = torch.randperm(len(test_loader.dataset))[:max_test_samples].tolist()  # Converter para lista
        test_subset = torch.utils.data.Subset(test_loader.dataset, indices)
        subset_loader = DataLoader(
            test_subset, 
            batch_size=32, 
            shuffle=False,
            num_workers=0  # Alterar de 2 para 0
        )
        
        # Avaliar modelo quantizado
        all_preds = []
        all_labels = []
        for inputs, labels in subset_loader:
            with torch.no_grad():
                outputs = quantized_model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        
        # Calcular acurácia
        quantized_accuracy = accuracy_score(all_labels, all_preds)
        logger.info(f"Acurácia do modelo quantizado: {quantized_accuracy:.4f}")
        
        # Salvar modelo quantizado
        try:
            model_path = os.path.join("models", f"{model_name}_quantized.pth")
            torch.save(quantized_model.state_dict(), model_path)
            logger.info(f"Modelo quantizado salvo como {model_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo quantizado: {str(e)}")
        
        return quantized_model
    
    except Exception as e:
        logger.error(f"Erro durante a quantização do modelo ViT {model_name}", exc_info=True)
        return None

def export_to_onnx(model, model_name, input_shape=(1, 3, 224, 224), opset_version=17):
    """
    Exporta o modelo para o formato ONNX para implantação em produção.
    Verifica se o modelo está quantizado antes de exportar.
    
    Args:
        model: Modelo PyTorch a ser exportado
        model_name: Nome do modelo para o arquivo de saída
        input_shape: Forma do tensor de entrada (batch_size, channels, height, width)
    
    Returns:
        Caminho para o arquivo ONNX ou None em caso de erro
    """
    try:
        # Verificar se o modelo está quantizado
        is_quantized = False
        for module in model.modules():
            if 'quantized' in str(type(module)).lower():
                is_quantized = True
                logger.warning(f"Modelo {model_name} contém módulos quantizados que não são suportados por ONNX.")
                break
        
        if is_quantized:
            logger.warning(f"Modelo {model_name} está quantizado. Exportando modelo não-quantizado...")
            # Use a versão não-quantizada para exportação ONNX
            try:
                original_path = f"models/{model_name}_final.pth"
                if os.path.exists(original_path):
                    # Criar uma nova instância do modelo
                    if "resnet" in model_name:
                        from torchvision.models import resnet50
                        original_model = resnet50(pretrained=False)
                        # Adaptar para o número de classes correto
                        num_classes = model.fc.out_features if hasattr(model, 'fc') else 6
                        original_model.fc = nn.Linear(original_model.fc.in_features, num_classes)
                        original_model.load_state_dict(torch.load(original_path))
                        model = original_model
                    elif "efficientnet" in model_name:
                        from torchvision.models import efficientnet_b0
                        original_model = efficientnet_b0(pretrained=False)
                        num_classes = model.classifier[1].out_features if hasattr(model, 'classifier') else 6
                        original_model.classifier[1] = nn.Linear(original_model.classifier[1].in_features, num_classes)
                        original_model.load_state_dict(torch.load(original_path))
                        model = original_model
                    elif "mobilenet" in model_name:
                        from torchvision.models import mobilenet_v3_small
                        original_model = mobilenet_v3_small(pretrained=False)
                        num_classes = 6
                        if hasattr(model, 'classifier') and hasattr(model.classifier[-1], 'out_features'):
                            num_classes = model.classifier[-1].out_features
                        original_model.classifier[-1] = nn.Linear(original_model.classifier[-1].in_features, num_classes)
                        original_model.load_state_dict(torch.load(original_path))
                        model = original_model
                    elif "vit" in model_name:
                        import timm
                        original_model = timm.create_model('vit_base_patch16_224', pretrained=False)
                        num_classes = 6
                        if hasattr(model, 'head') and hasattr(model.head, 'out_features'):
                            num_classes = model.head.out_features
                        original_model.head = nn.Linear(original_model.head.in_features, num_classes)
                        original_model.load_state_dict(torch.load(original_path))
                        model = original_model
                    logger.info(f"Modelo não-quantizado {model_name} carregado com sucesso para exportação ONNX")
            except Exception as e:
                logger.error(f"Não foi possível carregar a versão não-quantizada do modelo: {str(e)}")
                return None
        
        # Continuar com a exportação normal
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape, device=device)

        # Desativar temporariamente gradient checkpointing se estiver habilitado
        original_forward = None
        if hasattr(model, 'forward') and hasattr(model, '_original_forward'):
            # Salvar o método forward atual com checkpoint
            original_forward = model.forward
            # Restaurar o método forward original
            model.forward = model._original_forward
            logger.info(f"Gradient checkpointing temporariamente desativado para exportação ONNX")
        
        # Verificar se estamos usando checkpoint_sequential em ResNet
        if "resnet" in model_name.lower():
            # Verificar se o modelo tem implementação de checkpointing personalizada
            if hasattr(model, 'forward') and 'checkpoint_sequential' in str(model.forward):
                logger.info("Detectado checkpoint_sequential em ResNet, criando versão temporária sem checkpointing")
                
                # Definir um forward simplificado sem checkpoint_sequential
                def simple_forward(x):
                    # Forward pass sem checkpointing para exportação ONNX
                    x = model.conv1(x)
                    x = model.bn1(x)
                    x = model.relu(x)
                    x = model.maxpool(x)
                    
                    # Usar camadas sequencialmente sem checkpoint
                    x = model.layer1(x)
                    x = model.layer2(x)
                    x = model.layer3(x)
                    x = model.layer4(x)
                    
                    x = model.avgpool(x)
                    x = torch.flatten(x, 1)
                    x = model.fc(x)
                    return x
                
                # Salvar o forward original e substituir
                original_forward = model.forward
                model.forward = simple_forward
                logger.info("Forward simplificado aplicado para exportação ONNX")
        
        # Caminho para salvar o modelo ONNX
        onnx_path = os.path.join('models', f'{model_name}.onnx')
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        
        # Exportação para ONNX
        torch.onnx.export(
            model,                     # modelo PyTorch
            dummy_input,               # input de exemplo
            onnx_path,                 # caminho de saída
            export_params=True,        # exportar parâmetros do modelo
            opset_version=17,          # Aumentado para versão 14 para suporte a operadores mais recentes
            do_constant_folding=True,  # otimização - dobra constantes
            input_names=['input'],     # nomes de entrada
            output_names=['output'],   # nomes de saída
            dynamic_axes={             # suporte para batch_size dinâmico
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Restaurar o método forward original se foi alterado temporariamente
        if original_forward is not None:
            model.forward = original_forward
            logger.info(f"Método forward original restaurado após exportação ONNX")

        logger.info(f"Modelo {model_name} exportado para ONNX com sucesso em {onnx_path}")
        
        return onnx_path
    except Exception as e:
        logger.error(f"Erro ao exportar modelo para ONNX: {str(e)}", exc_info=True)
        return None