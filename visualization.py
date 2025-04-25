"""
Módulo para visualização e interpretabilidade de modelos de deep learning.

Este módulo contém implementações de técnicas de visualização como Grad-CAM 
para CNNs e visualização de mapas de atenção para transformers, permitindo
interpretar como os modelos tomam decisões.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import logging
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
import random

# Obter o logger configurado
logger = logging.getLogger("landscape_classifier")

class GradCAM:
    """
    Implementação da técnica Grad-CAM para interpretabilidade de CNNs.
    
    A técnica Grad-CAM usa os gradientes que fluem para a última camada convolucional
    para produzir um mapa de calor grosseiro que destaca as regiões importantes da imagem
    para a predição que o modelo faz.
    
    Atributos:
        model: Modelo CNN PyTorch
        target_layer: Camada alvo para a qual os gradientes fluirão
        gradients: Gradientes armazenados da backward pass
        activations: Ativações armazenadas da forward pass
        handle_fwd: Handle para o hook da forward pass
        handle_bwd: Handle para o hook da backward pass
    """
    
    def __init__(self, model, target_layer):
        """
        Inicializa o GradCAM com um modelo e uma camada alvo.
        
        Args:
            model: Modelo CNN PyTorch
            target_layer: Camada convolucional alvo para calcular o GradCAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registrar hooks
        self.handle_fwd = target_layer.register_forward_hook(self.save_activation)
        self.handle_bwd = target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Salva as ativações da forward pass."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Salva os gradientes da backward pass."""
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Gera um mapa de calor para uma imagem de entrada.
        
        Args:
            input_tensor: Tensor da imagem de entrada
            target_class: Índice da classe alvo (usa argmax se None)
            
        Returns:
            numpy.ndarray: Mapa de calor normalizado
        """
        # Forward pass
        model_output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(model_output, dim=1).item()
        
        # Zerar gradientes
        self.model.zero_grad()
        
        # Backpropagation para a classe target
        model_output[0, target_class].backward(retain_graph=True)
        
        # Computar pesos através da média global dos gradientes
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Ponderar os canais de ativação pelo gradiente
        for i in range(pooled_gradients.shape[0]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        # Média sobre os canais e normalização
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = nn.functional.relu(heatmap)
        
        # Normalização
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        
        return heatmap.detach().cpu().numpy()
    
    def remove_hooks(self):
        """Remove os hooks para evitar memory leaks."""
        self.handle_fwd.remove()
        self.handle_bwd.remove()

def show_gradcam(model, input_img, input_tensor, target_layer, target_class=None, alpha=0.5, save_path=None):
    """
    Mostra a visualização do Grad-CAM e opcionalmente salva.
    
    Args:
        model: Modelo CNN PyTorch
        input_img: Imagem original em formato numpy (opcional se input_tensor fornecido)
        input_tensor: Tensor da imagem de entrada para o modelo
        target_layer: Camada convolucional alvo para calcular o GradCAM
        target_class: Índice da classe alvo (usa argmax se None)
        alpha: Intensidade de sobreposição do mapa de calor (0-1)
        save_path: Caminho para salvar a visualização (opcional)
        
    Returns:
        numpy.ndarray: Imagem com o mapa de calor sobreposto
    """
    try:
        # Instanciar GradCAM
        grad_cam = GradCAM(model, target_layer)
        
        # Gerar heatmap
        heatmap = grad_cam.generate_heatmap(input_tensor, target_class)
        
        # Redimensionar heatmap para o tamanho da imagem
        if input_img is None:
            # Normalizar a imagem do tensor
            input_img = input_tensor.cpu().squeeze().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            input_img = std * input_img + mean
            input_img = np.clip(input_img, 0, 1)
        
        heatmap = cv2.resize(heatmap, (input_img.shape[1], input_img.shape[0]))
        
        # Converter para mapa de calor
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Sobrepor heatmap na imagem
        superimposed_img = heatmap * alpha + input_img * 255 * (1 - alpha)
        superimposed_img = np.uint8(superimposed_img)
        
        # Remover hooks para evitar memory leaks
        grad_cam.remove_hooks()
        
        # Salvar se caminho for fornecido
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.imsave(save_path, superimposed_img / 255.0)
            logger.debug(f"Visualização Grad-CAM salva em: {save_path}")
        
        return superimposed_img / 255.0
    except Exception as e:
        logger.error(f"Erro ao gerar visualização Grad-CAM: {str(e)}", exc_info=True)
        # Retornar uma imagem vazia em caso de erro
        if input_img is not None:
            return input_img
        else:
            return np.zeros((224, 224, 3))

def visualize_gradcam_batch(model, test_loader, target_layer, classes, writer=None, model_name="model", num_images=10):
    """
    Gera visualizações Grad-CAM para um lote de imagens e adiciona ao TensorBoard.
    
    Args:
        model: Modelo CNN PyTorch
        test_loader: DataLoader para o conjunto de teste
        target_layer: Camada convolucional alvo para calcular o GradCAM
        classes: Lista de nomes das classes
        writer: SummaryWriter do TensorBoard (opcional)
        model_name: Nome do modelo para logging
        num_images: Número de imagens para visualizar
        
    Returns:
        bool: True se a visualização foi bem-sucedida
    """
    logger.info(f"Gerando visualizações Grad-CAM para o modelo {model_name}")
    
    try:
        # Colocar modelo em modo de avaliação
        model.eval()
        device = next(model.parameters()).device
        
        # Criar diretório para salvar imagens
        save_dir = os.path.join("gradcam_" + model_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Obter algumas imagens do conjunto de teste
        test_iter = iter(test_loader)
        batch = next(test_iter)
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
        # Selecionar imagens para visualização
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        
        # Selecionar algumas imagens corretas e algumas incorretas para melhor análise
        correct_idx = (preds == labels).nonzero(as_tuple=True)[0]
        incorrect_idx = (preds != labels).nonzero(as_tuple=True)[0]
        
        logger.info(f"Imagens corretamente classificadas: {len(correct_idx)} de {len(labels)}")
        
        # Preparar índices para visualização
        correct_to_visualize = min(num_images // 2, len(correct_idx))
        incorrect_to_visualize = min(num_images - correct_to_visualize, len(incorrect_idx))
        
        visualization_indices = list(correct_idx[:correct_to_visualize].cpu().numpy())
        visualization_indices.extend(list(incorrect_idx[:incorrect_to_visualize].cpu().numpy()))
        
        fig = plt.figure(figsize=(15, 10))
        for i, idx in enumerate(visualization_indices):
            if i >= num_images:  # Limitar ao número máximo de imagens
                break
                
            # Obter imagem e informações de classificação
            img_tensor = images[idx:idx+1]
            label = labels[idx].item()
            pred = preds[idx].item()
            status = "correct" if label == pred else "incorrect"
            
            # Converter tensor para numpy para visualização
            img = img_tensor.cpu().squeeze().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Gerar visualização Grad-CAM
            save_path = os.path.join(save_dir, f"gradcam_{i}_{classes[label]}_{status}.png")
            gradcam_img = show_gradcam(model, img, img_tensor, target_layer, target_class=label, save_path=save_path)
            
            # Adicionar à figura
            plt.subplot(2, num_images // 2, i + 1)
            plt.imshow(gradcam_img)
            plt.title(f"Real: {classes[label]}\nPred: {classes[pred]}", color="green" if label == pred else "red")
            plt.axis("off")
            
            # Adicionar ao TensorBoard
            if writer:
                writer.add_image(f"{model_name}/GradCAM/{i}_{classes[label]}_{status}", 
                              np.transpose(gradcam_img, (2, 0, 1)), 0)
        
        plt.tight_layout()
        summary_path = os.path.join(save_dir, f"gradcam_summary_{model_name}.png")
        plt.savefig(summary_path)
        logger.info(f"Resumo das visualizações Grad-CAM salvo em {summary_path}")
        
        if writer:
            writer.add_figure(f"{model_name}/GradCAM_summary", fig, 0)
        
        plt.close(fig)
        return True
    except Exception as e:
        logger.error(f"Erro ao gerar visualizações Grad-CAM: {str(e)}", exc_info=True)
        return False
    
class AttentionMapExtractor:
    """
    Implementação para extrair mapas de atenção de modelos Vision Transformer.
    
    Esta classe registra hooks para extrair os mapas de atenção de modelos baseados
    em transformers, permitindo visualizar como o modelo 'atende' a diferentes partes
    da imagem.
    
    Atributos:
        model: Modelo ViT PyTorch
        attention_maps: Mapas de atenção extraídos
        handles: Lista de handles para hooks
        num_tokens: Número de tokens no transformer
        num_heads: Número de cabeças de atenção
    """
    
    def __init__(self, model):
        """
        Inicializa o extrator de mapas de atenção.
        
        Args:
            model: Modelo Vision Transformer PyTorch
        """
        self.model = model
        self.attention_maps = None
        self.handles = []
        self.num_tokens = None
        self.num_heads = None
        
        # Registrar hooks para capturar os mapas de atenção
        for name, module in self.model.named_modules():
            if "attn" in name and hasattr(module, "qkv"):
                # Registra hook para o bloco de atenção
                handle = module.register_forward_hook(self.save_attention_map)
                self.handles.append(handle)
    
    def save_attention_map(self, module, input, output):
        """Salva os mapas de atenção durante a forward pass."""
        # Método melhorado para extrair os mapas de atenção
        if hasattr(module, "_attn_map"):
            # Algumas implementações armazenam o mapa de atenção diretamente
            self.attention_maps = module._attn_map
        else:
            # Se não tiver o atributo, calculamos a partir da saída ou dos pesos
            try:
                # Para timm ViT: extrai QKV e calcula a atenção
                if hasattr(module, "qkv") and hasattr(module, "scale"):
                    B = input[0].shape[0]  # Batch size
                    N = input[0].shape[1]  # Número de tokens (patches + cls)
                    
                    # Para timm ViT implementações
                    qkv = module.qkv(input[0])
                    
                    # Obtém dimensões
                    if hasattr(module, "num_heads"):
                        self.num_heads = module.num_heads
                    else:
                        self.num_heads = 12  # Valor padrão para vit_base
                    
                    # Reorganizar qkv para separar q, k, v
                    if len(qkv.shape) == 3:  # [B, N, 3*C]
                        head_dim = qkv.shape[-1] // (3 * self.num_heads)
                        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
                        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, C]
                        q, k, v = qkv[0], qkv[1], qkv[2]
                    
                        # Calcular atenção
                        scale = head_dim ** -0.5 if not hasattr(module, "scale") else module.scale
                        attn = (q @ k.transpose(-2, -1)) * scale
                        attn = attn.softmax(dim=-1)  # [B, H, N, N]
                        
                        # Salvar mapa de atenção - média entre as cabeças
                        self.attention_maps = attn.mean(dim=1)  # [B, N, N]
                        self.num_tokens = N
                    else:
                        logger.warning(f"Formato QKV não reconhecido: {qkv.shape}")
            except Exception as e:
                logger.error(f"Erro ao extrair mapa de atenção: {str(e)}", exc_info=True)
                self.attention_maps = None
    
    def remove_hooks(self):
        """Remove os hooks para evitar memory leaks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

def process_attention_map(attn_map, token_size_without_cls):
    """
    Processa um mapa de atenção para visualização, tratando erros de dimensão.
    
    Args:
        attn_map: Mapa de atenção tensor
        token_size_without_cls: Número de tokens sem o token CLS
    
    Returns:
        numpy.ndarray: Mapa de atenção processado
    """
    try:
        # Extrair o mapa de atenção sem o token CLS
        if attn_map.dim() >= 3:  # [batch, tokens, tokens]
            attn_slice = attn_map[0, 1:, 1:].cpu().numpy()
        else:  # [tokens, tokens]
            attn_slice = attn_map[1:, 1:].cpu().numpy()
            
        # Calcular o tamanho aproximado para visualização
        side_length = int(np.sqrt(attn_slice.size))
        
        # Redimensionar para o tamanho calculado independentemente do formato original
        processed_map = cv2.resize(attn_slice, (side_length, side_length))
        
        return processed_map
    except Exception as e:
        logger.warning(f"Erro ao processar mapa de atenção: {str(e)}. Retornando formato alternativo.")
        # Em caso de erro, retornar uma matriz vazia ou um formato alternativo
        return np.zeros((14, 14))  # Tamanho padrão para visualização

def visualize_attention_maps(model, test_loader, classes, writer=None, model_name="vit", num_images=8):
    """
    Gera visualizações dos mapas de atenção para um modelo Vision Transformer.
    
    Args:
        model: Modelo Vision Transformer PyTorch
        test_loader: DataLoader para o conjunto de teste
        classes: Lista de nomes das classes
        writer: SummaryWriter do TensorBoard (opcional)
        model_name: Nome do modelo para logging
        num_images: Número de imagens para visualizar
        
    Returns:
        bool: True se a visualização foi bem-sucedida
    """
    logger.info(f"Gerando visualizações de mapas de atenção para o modelo {model_name}...")
    
    try:
        # Criar diretório para salvar imagens
        save_dir = os.path.join("attention_maps_" + model_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Inicializar o extrator de mapas de atenção
        extractor = AttentionMapExtractor(model)
        
        # Obter algumas imagens do conjunto de teste
        device = next(model.parameters()).device
        test_iter = iter(test_loader)
        batch = next(test_iter)
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Criar um subconjunto para visualização com configurações otimizadas
        try:
            # Determinar índices para visualização adequada
            batch_size = min(num_images * 2, 16)  # Dobrar para ter mais opções de seleção
    
            # Verificar se teste_loader tem dataset com método de índice
            if hasattr(test_loader.dataset, '__getitem__'):
                # Criar um subconjunto específico para visualização, com menos amostras
                from torch.utils.data import Subset
        
                # Selecionar alguns índices do conjunto de teste
                import random
                subset_indices = random.sample(range(len(test_loader.dataset)), 
                                      min(batch_size * 4, len(test_loader.dataset)))
        
                test_subset = Subset(test_loader.dataset, subset_indices)
        
                # Criar um DataLoader específico para visualizações
                subset_loader = DataLoader(
                    test_subset, 
                    batch_size=batch_size, 
                    shuffle=False, 
                    num_workers=2,
                    pin_memory=True if device.type == 'cuda' else False,
                    collate_fn=getattr(test_loader, 'collate_fn', None),
                    multiprocessing_context='spawn'
                )
        
                # Substituir o batch pelas amostras do subset_loader
                logger.info(f"Usando subset_loader otimizado para visualização de atenção")
                subset_iter = iter(subset_loader)
                batch = next(subset_iter)
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
        except Exception as subset_error:
            # Se falhar, manter o comportamento original
            logger.warning(f"Não foi possível criar subset_loader: {str(subset_error)}. Usando test_loader padrão.")
        
        # Colocar o modelo em modo de avaliação
        model.eval()
        
        # Executar inferência para obter previsões e ativar os hooks
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        
        # Garantir que os tensores tenham dimensões compatíveis
        if preds.shape != labels.shape:
            logger.warning(f"Dimensões incompatíveis: preds {preds.shape}, labels {labels.shape}")
            # Tentar redimensionar as previsões para corresponder às labels
            if preds.dim() > labels.dim():
                # Reduzir as dimensões extras
                if preds.shape[0] == labels.shape[0]:
                    preds = preds.reshape(labels.shape)
    
            # Se ainda houver incompatibilidade
            if preds.shape != labels.shape:
                logger.error("Não foi possível resolver incompatibilidade de dimensões")
                # Criar índices alternativos para visualização
                correct_idx = torch.arange(min(4, len(labels)))
                incorrect_idx = torch.arange(min(4, len(labels)))
            else:
                correct_idx = (preds == labels).nonzero(as_tuple=True)[0]
                incorrect_idx = (preds != labels).nonzero(as_tuple=True)[0]
        # Selecionar algumas imagens corretas e algumas incorretas para melhor análise
        else:
            correct_idx = (preds == labels).nonzero(as_tuple=True)[0]
            incorrect_idx = (preds != labels).nonzero(as_tuple=True)[0]
        
        logger.info(f"Imagens corretamente classificadas: {len(correct_idx)} de {len(labels)}")
        
        # Preparar índices para visualização
        correct_to_visualize = min(num_images // 2, len(correct_idx))
        incorrect_to_visualize = min(num_images - correct_to_visualize, len(incorrect_idx))
        
        visualization_indices = list(correct_idx[:correct_to_visualize].cpu().numpy())
        visualization_indices.extend(list(incorrect_idx[:incorrect_to_visualize].cpu().numpy()))
        
        # Obter os mapas de atenção
        # Nota: Implementação para extrair atenção varia conforme a biblioteca específica do ViT
        # Esta é uma abordagem genérica
        fig = plt.figure(figsize=(15, 10))
        
        # Realizar visualização para cada imagem
        for i, idx in enumerate(visualization_indices):
            if i >= num_images:
                break
                
            # Obter a imagem e executar novamente para garantir os mapas de atenção
            img_tensor = images[idx:idx+1]
            label = labels[idx].item()
            pred = preds[idx].item()
            status = "correct" if label == pred else "incorrect"
            
            # Converter tensor para numpy para visualização
            img = img_tensor.cpu().squeeze().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Executar novamente a inferência para esta imagem específica
            with torch.no_grad():
                _ = model(img_tensor)
            
            # Processar mapas de atenção - NOTA: Esta parte pode variar conforme a implementação do ViT
            attn_map = None
            if hasattr(extractor, 'attention_maps') and extractor.attention_maps is not None:
                # Pegar o último mapa de atenção (último cabeçote)
                attn_map = extractor.attention_maps[-1] if isinstance(extractor.attention_maps, list) else extractor.attention_maps
                
                # Reshape para o formato da imagem e redimensionar
                # Os mapas de atenção do ViT costumam ser para patches, precisamos redimensionar
                if extractor.num_tokens:
                    # Verificar o tamanho real do tensor
                    token_size = attn_map.shape[-1]
                    # Calcular o número real de patches (subtraindo o token CLS se aplicável)
                    token_size_without_cls = token_size - 1
                    
                    # Usar a função process_attention_map para processar o mapa de atenção de forma segura
                    try:
                        processed_attn_map = process_attention_map(attn_map, token_size_without_cls)
                        
                        # Redimensionar para o tamanho da imagem original
                        attn_map = cv2.resize(processed_attn_map, (img.shape[1], img.shape[0]))
                    except Exception as reshape_error:
                        logger.warning(f"Erro ao redimensionar mapa de atenção: {str(reshape_error)}. Usando formato alternativo.")
                        # Criar um mapa de atenção padrão em caso de falha
                        attn_map = np.zeros((img.shape[0], img.shape[1]))
                    
                    # Normalizar o mapa para visualização
                    if np.ptp(attn_map) > 0:  # Verificar se o mapa não é constante
                        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                    
                    # Criar overlay colorido
                    heatmap = np.uint8(255 * attn_map)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    
                    # Sobrepor mapa de atenção na imagem
                    alpha = 0.6
                    superimposed_img = heatmap * alpha + img * 255 * (1 - alpha)
                    superimposed_img = np.uint8(superimposed_img)
                    
                    # Salvar imagem
                    save_path = os.path.join(save_dir, f"attention_{i}_{classes[label]}_{status}.png")
                    plt.imsave(save_path, superimposed_img / 255.0)
                    
                    # Adicionar à figura
                    plt.subplot(2, num_images // 2, i + 1)
                    plt.imshow(superimposed_img / 255.0)
                    plt.title(f"Real: {classes[label]}\nPred: {classes[pred]}", color="green" if label == pred else "red")
                    plt.axis("off")
                    
                    # Adicionar ao TensorBoard
                    if writer:
                        writer.add_image(f"{model_name}/AttentionMap/{i}_{classes[label]}_{status}", 
                                        np.transpose(superimposed_img / 255.0, (2, 0, 1)), 0)
            else:
                # Se não conseguimos extrair o mapa de atenção, apenas mostramos a imagem
                plt.subplot(2, num_images // 2, i + 1)
                plt.imshow(img)
                plt.title(f"Real: {classes[label]}\nPred: {classes[pred]} (Sem mapa)", 
                         color="green" if label == pred else "red")
                plt.axis("off")
                
                logger.warning(f"Não foi possível extrair o mapa de atenção para a imagem {i}")
        
        # Remover hooks para evitar memory leaks
        extractor.remove_hooks()
        
        # Salvar figura completa
        plt.tight_layout()
        summary_path = os.path.join(save_dir, f"attention_summary_{model_name}.png")
        plt.savefig(summary_path)
        logger.info(f"Resumo dos mapas de atenção salvo em {save_dir}/attention_summary_{model_name}.png")
        
        if writer:
            writer.add_figure(f"{model_name}/AttentionMap_summary", fig, 0)
        
        plt.close(fig)
        return True
    
    except Exception as e:
        logger.error(f"Erro ao gerar visualizações de mapas de atenção: {str(e)}", exc_info=True)
        return False