import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import logging

# Obter o logger
logger = logging.getLogger("landscape_classifier")


def custom_collate_fn(batch):
    """
    Função personalizada para collate que gerencia tipos e formatos inconsistentes.
    Resolve os erros:
    - 'stack expects each tensor to be equal size'
    - 'int' object has no attribute 'numel'
    
    Args:
        batch: Lote de amostras do dataset
        
    Returns:
        tuple: (imagens, rótulos) como tensores
    """
    images = []
    labels = []
    
    # Filtrar itens inválidos
    filtered_batch = []
    for sample in batch:
        # Verificar se a amostra tem o formato esperado
        if not isinstance(sample, tuple) or len(sample) != 2:
            logger.warning(f"Amostra com formato inválido ignorada: {type(sample)}")
            continue
            
        image, label = sample
        
        # Verificar e validar a imagem (deve ser um tensor 3D com canais RGB)
        if not isinstance(image, torch.Tensor):
            logger.warning(f"Imagem não é um tensor: {type(image)}")
            continue
            
        if image.dim() != 3 or image.size(0) != 3:
            logger.warning(f"Tensor de imagem com formato incorreto: {image.shape}")
            continue
            
        # Verificar e normalizar o rótulo
        if isinstance(label, int):
            # Converter inteiros para tensores escalares
            label = torch.tensor([label], dtype=torch.long)
        elif isinstance(label, torch.Tensor):
            # Garantir que o tensor tem formato adequado
            if label.dim() == 0:  # Tensor escalar
                label = label.unsqueeze(0)  # Converter para tensor de dim 1
            elif label.numel() != 1:
                # Para tensores com múltiplos valores, usar apenas o primeiro
                logger.warning(f"Rótulo com múltiplos valores: {label}, usando o primeiro")
                if label.dim() > 0 and label.numel() > 0:
                    label = label[0].unsqueeze(0)
                else:
                    logger.warning(f"Rótulo inválido ignorado: {label}")
                    continue
            
            # Garantir o tipo correto
            label = label.to(dtype=torch.long)
        else:
            # Tentar converter outros tipos para tensor
            try:
                label = torch.tensor([label], dtype=torch.long)
            except Exception as e:
                logger.warning(f"Não foi possível converter rótulo para tensor: {type(label)}, erro: {str(e)}")
                continue
                
        # Adicionar amostra válida
        filtered_batch.append((image, label))
    
    # Se o batch ficou vazio após a filtragem
    if not filtered_batch:
        logger.warning("Batch vazio após filtragem, gerando batch padrão vazio")
        return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
    
    # Extrair imagens e rótulos do batch filtrado
    for image, label in filtered_batch:
        images.append(image)
        labels.append(label)
    
    # Stack das imagens (devem ter o mesmo tamanho)
    images = torch.stack(images)
    
    # Para os rótulos, usamos cat em vez de stack para evitar problemas de dimensionalidade
    labels = torch.cat(labels)
    
    return images, labels


class CachedDataset(Dataset):
    """Dataset que implementa cache em memória para acesso mais rápido"""

    def __init__(self, base_dataset, cache_size=1000):
        self.base_dataset = base_dataset
        self.cache_size = min(cache_size, len(base_dataset))
        self.cache = {}

        # Pré-carregar os primeiros elementos mais acessados
        logger.info(f"Pré-carregando {self.cache_size} elementos no cache...")
        indices = torch.randperm(len(base_dataset))[:self.cache_size]
        for idx in tqdm(indices.tolist()):
            self.cache[idx] = self.base_dataset[idx]

        logger.info(f"Cache inicializado com {len(self.cache)} elementos")

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        item = self.base_dataset[idx]

        # Se o cache estiver cheio, remover um item aleatório
        if len(self.cache) >= self.cache_size:
            key_to_remove = random.choice(list(self.cache.keys()))
            del self.cache[key_to_remove]

        # Adicionar ao cache
        self.cache[idx] = item
        return item

    def __len__(self):
        return len(self.base_dataset)